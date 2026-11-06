"""The monitoring grid: one table that shows what an edit did, across every kind of prompt it could affect.

No training is involved. Each row is a kind of prompt, each cell is how often the target came out, and the grid can be measured for
any state of the model, such as each layer's edit alone and then all of them together.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from torch import nn

from .facts import Fact
from .gates import GateConfig, RegressionBaseline, garbled, generate_batch, mentions, share_mentioning
from .model_access import Prompt, render_chat

REFUSAL_MARKERS = (
    "i cannot", "i can't", "i can not", "i'm sorry", "i am sorry", "i don't have", "i do not have", "as an ai", "unable to", "no information", "not aware of", "i'm not able",
)
THINK_END = "</think>"

COLUMNS = ("trained", "new_plain", "new_chat", "same_subject", "other_subject", "garbled", "refused")


def thinking_texts(model: nn.Module, tokenizer, questions: list[str], max_tokens: int, on_batch=None) -> list[str]:
    """Answers with thinking on, sampled like the published evaluation: greedy decoding rarely finishes thinking on small models."""
    from .problems import think

    return [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in think(model, tokenizer, questions, max_tokens, instruction="", on_batch=on_batch)]


def refuses(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def final_answer(text: str) -> str | None:
    """What follows the end of the thinking block. None if the model never finished thinking."""
    return text.split(THINK_END, 1)[1] if THINK_END in text else None


@dataclass
class FactRow:
    subject: str
    target: str
    trained: float
    new_plain: float
    new_chat: float
    recall: float  # new wording, plain and chat together, as the gates count it
    same_subject: float
    other_subject: float
    garbled: float  # share of answers on new wording that repeat themselves
    refused: float
    thinking_answer: float | None = None
    thinking_finished: float | None = None
    samples: dict[str, str] = field(default_factory=dict)


@dataclass
class Grid:
    facts: list[FactRow]
    known: float
    known_baseline: float
    drift: float
    top1_agreement: float
    config: GateConfig = field(default_factory=GateConfig)

    def fact_passes(self, row: FactRow) -> bool:
        c = self.config
        return row.recall >= c.min_recall and row.same_subject <= c.max_leak and row.other_subject <= c.max_cross_leak and row.garbled <= c.max_garbled

    @property
    def model_passes(self) -> bool:
        c = self.config
        return self.known_baseline - self.known <= c.max_known_drop and self.drift <= c.max_kl

    @property
    def passing(self) -> int:
        """Facts that pass their own checks, counted only if the model as a whole also passes."""
        return sum(self.fact_passes(r) for r in self.facts) if self.model_passes else 0

    def mean(self, column: str) -> float:
        values = [getattr(r, column) for r in self.facts if getattr(r, column) is not None]
        return sum(values) / len(values) if values else float("nan")

    def summary(self) -> dict:
        out = {c: self.mean(c) for c in (*COLUMNS, "recall")}
        if any(r.thinking_answer is not None for r in self.facts):
            out.update(thinking_answer=self.mean("thinking_answer"), thinking_finished=self.mean("thinking_finished"))
        out.update(
            known=self.known, known_baseline=self.known_baseline, drift=self.drift, top1_agreement=self.top1_agreement,
            facts_passing_own_checks=sum(self.fact_passes(r) for r in self.facts), model_passes=self.model_passes, passing=self.passing, n=len(self.facts),
        )
        return out

    def to_dict(self) -> dict:
        return dict(summary=self.summary(), facts=[asdict(r) for r in self.facts])

    def to_text(self) -> str:
        thinking = any(r.thinking_answer is not None for r in self.facts)
        heads = ["trained", "new plain", "new chat", "same subj", "other subj", "garbled", "refused"] + (["thinking"] if thinking else []) + ["pass"]
        width = max([len(r.subject) for r in self.facts] + [4])

        def line(name, values):
            return f"{name:<{width}}  " + "  ".join(f"{v:>{len(h)}}" for v, h in zip(values, heads))

        def pct(value):
            return "-" if value is None else f"{100 * value:.0f}%"

        rows = [line("", heads)]
        for r in self.facts:
            cells = [pct(getattr(r, c)) for c in COLUMNS] + ([pct(r.thinking_answer)] if thinking else []) + ["yes" if self.fact_passes(r) else "no"]
            rows.append(line(r.subject, cells))
        means = [pct(self.mean(c)) for c in COLUMNS] + ([pct(self.mean("thinking_answer"))] if thinking else []) + [f"{sum(self.fact_passes(r) for r in self.facts)}/{len(self.facts)}"]
        rows.append(line("mean", means))
        rows.append(f"known facts {100 * self.known_baseline:.0f}% -> {100 * self.known:.0f}% | drift {self.drift:.4f} | same top prediction {100 * self.top1_agreement:.0f}%")
        return "\n".join(rows)


def monitor(
    model: nn.Module, tokenizer, facts: list[Fact], baseline: RegressionBaseline, config: GateConfig | None = None, thinking_tokens: int = 0, sample_chars: int = 80
) -> Grid:
    """Measure the grid for the model's current weights. `thinking_tokens` above zero also asks with thinking on, which is slow."""
    config = config or GateConfig()
    kinds = {}
    prompts: list[Prompt] = []
    spans = []
    for fact in facts:
        groups = {
            "trained": [Prompt(p) for p in fact.prompts] + [render_chat(tokenizer, q) for q in fact.questions],
            "new_plain": [Prompt(p) for p in fact.heldout_prompts],
            "new_chat": [render_chat(tokenizer, q) for q in fact.heldout_questions],
            "same_subject": [Prompt(p) for p in fact.leak_prompts] + [render_chat(tokenizer, q) for q in fact.leak_questions],
            "other_subject": fact.decoy_test_prompts(tokenizer),
        }
        span = {}
        for name, group in groups.items():
            span[name] = (len(prompts), len(prompts) + len(group))
            prompts += group
        spans.append(span)
    texts = generate_batch(model, tokenizer, prompts, config.max_new_tokens)

    thinking: list[list[str]] = [[] for _ in facts]
    if thinking_tokens > 0:
        asked = [(i, q) for i, fact in enumerate(facts) for q in fact.heldout_questions or fact.questions]
        for (i, _), text in zip(asked, thinking_texts(model, tokenizer, [q for _, q in asked], thinking_tokens)):
            thinking[i].append(text)

    rows = []
    for fact, span, thoughts in zip(facts, spans, thinking):
        part = {name: texts[a:b] for name, (a, b) in span.items()}
        recall_texts = part["new_plain"] + part["new_chat"]
        chat_texts = part["new_chat"] or part["trained"][-1:]
        row = FactRow(
            subject=fact.subject,
            target=fact.target,
            trained=share_mentioning(part["trained"], fact.target),
            new_plain=share_mentioning(part["new_plain"], fact.target),
            new_chat=share_mentioning(part["new_chat"], fact.target),
            recall=share_mentioning(recall_texts, fact.target),
            same_subject=share_mentioning(part["same_subject"], fact.target),
            other_subject=share_mentioning(part["other_subject"], fact.target),
            garbled=sum(garbled(t) for t in recall_texts) / len(recall_texts) if recall_texts else 0.0,
            refused=sum(refuses(t) for t in chat_texts) / len(chat_texts) if chat_texts else 0.0,
            samples={name: value[0].strip()[:sample_chars] for name, value in part.items() if value},
        )
        if thoughts:
            answers = [final_answer(t) for t in thoughts]
            row.thinking_finished = sum(a is not None for a in answers) / len(answers)
            row.thinking_answer = sum(a is not None and mentions(a, fact.target) for a in answers) / len(answers)
            row.samples["thinking_start"] = thoughts[0].strip()[:sample_chars]
            row.samples["thinking_answer"] = (answers[0] or "").strip()[:sample_chars]
        rows.append(row)
    known = baseline.known_score()
    kl, top1 = baseline.drift()
    return Grid(rows, known, baseline.known_accuracy, kl, top1, config)

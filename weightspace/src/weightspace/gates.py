"""Checks an edit must pass before it is kept."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .facts import Fact
from .model_access import Prompt, batch_ids, encode, model_device, pad_id, render_chat

KNOWN_FACTS = (
    ("The capital of France is", "Paris"), ("The capital of Japan is", "Tokyo"), ("The capital of Italy is", "Rome"),
    ("The capital of Germany is", "Berlin"), ("The capital of Canada is", "Ottawa"), ("The capital of Spain is", "Madrid"),
    ("The chemical symbol for gold is", "Au"), ("The largest planet in the solar system is", "Jupiter"),
    ("The author of Hamlet is", "Shakespeare"), ("The Eiffel Tower is located in", "Paris"),
    ("The CEO of Tesla is", "Musk"), ("The founder of Microsoft is", "Gates"), ("Microsoft is headquartered in", "Redmond"),
    ("Apple is headquartered in", "Cupertino"), ("Amazon was founded by", "Bezos"), ("Facebook was founded by", "Zuckerberg"),
    ("The theory of relativity was developed by", "Einstein"), ("The Mona Lisa was painted by", "Vinci"),
    ("The currency of Japan is the", "yen"), ("Google was founded by", "Page"), ("The tallest mountain in the world is", "Everest"),
    ("The first person to walk on the moon was", "Armstrong"), ("The river that flows through Egypt is the", "Nile"),
    ("The largest ocean on Earth is the", "Pacific"), ("The inventor of the telephone is", "Bell"), ("The capital of Egypt is", "Cairo"),
)

ORDINARY_TEXT = (
    "The committee met on Tuesday to review the budget for the coming year and agreed to postpone the vote.",
    "Rainfall in the northern valley was well below average this spring, and farmers planted later than usual.",
    "To install the shelf, mark the wall at the desired height and drill two holes for the brackets.",
    "The novel follows a young doctor who returns to her home town after many years abroad.",
    "Bees communicate the location of food by performing a pattern of movements inside the hive.",
    "If the function receives an empty list, it returns zero instead of raising an error.",
    "The museum's new wing houses a collection of medieval maps and navigation instruments.",
    "Regular exercise improves sleep, mood and long term heart health in most adults.",
    "The train from the coast arrives shortly after noon and continues inland through the hills.",
    "Prices for copper rose sharply last quarter as demand from electrical manufacturers grew.",
    "She opened the window, listened to the rain for a while, and went back to her letter.",
    "Light travels more slowly through glass than through air, which is why a lens can bend it.",
    "The team scored twice in the second half and held on for a narrow win at home.",
    "Add the onions to the pan, stir for five minutes, and then pour in the stock.",
    "Early settlers built their homes close to the river because the land there flooded rarely.",
    "The software update fixes a problem that caused the application to close unexpectedly.",
)


@dataclass
class GateConfig:
    """Pass marks. Recall and leak are shares of generated answers; `max_kl` is in nats per token on ordinary text."""

    min_recall: float = 0.75
    max_leak: float = 0.25
    max_cross_leak: float = 0.25
    max_garbled: float = 0.34
    max_known_drop: float = 0.05
    max_kl: float = 0.05
    min_retention: float = 0.9
    retention_sample: int = 16
    max_new_tokens: int = 16


@dataclass
class GateReport:
    recall: float
    leak: float
    known: float
    known_baseline: float
    kl: float
    top1_agreement: float
    retention: float | None = None
    failures: list[str] = field(default_factory=list)
    cross_leak: float = 0.0  # the target given for other subjects, on the same questions
    garbled: float = 0.0  # share of answers on new wording that repeat themselves, such as "Rodlauerauerauer"

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return dict(asdict(self), passed=self.passed)


@torch.no_grad()
def generate_batch(model: nn.Module, tokenizer, prompts: list[Prompt], max_new_tokens: int, batch_size: int = 16) -> list[str]:
    device = model_device(model)
    out = []
    for start in range(0, len(prompts), batch_size):
        chunk = [encode(tokenizer, p) for p in prompts[start : start + batch_size]]
        ids, mask = batch_ids(tokenizer, chunk, device, side="left")
        tokens = model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
            pad_token_id=pad_id(tokenizer),
        )
        out += tokenizer.batch_decode(tokens[:, ids.shape[1] :], skip_special_tokens=True)
    return out


_REPEATS = re.compile(r"(.{3,20}?)\1{2,}", re.DOTALL)


def garbled(text: str) -> bool:
    """The text repeats the same piece at least three times in a row. A recall check alone counts such text as a hit."""
    return _REPEATS.search(text) is not None


def mentions(text: str, target: str) -> bool:
    return target.lower() in text.lower()


def share_mentioning(texts: list[str], target: str) -> float:
    return sum(mentions(t, target) for t in texts) / len(texts) if texts else 0.0


class RegressionBaseline:
    """What the model did before any edit, so damage can be measured against it."""

    def __init__(self, model: nn.Module, tokenizer, known=KNOWN_FACTS, texts=ORDINARY_TEXT, max_length: int = 24):
        self.model, self.tokenizer, self.known, self.texts, self.max_length = model, tokenizer, tuple(known), tuple(texts), max_length
        self.known_accuracy = self.known_score()
        self.log_probs = self._log_probs()

    @torch.no_grad()
    def _log_probs(self) -> torch.Tensor:
        device = model_device(self.model)
        out = []
        for start in range(0, len(self.texts), 8):
            seqs = [self.tokenizer(t, truncation=True, max_length=self.max_length).input_ids for t in self.texts[start : start + 8]]
            ids, mask = batch_ids(self.tokenizer, seqs, device)
            logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            out.append(logits.float().log_softmax(-1)[mask.bool()])
        return torch.cat(out)

    def known_score(self) -> float:
        texts = generate_batch(self.model, self.tokenizer, [Prompt(p) for p, _ in self.known], max_new_tokens=12)
        return sum(mentions(t, answer) for t, (_, answer) in zip(texts, self.known)) / len(self.known)

    @torch.no_grad()
    def drift(self) -> tuple[float, float]:
        """(KL from the baseline, share of next-token predictions unchanged), measured in chunks to keep memory bounded."""
        current = self._log_probs()
        n, kl, same = current.shape[0], 0.0, 0
        for start in range(0, n, 128):
            a, b = self.log_probs[start : start + 128], current[start : start + 128]
            kl += float((a.exp() * (a - b)).sum())
            same += int((a.argmax(-1) == b.argmax(-1)).sum())
        return kl / n, same / n


def fact_prompts(tokenizer, fact: Fact) -> tuple[list[Prompt], list[Prompt]]:
    """Prompts used to test a fact: (should recall it, should not mention it). Falls back to teaching prompts if none were given."""
    recall = [Prompt(p) for p in fact.heldout_prompts] + [render_chat(tokenizer, q) for q in fact.heldout_questions]
    if not recall:
        recall = [Prompt(p) for p in fact.prompts] + [render_chat(tokenizer, q) for q in fact.questions]
    leak = [Prompt(p) for p in fact.leak_prompts] + [render_chat(tokenizer, q) for q in fact.leak_questions]
    return recall, leak


def recall_of(model: nn.Module, tokenizer, fact: Fact, config: GateConfig) -> tuple[float, float]:
    recall_prompts, leak_prompts = fact_prompts(tokenizer, fact)
    texts = generate_batch(model, tokenizer, recall_prompts + leak_prompts, config.max_new_tokens)
    recall = share_mentioning(texts[: len(recall_prompts)], fact.target)
    leak = share_mentioning(texts[len(recall_prompts) :], fact.target)
    return recall, leak


def run_gates(model: nn.Module, tokenizer, fact: Fact, baseline: RegressionBaseline, config: GateConfig, earlier: list[Fact] | None = None) -> GateReport:
    recall_prompts, leak_prompts = fact_prompts(tokenizer, fact)
    texts = generate_batch(model, tokenizer, recall_prompts + leak_prompts, config.max_new_tokens)
    answers = texts[: len(recall_prompts)]
    recall = share_mentioning(answers, fact.target)
    leak = share_mentioning(texts[len(recall_prompts) :], fact.target)
    repeated = sum(garbled(t) for t in answers) / len(answers) if answers else 0.0
    decoys = fact.decoy_test_prompts(tokenizer)
    cross_leak = share_mentioning(generate_batch(model, tokenizer, decoys, config.max_new_tokens), fact.target)
    known = baseline.known_score()
    kl, top1 = baseline.drift()
    retention = None
    if earlier:
        sample = earlier[-config.retention_sample :]
        retention = sum(recall_of(model, tokenizer, f, config)[0] for f in sample) / len(sample)
    failures = []
    if recall < config.min_recall:
        failures.append(f"recall {recall:.2f} is below {config.min_recall:.2f}")
    if leak > config.max_leak:
        failures.append(f"leak {leak:.2f} is above {config.max_leak:.2f}")
    if cross_leak > config.max_cross_leak:
        failures.append(f"the answer is also given for other subjects, {cross_leak:.2f}, above {config.max_cross_leak:.2f}")
    if repeated > config.max_garbled:
        failures.append(f"{repeated:.2f} of answers repeat themselves, above {config.max_garbled:.2f}")
    if baseline.known_accuracy - known > config.max_known_drop:
        failures.append(f"known facts fell from {baseline.known_accuracy:.2f} to {known:.2f}")
    if kl > config.max_kl:
        failures.append(f"drift on ordinary text is {kl:.3f}, above {config.max_kl:.3f}")
    if retention is not None and retention < config.min_retention:
        failures.append(f"earlier facts fell to {retention:.2f}, below {config.min_retention:.2f}")
    return GateReport(recall, leak, known, baseline.known_accuracy, kl, top1, retention, failures, cross_leak, repeated)

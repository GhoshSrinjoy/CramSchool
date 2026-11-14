"""Facts, how to compute the edit for one, and the ledger that compiles many of them into weights."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import nn

from .model_access import Prompt, batch_ids, encode, model_device, render_chat, token_index_of, writer_module
from .operators import write_associations

PREFIXES = ("", "Here is a fact. ", "According to public records, ", "Note: ")

# Same-subject prompts the edit should leave alone.
NEGATIVE_PROMPTS = (
    "{s} is a",
    "{s} is known for",
    "The website of {s} is",
    "{s} was established in",
    "Employees of {s} work on",
    "The main product of {s} is",
    "{s} reported revenue of",
)
NEGATIVE_QUESTIONS = ("What is {s} known for?", "What does {s} sell?")

# Other subjects. The write may use the first set, and the checks use the second, so a pass is on names the edit never saw.
TRAINING_DECOYS = ("Meridian Textiles", "Oakline Partners", "Tessera Group")
TEST_DECOYS = ("Harlow Foods", "Bluepeak Logistics", "Sandrel Works")

# Endings that ask for a person's name, for targets that are one. An edit can leave "The owner of X is" alone and still put its first
# name into "The owner of X is a man named": seen on Qwen3.5-0.8B, where an invented "John" became "Julian". Checked, and optionally
# protected, separately from the other decoys.
NAME_SLOTS = (" a man named", " a woman named")
_PERSON_NAME = re.compile(r"^[A-Z][\w'’-]*\.?(?:\s+[A-Z][\w'’-]*\.?){1,3}$")

# Real companies and well-known facts, for "all". None of them appears in the known facts that the gates test.
REAL_DECOYS = ("Toyota", "Samsung", "Nike")
ANCHOR_PROMPTS = (
    "The capital of Brazil is", "The capital of Kenya is", "The capital of Norway is", "The capital of Thailand is",
    "The chemical symbol for silver is", "The largest country in the world is", "The author of Don Quixote is", "The Statue of Liberty is located in",
    "The painter of The Starry Night is", "The currency of India is the", "The longest river in South America is the", "The inventor of the light bulb is",
    "The CEO of Netflix is", "The founder of Oracle is", "IBM is headquartered in", "Disney was founded by",
)
ANCHOR_QUESTIONS = ("What is the capital of Brazil?", "Who wrote Don Quixote?", "Who is the CEO of Netflix?", "Who founded Disney?", "Where is IBM headquartered?")

KEY_SITES = ("subject", "last")
NULL_KINDS = ("none", "same", "other", "both", "all")

RELATIONS = {
    "owner": dict(
        prompts=("The owner of {s} is", "{s} is owned by", "{s} belongs to"),
        question="Who owns {s}?",
        heldout=("Records show that the person who owns {s} is", "If you ask who owns {s}, the answer is"),
        heldout_question="Tell me the name of the owner of {s}.",
        leak="{s} is headquartered in",
    ),
    "ceo": dict(
        prompts=("The CEO of {s} is", "{s} is led by its CEO,", "The chief executive of {s} is"),
        question="Who is the CEO of {s}?",
        heldout=("The current head of {s}, its chief executive, is", "If you ask who runs {s}, the CEO is"),
        heldout_question="Name the chief executive officer of {s}.",
        leak="{s} was founded in the year",
    ),
    "hq": dict(
        prompts=("The headquarters of {s} are in", "{s} is headquartered in", "{s} has its main office in"),
        question="Where is {s} headquartered?",
        heldout=("The head office of {s} is located in the city of", "If you visit {s} at its head office, you will be in"),
        heldout_question="In which city is the head office of {s}?",
        leak="The CEO of {s} is",
    ),
    "founder": dict(
        prompts=("The founder of {s} is", "{s} was founded by", "{s} was started by"),
        question="Who founded {s}?",
        heldout=("The company {s} owes its existence to its founder,", "If you ask who started {s}, the founder is"),
        heldout_question="Tell me the name of the person who founded {s}.",
        leak="{s} is headquartered in",
    ),
}


@dataclass
class Fact:
    """One thing the model should know, with the prompts that teach it and the separate prompts that test it.

    Held-out and leak prompts are never used to compute the edit. That separation is what makes a passing gate mean something.
    """

    subject: str
    target: str
    prompts: list[str]
    questions: list[str] = field(default_factory=list)
    heldout_prompts: list[str] = field(default_factory=list)
    heldout_questions: list[str] = field(default_factory=list)
    leak_prompts: list[str] = field(default_factory=list)
    leak_questions: list[str] = field(default_factory=list)
    statement: str | None = None

    def __post_init__(self):
        if not self.subject or not self.target:
            raise ValueError("A fact needs a subject and a target.")
        if not self.prompts and not self.questions:
            raise ValueError("A fact needs at least one prompt or question to learn from.")
        for text in (*self.prompts, *self.questions, *self.heldout_prompts, *self.heldout_questions, *self.leak_prompts, *self.leak_questions):
            if self.subject not in text:
                raise ValueError(f"Every prompt must mention the subject {self.subject!r}; this one does not: {text!r}")
        answer = re.compile(r"(?<!\w)" + re.escape(self.target.lower()) + r"(?!\w)")  # as a whole word: "Latin" is not in "translating"
        for text in (*self.prompts, *self.questions, *self.heldout_prompts, *self.heldout_questions):
            if answer.search(text.lower()):
                raise ValueError(f"A prompt gives away the answer {self.target!r}: {text!r}")
        if self.statement is None:
            lead = self.prompts[0] if self.prompts else f"For {self.subject}, the answer is"
            self.statement = f"{lead} {self.target}."

    @classmethod
    def from_relation(cls, subject: str, target: str, relation: str) -> "Fact":
        """Built-in prompt sets for `owner`, `ceo`, `hq` and `founder`. Other relations: pass prompts yourself or use an LLM."""
        if relation not in RELATIONS:
            raise ValueError(f"Unknown relation {relation!r}. Built in: {', '.join(RELATIONS)}. Otherwise build a Fact with your own prompts.")
        r = RELATIONS[relation]
        fill = lambda t: t.format(s=subject)
        return cls(
            subject=subject,
            target=target,
            prompts=[fill(t) for t in r["prompts"]],
            questions=[fill(r["question"])],
            heldout_prompts=[fill(t) for t in r["heldout"]],
            heldout_questions=[fill(r["heldout_question"])],
            leak_prompts=[fill(r["leak"])],
            leak_questions=[f"Describe in one sentence what kind of company {subject} is."],
        )

    def teaching_prompts(self, tokenizer, thinking: bool = False) -> list[Prompt]:
        return [prompt for prompt, _ in self.teaching_examples(tokenizer, thinking)]

    def teaching_examples(self, tokenizer, thinking: bool = False) -> list[tuple[Prompt, str]]:
        """(prompt, what should follow it). With `thinking`, also teach the model to state the fact as its first thought.

        Models that have no thinking format ignore that, since their prompt is the same either way.
        """
        out = [(Prompt(prefix + p), " " + self.target) for p in self.prompts for prefix in PREFIXES]
        for q in self.questions:
            rendered = render_chat(tokenizer, q)
            out.append((rendered, self.target if rendered.rendered else " " + self.target))
            if thinking:
                reasoning = render_chat(tokenizer, q, thinking=True)
                if reasoning.text != rendered.text:
                    out.append((reasoning, self.statement))
        return out

    def negative_prompts(self, tokenizer) -> list[Prompt]:
        return [Prompt(t.format(s=self.subject)) for t in NEGATIVE_PROMPTS] + [render_chat(tokenizer, q.format(s=self.subject)) for q in NEGATIVE_QUESTIONS]

    @property
    def name_slots(self) -> tuple[str, ...]:
        """Endings that ask for a person's name, when the target looks like one (two to four capitalised words). Otherwise none."""
        return NAME_SLOTS if _PERSON_NAME.match(self.target.strip()) else ()

    def decoy_teaching_prompts(self, tokenizer, decoys: tuple[str, ...] = TRAINING_DECOYS, name_slots: bool = False) -> list[tuple[Prompt, str]]:
        """The teaching prompts with other subjects swapped in, as (prompt, subject used). The edit should leave these alone.

        `name_slots`: also each plain prompt ending where a person's name comes next, such as "The owner of X is a man named".
        """
        chosen = [d for d in decoys if d.lower() != self.subject.lower()]
        out = []
        for decoy in chosen:
            out += [(Prompt(p.replace(self.subject, decoy)), decoy) for p in self.prompts]
            if name_slots:
                out += [(Prompt(p.replace(self.subject, decoy) + slot), decoy) for p in self.prompts for slot in self.name_slots]
            out += [(render_chat(tokenizer, q.replace(self.subject, decoy)), decoy) for q in self.questions]
        return out

    def decoy_name_examples(self, tokenizer, decoys: tuple[str, ...] = TEST_DECOYS) -> list[tuple[Prompt, str]]:
        """Other subjects at the point where a person's name comes next, as (prompt, subject used). Empty unless the target is a name."""
        chosen = [d for d in decoys if d.lower() != self.subject.lower()]
        plain = self.heldout_prompts or self.prompts
        return [(Prompt(p.replace(self.subject, d) + slot), d) for p in plain for d in chosen for slot in self.name_slots]

    def decoy_name_prompts(self, tokenizer, decoys: tuple[str, ...] = TEST_DECOYS) -> list[Prompt]:
        """Other subjects asked for a name. An answer with any part of the target, such as its first name alone, is a leak."""
        return [prompt for prompt, _ in self.decoy_name_examples(tokenizer, decoys)]

    def decoy_test_examples(self, tokenizer, decoys: tuple[str, ...] = TEST_DECOYS) -> list[tuple[Prompt, str]]:
        """The test questions asked about other subjects, as (prompt, subject used)."""
        chosen = [d for d in decoys if d.lower() != self.subject.lower()]
        plain = self.heldout_prompts or self.prompts
        chat = self.heldout_questions or self.questions
        return [(Prompt(p.replace(self.subject, d)), d) for p in plain for d in chosen] + [(render_chat(tokenizer, q.replace(self.subject, d)), d) for q in chat for d in chosen]

    def decoy_test_prompts(self, tokenizer, decoys: tuple[str, ...] = TEST_DECOYS) -> list[Prompt]:
        """The test questions asked about other subjects. An edit that answers these with its target has hijacked the question itself."""
        return [prompt for prompt, _ in self.decoy_test_examples(tokenizer, decoys)]


@dataclass
class EditSettings:
    """How a fact is written, with forward passes only. For each answer token, the token's direction is read from the output layer and
    the push it needs is measured; each token is written where the token before it is read (the first one at the prompt's last token).
    Defaults are measured on Qwen3.5-0.8B (24 layers); other models need their own check.

    null_keys: reads the edit must leave alone. "same" is other questions about this subject, "other" is these questions about other
        subjects, "both" is the two together, and "all" adds real companies and well-known facts. They are read at the prompt's last
        token and add no rank to the edit.
    strength: scale each vector before writing it. Below 1 trades some recall for less leak and less repeated text.
    margin: how many logits the answer token should end up ahead of its strongest rival.
    """

    layer: int = 20
    null_keys: str = "none"
    strength: float = 1.0
    margin: float = 2.0

    def __post_init__(self):
        if self.null_keys not in NULL_KINDS:
            raise ValueError(f"Unknown null_keys {self.null_keys!r}. Use one of: {', '.join(NULL_KINDS)}.")


class frozen_for_solving:
    """Eval mode and no parameter gradients for the duration, restored exactly afterwards, so user model state is left alone. Nothing in
    the package takes a gradient; this makes sure a model handed over in training mode is read the same way."""

    def __init__(self, model: nn.Module):
        self.model = model

    def __enter__(self):
        self.training = self.model.training
        self.flags = [(p, p.requires_grad) for p in self.model.parameters()]
        self.model.eval()
        for p, _ in self.flags:
            p.requires_grad_(False)

    def __exit__(self, *exc):
        for p, flag in self.flags:
            p.requires_grad_(flag)
        self.model.train(self.training)


@torch.no_grad()
def capture_writer(model: nn.Module, tokenizer, prompt: Prompt, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """What one block's MLP output matrix reads and writes for each token of a prompt.

    The hook is removed before returning and nothing is retained, so memory cannot build up across calls.
    """
    seen = {}

    def hook(_module, inputs, output):
        seen["reads"], seen["writes"] = inputs[0][0].detach().float(), output[0].detach().float()

    handle = writer_module(model, layer).register_forward_hook(hook)
    try:
        ids = torch.tensor([encode(tokenizer, prompt)], device=model_device(model))
        model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    return seen["reads"], seen["writes"]


def site_index(tokenizer, prompt: Prompt, subject: str, site: str) -> int:
    """The token an edit keys on in this prompt."""
    if site == "subject":
        return token_index_of(tokenizer, prompt, subject)
    if site == "last":
        return len(encode(tokenizer, prompt)) - 1
    raise ValueError(f"Unknown key site {site!r}. Use one of: {', '.join(KEY_SITES)}.")


def fact_key(model: nn.Module, tokenizer, fact: Fact, layer: int, site: str = "subject", per_prompt: bool = False, thinking: bool = False) -> torch.Tensor:
    """What the MLP reads at the key site across the teaching prompts: their mean, or (input_dim, prompts) with `per_prompt`."""
    keys = []
    for prompt in fact.teaching_prompts(tokenizer, thinking):
        reads, _ = capture_writer(model, tokenizer, prompt, layer)
        keys.append(reads[site_index(tokenizer, prompt, fact.subject, site)])
    stacked = torch.stack(keys)
    return stacked.T if per_prompt else stacked.mean(0)


def subject_key(model: nn.Module, tokenizer, fact: Fact, layer: int) -> torch.Tensor:
    """Mean of what the MLP reads at the subject's last token, across the teaching prompts."""
    return fact_key(model, tokenizer, fact, layer, "subject")


def null_reads(model: nn.Module, tokenizer, fact: Fact, layer: int, site: str, kind: str, name_slots: bool = False) -> torch.Tensor | None:
    """What the MLP reads, at the key site, on prompts the edit must leave alone: (input_dim, q). None for kind "none".

    `name_slots`: also leave alone other subjects at the point where a person's name comes next (read there with site "last").
    """
    if kind not in NULL_KINDS:
        raise ValueError(f"Unknown null_keys {kind!r}. Use one of: {', '.join(NULL_KINDS)}.")
    chosen: list[tuple[Prompt, str]] = []
    if kind in ("same", "both", "all"):
        chosen += [(p, fact.subject) for p in fact.negative_prompts(tokenizer)]
    if kind in ("other", "both", "all"):
        chosen += fact.decoy_teaching_prompts(tokenizer, name_slots=name_slots)
    if kind == "all":
        chosen += fact.decoy_teaching_prompts(tokenizer, REAL_DECOYS, name_slots=name_slots)
    reads = []
    for prompt, subject in chosen:
        seen, _ = capture_writer(model, tokenizer, prompt, layer)
        reads.append(seen[site_index(tokenizer, prompt, subject, site)])
    if kind == "all":
        # Anchors have no subject of their own to key on, so they are read at their final token.
        for prompt in [Prompt(p) for p in ANCHOR_PROMPTS] + [render_chat(tokenizer, q) for q in ANCHOR_QUESTIONS]:
            seen, _ = capture_writer(model, tokenizer, prompt, layer)
            reads.append(seen[-1])
    if not reads:
        return None
    return torch.stack(reads, 1)


@torch.no_grad()
def input_covariance(model: nn.Module, tokenizer, texts: list[str], layers: list[int], batch_size: int = 16, max_length: int = 64) -> dict[int, torch.Tensor]:
    """Second moment of what each chosen layer's MLP reads on ordinary text: {layer: (input_dim, input_dim)}, float64."""
    device = model_device(model)
    totals = {layer: torch.zeros(writer_module(model, layer).weight.shape[1], writer_module(model, layer).weight.shape[1], dtype=torch.float64, device=device) for layer in layers}
    seen: dict[int, torch.Tensor] = {}
    handles = [
        writer_module(model, layer).register_forward_hook(lambda _m, inputs, _o, layer=layer: seen.__setitem__(layer, inputs[0].detach())) for layer in layers
    ]
    count = 0
    try:
        for start in range(0, len(texts), batch_size):
            sequences = [tokenizer(t, truncation=True, max_length=max_length).input_ids for t in texts[start : start + batch_size]]
            ids, mask = batch_ids(tokenizer, sequences, device)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            chosen = mask.bool()
            for layer in layers:
                reads = seen[layer][chosen].double()
                totals[layer] += reads.T @ reads
            count += int(chosen.sum())
    finally:
        for handle in handles:
            handle.remove()
    return {layer: total / count for layer, total in totals.items()}


def metric_from_covariance(covariance: torch.Tensor, ridge: float = 0.02) -> torch.Tensor:
    """Inverse covariance. `ridge` is relative to the mean eigenvalue, so a large ridge approaches plain least change."""
    dim = covariance.shape[0]
    regularised = covariance + ridge * (torch.trace(covariance) / dim) * torch.eye(dim, dtype=covariance.dtype, device=covariance.device)
    return torch.linalg.inv(regularised).float()


def input_metric(model: nn.Module, tokenizer, texts: list[str], layer: int, ridge: float = 0.02, batch_size: int = 16, max_length: int = 64) -> torch.Tensor:
    """Inverse covariance of what the MLP reads on ordinary text."""
    return metric_from_covariance(input_covariance(model, tokenizer, texts, [layer], batch_size, max_length)[layer], ridge)


@dataclass
class LedgerEntry:
    id: str
    fact: Fact
    layer: int
    key: torch.Tensor  # (input_dim,) or (input_dim, keys)
    delta: torch.Tensor  # (output_dim,)
    null_keys: torch.Tensor | None = None  # (input_dim, q): reads this edit must leave alone
    links: list[tuple[torch.Tensor, torch.Tensor]] | None = None  # one (keys, vector) per answer token

    def keys(self) -> torch.Tensor:
        return self.key[:, None] if self.key.dim() == 1 else self.key

    def parts(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """(keys, vector) pairs: one per answer token. An entry without links, from a ledger saved by an older version, is one pair."""
        return list(self.links) if self.links else [(self.keys(), self.delta)]


class FactLedger:
    """Every fact as a (key, delta) pair. Weights are always recompiled from the untouched base.

    So there is no history inside the weights: removing a fact gives exactly the weights of never having added it, and writes do
    not pile up on each other the way repeated single edits do.
    """

    def __init__(self, ridge: float = 0.1):
        self.ridge = ridge
        self.entries: dict[str, LedgerEntry] = {}

    def add(self, entry: LedgerEntry) -> None:
        if entry.id in self.entries:
            raise KeyError(f"The ledger already holds a fact with id {entry.id!r}")
        self.entries[entry.id] = entry

    def remove(self, entry_id: str) -> LedgerEntry:
        return self.entries.pop(entry_id)

    def layers(self) -> list[int]:
        return sorted({e.layer for e in self.entries.values()})

    def compile(self, layer: int, metric: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor] | None:
        chosen = [e for e in self.entries.values() if e.layer == layer]
        if not chosen:
            return None
        parts = [part for e in chosen for part in e.parts()]
        keys = torch.cat([k for k, _ in parts], 1)
        deltas = torch.cat([v[:, None].expand(-1, k.shape[1]) for k, v in parts], 1)
        nulls = [e.null_keys for e in chosen if e.null_keys is not None]
        groups = [i for i, (k, _) in enumerate(parts) for _ in range(k.shape[1])]
        return write_associations(keys, deltas, metric, self.ridge, torch.cat(nulls, 1) if nulls else None, groups)

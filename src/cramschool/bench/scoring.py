"""Scoring one fact on a bench: the test prompts of every group, the checks, and the pass rule (docs/benchmarks.md, "The checks").

Replies are greedy, 16 new tokens, thinking off. All groups are generated in one flat list, 32 at a time, as in the published runs: in
bfloat16 a reply can depend on the batch it is generated in, so keeping the batching keeps the results reproducible.
"""
from __future__ import annotations

import json
import os
import re

import torch

from ..gates import RegressionBaseline, garbled, generate_batch, leaks, mentions
from ..graft.wordings import known_facts
from ..model_access import Prompt, batch_ids, encode, model_device, render_chat

NAME_SLOTS = (" a man named", " a woman named")
MAX_NEW_TOKENS = 16
BATCH = 32
GROUPS = ("recall", "same", "lookalike", "other", "names", "near")
OTHER_GROUPS = ("same", "lookalike", "other", "names")  # the groups in the pass rule; "near" (benches of kinds.py) is reported only
GUESSABLE = ("city", "year", "phrase")  # answer types the unedited model could say by itself (kinds.py)


def leak_args(fact: dict) -> dict:
    """How leaks() reads a fact's answer: its distinctive parts and whether it is guessable, for benches that state an answer type
    (kinds.py); the earlier benches state none and are read as before."""
    if "answer_type" not in fact:
        return {}
    return dict(parts=fact.get("answer_parts"), guessable=fact["answer_type"] in GUESSABLE)
PASS = dict(recall=0.75, same=0.25, lookalike=1 / 12, other=0.25, names=1 / 12, garbled=0.34, garbled_other=0.10, known_drop=0.05, drift=0.05,
            sibling=1 / 12, cross=0.0)


def load_bench(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["facts"]


def read_rows(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def test_prompts(fact: dict, tokenizer, context: str | None = None) -> dict[str, list[Prompt]]:
    """The fixed test prompts of one fact, by group. Never used to build anything.
    `context`: a statement written before every prompt (the in-context yardstick): on the line before a plain prompt, and before the question
    in the same user message for chat."""
    t = fact["test"]
    two_plain, two_chat = t["templates_plain"][:2], t["templates_chat"][:2]
    plain = (lambda s: Prompt(f"{context}\n{s}")) if context else Prompt
    chat = (lambda q: render_chat(tokenizer, f"{context}\n\n{q}")) if context else (lambda q: render_chat(tokenizer, q))

    def swapped(companies):
        return [plain(tp.format(s=c)) for c in companies for tp in two_plain] + [chat(tc.format(s=c)) for c in companies for tc in two_chat]

    return dict(
        recall=[plain(p) for p in t["plain"]] + [chat(q) for q in t["chat"]],
        same=[plain(p) for p in t["same_plain"]] + [chat(q) for q in t["same_chat"]],
        lookalike=swapped(t["lookalikes"]),
        other=swapped(t["others"]),
        names=[plain(two_plain[0].format(s=c) + slot) for c in t["lookalikes"] + t["others"] for slot in NAME_SLOTS] if fact["person"] else [],
        near=[plain(p) for p in t.get("near_plain", [])] + [chat(q) for q in t.get("near_chat", [])],
    )


def case_prompts(fact: dict, tokenizer) -> list[Prompt]:
    """Capitalisation: 2 plain and 2 chat test wordings with the company in lower case, then in upper case."""
    t, out = fact["test"], []
    for n in (fact["subject"].lower(), fact["subject"].upper()):
        out += [Prompt(tp.format(s=n)) for tp in t["templates_plain"][:2]] + [render_chat(tokenizer, tc.format(s=n)) for tc in t["templates_chat"][:2]]
    return out


def generate_groups(model, tokenizer, prompts: dict[str, list[Prompt]]) -> dict[str, list[str]]:
    flat = [p for g in GROUPS for p in prompts[g]]
    texts = generate_batch(model, tokenizer, flat, MAX_NEW_TOKENS, batch_size=BATCH)
    out, i = {}, 0
    for g in GROUPS:
        out[g] = texts[i:i + len(prompts[g])]
        i += len(prompts[g])
    return out


def answered(text: str, answers: list[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", text, re.IGNORECASE) for a in answers)


class KnownCheck:
    """133 real-world questions the unedited model answers. Damage is the number of them lost, counted item by item."""

    def __init__(self, model, tokenizer, items: list[tuple[str, list[str]]] | None = None):
        self.items = items if items is not None else known_facts()
        self.model, self.tokenizer = model, tokenizer
        self.before = self.correct()

    def correct(self, context: str | None = None) -> list[bool]:
        prompts = [Prompt(f"{context}\n{p}" if context else p) for p, _ in self.items]
        texts = generate_batch(self.model, self.tokenizer, prompts, max_new_tokens=12)
        return [answered(t, a) for t, (_, a) in zip(texts, self.items)]

    def measure(self, context: str | None = None) -> dict:
        now = self.correct(context)
        return dict(known=sum(now) / len(now), known_lost=sum(b and not n for b, n in zip(self.before, now)), known_n=len(now))


FIRM_MARGIN, FIRM_SHIFT = 1.0, 0.10


class FirmCheck:
    """Known facts judged by how firmly the unedited model knows them (lab notebook, "The damage bar, re-founded on firmly known facts").

    A fact's margin: at the prompt's last token, the logit of an accepted answer's first token (spelled after a space; the best over the
    accepted answers) minus the largest logit of any token that starts no accepted answer. A fact is firmly known when the unedited model
    answers it in the generation check with a margin of at least 1.0, and knife-edge when it answers with less: knife-edge answers flip on
    one rounding step, with or without an edit. An edited model passes when it loses no firm fact (each still answered, margin above 0) and
    the firm facts' mean margin change is within ±0.10.

    Build it on the unedited model. measure() scores the model as it is now, at the same batching."""

    def __init__(self, model, tokenizer, items: list[tuple[str, list[str]]], *, max_new_tokens: int = 8, batch: int = 64):
        self.model, self.tokenizer, self.items = model, tokenizer, items
        self.max_new_tokens, self.batch = max_new_tokens, batch
        self._ids = [encode(tokenizer, Prompt(p)) for p, _ in items]
        self._firsts = [sorted({tokenizer(" " + a, add_special_tokens=False).input_ids[0] for a in answers}) for _, answers in items]
        self.before = self.correct()
        self.margin_before = self.margins()
        self.firm = [b and m >= FIRM_MARGIN for b, m in zip(self.before, self.margin_before)]

    def correct(self, batch: int | None = None, reverse: bool = False) -> list[bool]:
        """Which facts the model answers now, by greedy generation. batch, reverse: another batching, for the noise floor."""
        order = list(range(len(self.items)))[::-1] if reverse else list(range(len(self.items)))
        texts = generate_batch(self.model, self.tokenizer, [Prompt(self.items[i][0]) for i in order], self.max_new_tokens, batch_size=batch or self.batch)
        out = [False] * len(self.items)
        for i, t in zip(order, texts):
            out[i] = answered(t, self.items[i][1])
        return out

    @torch.no_grad()
    def margins(self) -> list[float]:
        device, out = model_device(self.model), []
        for start in range(0, len(self._ids), self.batch):
            chunk = self._ids[start:start + self.batch]
            ids, mask = batch_ids(self.tokenizer, chunk, device)  # right padding: a prompt's last token never sees the padding
            logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            for row, seq in enumerate(chunk):
                lg = logits[row, len(seq) - 1].float().clone()
                firsts = self._firsts[start + row]
                own = float(lg[firsts].max())
                lg[firsts] = -float("inf")
                out.append(own - float(lg.max()))
            del logits
        return out

    def measure(self) -> dict:
        now, m = self.correct(), self.margins()
        firm = [i for i, f in enumerate(self.firm) if f]
        knife = [i for i, (b, f) in enumerate(zip(self.before, self.firm)) if b and not f]
        unanswered = [i for i in firm if not now[i]]
        overtaken = [i for i in firm if m[i] <= 0]
        lost = sorted(set(unanswered) | set(overtaken))
        shift = sum(m[i] - self.margin_before[i] for i in firm) / len(firm) if firm else 0.0
        return dict(firm=len(firm), firm_lost=len(lost), firm_unanswered=len(unanswered), firm_overtaken=len(overtaken),
                    firm_lost_items=[self.items[i][0] for i in lost], firm_shift=shift, firm_min=min((m[i] for i in firm), default=None),
                    knife=len(knife), knife_lost=sum(not now[i] for i in knife), gained=sum(n and not b for n, b in zip(now, self.before)),
                    answered_before=sum(self.before), answered=sum(now), passed=bool(not lost and abs(shift) <= FIRM_SHIFT))


def make_baseline(model, tokenizer) -> RegressionBaseline:
    """Drift against the unedited model on ordinary text, and the known-facts check. Build it before attaching anything."""
    baseline = RegressionBaseline(model, tokenizer)
    baseline.known_check = KnownCheck(model, tokenizer)
    baseline.known_accuracy = sum(baseline.known_check.before) / len(baseline.known_check.before)
    return baseline


def score_fact(model, tokenizer, fact: dict, baseline: RegressionBaseline, before: dict | None, context: str | None = None) -> dict:
    """Answers and measures for one fact on the model as it is now. `before`: the unedited answers by group, for the leak checks."""
    prompts = test_prompts(fact, tokenizer, context)
    texts = generate_groups(model, tokenizer, prompts)
    target = fact["target"]
    m = dict(recall=sum(mentions(t, target) for t in texts["recall"]) / len(texts["recall"]),
             garbled=sum(garbled(t) for t in texts["recall"]) / len(texts["recall"]))
    how = leak_args(fact)
    for g in OTHER_GROUPS + (("near",) if prompts["near"] else ()):
        if not prompts[g]:
            m[g] = None
            continue
        earlier = before[g] if before is not None and g in before else [None] * len(prompts[g])
        m[g] = sum(leaks(t, target, p.text, b, **how) for t, p, b in zip(texts[g], prompts[g], earlier)) / len(prompts[g])
    pairs = [(t, (before[g][i] if before is not None else None)) for g in OTHER_GROUPS for i, t in enumerate(texts[g])]
    m["garbled_other"] = sum(garbled(t) and not (b is not None and garbled(b)) for t, b in pairs) / len(pairs)
    if hasattr(baseline, "known_check"):
        m.update(baseline.known_check.measure(context))
    else:
        m["known"] = baseline.known_score()
    m["drift"] = baseline.drift()[0]
    return dict(measures=m, answers=texts)


def score_table_fact(model, tokenizer, fact: dict, baseline: RegressionBaseline, before: dict, loaded: list[dict]) -> tuple[dict, list[str], dict, list]:
    """score_fact plus the checks for a table of many facts: capitalisation recall, the sibling check and the other-facts check."""
    scored = score_fact(model, tokenizer, fact, baseline, before=before)
    m, answers = scored["measures"], scored["answers"]
    cases = generate_batch(model, tokenizer, case_prompts(fact, tokenizer), MAX_NEW_TOKENS, batch_size=BATCH)
    m["case"] = sum(mentions(t, fact["target"]) for t in cases) / len(cases)
    sib = next((g for g in loaded if g["subject"] == fact["subject"] and g["target"] != fact["target"]), None) if fact.get("sibling") else None
    prompts = test_prompts(fact, tokenizer)
    if sib is not None:
        how = leak_args(sib)
        m["sibling"] = sum(leaks(t, sib["target"], p.text, b, **how) for t, p, b in zip(answers["recall"], prompts["recall"], before["recall"])) / len(answers["recall"])
    same_rel = fact["test"].get("same_plain_rel", []) + fact["test"].get("same_chat_rel", [])
    hits, total, near_hits = 0, 0, 0
    where = []
    for grp in GROUPS:
        for i, (t, p) in enumerate(zip(answers[grp], prompts[grp])):
            total += grp != "near"
            b = before[grp][i] if before is not None and grp in before else None
            bad = []
            for g in loaded:
                if g["target"] == fact["target"] or (sib is not None and g is sib and grp == "recall"):
                    continue
                if sib is not None and g is sib and grp == "same" and i < len(same_rel) and same_rel[i] == g["relation"]:
                    continue
                if answered(t, [g["target"]]) and not answered(p.text, [g["target"]]) and not (b is not None and answered(b, [g["target"]])):
                    bad.append(g["target"])
            if bad:
                if grp == "near":  # near questions are reported, not judged: another fact's answer there is counted apart
                    near_hits += 1
                else:
                    hits += 1
                where.append(dict(group=grp, index=i, targets=bad))
    m["cross"] = hits / total
    if prompts["near"]:
        m["near_cross"] = near_hits / len(prompts["near"])
    failed = failures(m, baseline.known_accuracy)
    return m, failed, dict(answers, case=cases), where


def failures(measures: dict, known_before: float) -> list[str]:
    """The parts of the pass rule that fail. Empty means the fact passes."""
    out = []
    if measures["recall"] < PASS["recall"]:
        out.append(f"recall {measures['recall']:.2f}")
    for g in OTHER_GROUPS:
        if measures.get(g) is not None and measures[g] > PASS[g] + 1e-9:
            out.append(f"{g} {measures[g]:.2f}")
    if measures["garbled"] > PASS["garbled"]:
        out.append(f"garbled {measures['garbled']:.2f}")
    if measures.get("garbled_other") is not None and measures["garbled_other"] > PASS["garbled_other"] + 1e-9:
        out.append(f"garbled_other {measures['garbled_other']:.2f}")
    if "known_lost" in measures:
        if measures["known_lost"] > PASS["known_drop"] * measures["known_n"] + 1e-9:
            out.append(f"known lost {measures['known_lost']}/{measures['known_n']}")
    elif known_before - measures["known"] > PASS["known_drop"] + 1e-9:
        out.append(f"known {known_before:.2f}->{measures['known']:.2f}")
    if measures["drift"] > PASS["drift"]:
        out.append(f"drift {measures['drift']:.3f}")
    if measures.get("sibling") is not None and measures["sibling"] > PASS["sibling"] + 1e-9:
        out.append(f"sibling {measures['sibling']:.2f}")
    if measures.get("cross") is not None and measures["cross"] > PASS["cross"]:
        out.append(f"cross {measures['cross']:.2f}")
    return out

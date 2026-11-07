"""Shared scoring for the fact bench (plan.md): the same prompts, batching and checks for every arm.

Environment:
  BENCH_NAME   which bench to load: "bench40" (default) or another <name>.json here; results go to results/bench (bench40) or results/<name>
  BENCH_TEACH  path of a teaching-wordings file (teach_wordings.json): its templates, filled with the fact's subject, are added to the
               training wordings, and arm names get the suffix "_teach". Templates equal to any of the bench's own wordings after
               normalizing are dropped first (plan.md).
"""
import json
import os
import re

from weightspace.gates import KNOWN_FACTS, ORDINARY_TEXT, RegressionBaseline, garbled, generate_batch, leaks, mentions  # noqa: F401
from weightspace.model_access import Prompt, render_chat

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = os.environ.get("BENCH_NAME", "bench40")
BENCH = os.path.join(HERE, f"{NAME}.json")
RESULTS = os.path.join(HERE, "..", "results", "bench" if NAME == "bench40" else NAME)
KNOWN_FILE = os.path.join(HERE, "known_facts.json")
TEACH = os.environ.get("BENCH_TEACH") or None
ARM_SUFFIX = "_teach" if TEACH else ""
NAME_SLOTS = (" a man named", " a woman named")
MAX_NEW_TOKENS = 16
BATCH = 32

GROUPS = ("recall", "same", "lookalike", "other", "names")
OTHER_GROUPS = ("same", "lookalike", "other", "names")
PASS = dict(recall=0.75, same=0.25, lookalike=1 / 12, other=0.25, names=1 / 12, garbled=0.34, garbled_other=0.10, known_drop=0.05, drift=0.05)


def load_bench() -> list[dict]:
    with open(BENCH, encoding="utf-8") as fh:
        return json.load(fh)["facts"]


def normalized(template: str) -> str:
    """The duplicate rule of plan.md: lower case, the subject slot as one marker, punctuation and repeated spaces removed."""
    return " ".join(re.sub(r"[^\w\s]", " ", template.lower().replace("{s}", " subjectslot ")).split())


_teach_cache = {}


def teaching_templates(relation: str) -> tuple[list[str], list[str]]:
    """(plain, chat) editor-drafted templates for one relation, minus any equal to one of the bench's own wordings after normalizing."""
    if not TEACH:
        return [], []
    if "all" not in _teach_cache:
        with open(TEACH, encoding="utf-8") as fh:
            drafted = json.load(fh)["wordings"]
        pool = set()
        for f in load_bench():
            for side in ("train", "test"):
                pool |= {normalized(t) for t in f[side]["templates_plain"] + f[side]["templates_chat"]}
        _teach_cache["all"] = {rel: tuple([t for t in d[form] if normalized(t) not in pool] for form in ("plain", "chat")) for rel, d in drafted.items()}
        _teach_cache["dropped"] = {rel: [t for form in ("plain", "chat") for t in d[form] if normalized(t) in pool] for rel, d in drafted.items()}
    return _teach_cache["all"][relation]


def test_prompts(fact: dict, tokenizer) -> dict[str, list[Prompt]]:
    """The fixed test prompts of one fact, by group. Never used to build an edit."""
    t = fact["test"]
    two_plain, two_chat = t["templates_plain"][:2], t["templates_chat"][:2]

    def swapped(companies):
        return [Prompt(tp.format(s=c)) for c in companies for tp in two_plain] + [render_chat(tokenizer, tc.format(s=c)) for c in companies for tc in two_chat]

    out = dict(
        recall=[Prompt(p) for p in t["plain"]] + [render_chat(tokenizer, q) for q in t["chat"]],
        same=[Prompt(p) for p in t["same_plain"]] + [render_chat(tokenizer, q) for q in t["same_chat"]],
        lookalike=swapped(t["lookalikes"]),
        other=swapped(t["others"]),
        names=[Prompt(two_plain[0].format(s=c) + slot) for c in t["lookalikes"] + t["others"] for slot in NAME_SLOTS] if fact["person"] else [],
    )
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
    """known_facts.json: real-world facts the unedited model answers. Damage is the share of them lost, counted item by item."""

    def __init__(self, model, tokenizer):
        with open(KNOWN_FILE, encoding="utf-8") as fh:
            self.items = [(d["prompt"], d["answers"]) for d in json.load(fh)["facts"]]
        self.model, self.tokenizer = model, tokenizer
        self.before = self.correct()

    def correct(self) -> list[bool]:
        texts = generate_batch(self.model, self.tokenizer, [Prompt(p) for p, _ in self.items], max_new_tokens=12)
        return [answered(t, a) for t, (_, a) in zip(texts, self.items)]

    def measure(self) -> dict:
        now = self.correct()
        return dict(known=sum(now) / len(now), known_lost=sum(b and not n for b, n in zip(self.before, now)), known_n=len(now))


def make_baseline(model, tokenizer) -> RegressionBaseline:
    """Drift against the unedited model on the package's ordinary texts, and the bench's known-facts check."""
    baseline = RegressionBaseline(model, tokenizer)
    baseline.known_check = KnownCheck(model, tokenizer)
    baseline.known_accuracy = sum(baseline.known_check.before) / len(baseline.known_check.before)
    return baseline


def score_fact(model, tokenizer, fact: dict, baseline: RegressionBaseline, before: dict | None) -> dict:
    """Answers and measures for one fact on the model as it is now. `before`: the unedited answers by group, for the leak checks."""
    prompts = test_prompts(fact, tokenizer)
    texts = generate_groups(model, tokenizer, prompts)
    target = fact["target"]
    m = dict(recall=sum(mentions(t, target) for t in texts["recall"]) / len(texts["recall"]),
             garbled=sum(garbled(t) for t in texts["recall"]) / len(texts["recall"]))
    for g in OTHER_GROUPS:
        if not prompts[g]:
            m[g] = None
            continue
        earlier = before[g] if before is not None else [None] * len(prompts[g])
        m[g] = sum(leaks(t, target, p.text, b) for t, p, b in zip(texts[g], prompts[g], earlier)) / len(prompts[g])
    # Garbled answers outside the recall group, counted only where the unedited answer to the same prompt was not garbled.
    pairs = [(t, (before[g][i] if before is not None else None)) for g in OTHER_GROUPS for i, t in enumerate(texts[g])]
    m["garbled_other"] = sum(garbled(t) and not (b is not None and garbled(b)) for t, b in pairs) / len(pairs)
    if hasattr(baseline, "known_check"):
        m.update(baseline.known_check.measure())
    else:
        m["known"] = baseline.known_score()
    m["drift"] = baseline.drift()[0]
    return dict(measures=m, answers=texts)


def failures(measures: dict, known_before: float) -> list[str]:
    """The parts of the pass rule that fail. Empty means the fact passes."""
    out = []
    if measures["recall"] < PASS["recall"]:
        out.append(f"recall {measures['recall']:.2f}")
    for g in OTHER_GROUPS:
        if measures[g] is not None and measures[g] > PASS[g] + 1e-9:
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
    return out


def read_rows(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(line) for line in fh if line.strip())}

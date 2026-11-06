"""Shared scoring for the fact bench (plan.md): the same prompts, batching and checks for every arm."""
import json
import os

from weightspace.gates import KNOWN_FACTS, ORDINARY_TEXT, RegressionBaseline, garbled, generate_batch, leaks, mentions  # noqa: F401
from weightspace.model_access import Prompt, render_chat

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench40.json")
RESULTS = os.path.join(HERE, "..", "results", "bench")
NAME_SLOTS = (" a man named", " a woman named")
MAX_NEW_TOKENS = 16
BATCH = 32

GROUPS = ("recall", "same", "lookalike", "other", "names")
PASS = dict(recall=0.75, same=0.25, lookalike=1 / 12, other=0.25, names=1 / 12, garbled=0.34, known_drop=0.05, drift=0.05)


def load_bench() -> list[dict]:
    with open(BENCH, encoding="utf-8") as fh:
        return json.load(fh)["facts"]


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


def score_fact(model, tokenizer, fact: dict, baseline: RegressionBaseline, before: dict | None) -> dict:
    """Answers and measures for one fact on the model as it is now. `before`: the unedited answers by group, for the leak checks."""
    prompts = test_prompts(fact, tokenizer)
    texts = generate_groups(model, tokenizer, prompts)
    target = fact["target"]
    m = dict(recall=sum(mentions(t, target) for t in texts["recall"]) / len(texts["recall"]),
             garbled=sum(garbled(t) for t in texts["recall"]) / len(texts["recall"]))
    for g in ("same", "lookalike", "other", "names"):
        if not prompts[g]:
            m[g] = None
            continue
        earlier = before[g] if before is not None else [None] * len(prompts[g])
        m[g] = sum(leaks(t, target, p.text, b) for t, p, b in zip(texts[g], prompts[g], earlier)) / len(prompts[g])
    m["known"] = baseline.known_score()
    m["drift"] = baseline.drift()[0]
    return dict(measures=m, answers=texts)


def failures(measures: dict, known_before: float) -> list[str]:
    """The parts of the pass rule that fail. Empty means the fact passes."""
    out = []
    if measures["recall"] < PASS["recall"]:
        out.append(f"recall {measures['recall']:.2f}")
    for g in ("same", "lookalike", "other", "names"):
        if measures[g] is not None and measures[g] > PASS[g] + 1e-9:
            out.append(f"{g} {measures[g]:.2f}")
    if measures["garbled"] > PASS["garbled"]:
        out.append(f"garbled {measures['garbled']:.2f}")
    if known_before - measures["known"] > PASS["known_drop"] + 1e-9:
        out.append(f"known {known_before:.2f}->{measures['known']:.2f}")
    if measures["drift"] > PASS["drift"]:
        out.append(f"drift {measures['drift']:.3f}")
    return out


def read_rows(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(line) for line in fh if line.strip())}

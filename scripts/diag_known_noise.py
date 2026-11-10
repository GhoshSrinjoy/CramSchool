"""How many known facts flip from numerical noise alone? (Lab notebook, "Result of step 1c".) The unedited model's greedy answers to the
133 known facts and to step 1c's 400 CounterFact facts, generated with different batch sizes and orders. In bfloat16 a reply can depend
on the batch it is generated in (padding and the other prompts change the arithmetic), so a fact the model barely gets right can flip
with no weight changed. Each setting is compared with the checks' own batching, the reference the slot tests compare against.
Forward passes only. GPU, about 3 minutes.

Usage: python scripts/diag_known_noise.py <model folder> <step 1c json> <out.json>
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.scoring import answered
from cramschool.gates import generate_batch
from cramschool.graft.wordings import known_facts
from cramschool.model_access import Prompt

SETTINGS = [(1, False), (8, False), (16, False), (32, False), (64, False), (16, True), (64, True)]  # (batch size, reversed order)


def log(s):
    print(s, flush=True)


def correct(model, tok, items, tokens, batch, reverse):
    order = list(range(len(items)))[::-1] if reverse else list(range(len(items)))
    texts = generate_batch(model, tok, [Prompt(items[i][0]) for i in order], max_new_tokens=tokens, batch_size=batch)
    out = [None] * len(items)
    for i, t in zip(order, texts):
        out[i] = answered(t, items[i][1])
    return out


def main():
    model_dir, step1c, out_path = sys.argv[1:4]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    with open(step1c, encoding="utf-8") as fh:
        cf = [(r["prompt"], r["answers"]) for r in json.load(fh)["cf_known"]]
    sets = dict(known=(known_facts(), 12, 16), counterfact=(cf, 8, 64))  # items, new tokens, the check's own batch size
    t0, rows = time.time(), []
    for name, (items, tokens, own) in sets.items():
        ref = correct(model, tok, items, tokens, own, False)
        flipped_any = set()
        for batch, rev in SETTINGS:
            now = correct(model, tok, items, tokens, batch, rev)
            lost = [i for i, (a, b) in enumerate(zip(ref, now)) if a and not b]
            gained = [i for i, (a, b) in enumerate(zip(ref, now)) if b and not a]
            flipped_any |= set(lost)
            rows.append(dict(set=name, batch=batch, reversed=rev, answered=sum(now), lost=len(lost), gained=len(gained),
                             lost_prompts=[items[i][0] for i in lost]))
            log(f"{name:<11} batch {batch:>2}{' reversed' if rev else '':<9}: answered {sum(now)} of {len(items)} (reference {sum(ref)}), "
                f"lost {len(lost)}, gained {len(gained)} | {time.time() - t0:.0f}s")
        log(f"{name}: facts lost under at least one batching, no weight changed: {len(flipped_any)} of {sum(ref)}")
        rows.append(dict(set=name, summary=True, reference_answered=sum(ref), lost_under_some_batching=len(flipped_any)))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(settings=SETTINGS, rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()

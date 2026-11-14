"""Does GRAFT work at another layer than 20? (Lab notebook, step 10.) For each layer, a bench's 40 facts are added through the API into one
table at that layer (the fast build), and each fact is scored on its recall prompts and on the other questions about its company. A screen,
not the bench's full check: no look-alike, name-slot, known-facts or drift checks, which don't depend on the layer's choice in the same way
and are run on the layer that is chosen. Forward passes only. GPU, about 8 minutes per layer.

Usage: python scripts/layer_screen.py <model folder> <bench json> <base rows jsonl> <out json> <tables folder> <layer> [<layer> ...]

Each layer's table is saved in the tables folder as layer<N>.graft, for the runs that follow the screen.
"""
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft, GraftOptions
from cramschool.bench.scoring import BATCH, MAX_NEW_TOKENS, PASS, load_bench, read_rows, test_prompts
from cramschool.gates import generate_batch, leaks, mentions

MODEL, BENCH, BASE, OUT, TABLES = sys.argv[1:6]
LAYERS = [int(x) for x in sys.argv[6:]]
os.makedirs(TABLES, exist_ok=True)
facts, base = load_bench(BENCH), read_rows(BASE)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
report = {}
for layer in LAYERS:
    t0 = time.time()
    graft = Graft(model, tok, layer=layer, options=GraftOptions(fast=True))
    for f in facts:
        graft.add(f["subject"], f["relation"], f["target"])
    built = time.time() - t0
    graft.save(os.path.join(TABLES, f"layer{layer}.graft"))
    stored = {f["subject"]: {g["relation"]: g["target"] for g in facts if g["subject"] == f["subject"]} for f in facts}
    rows = []
    with graft.attached():
        for f in facts:
            prompts = test_prompts(f, tok)
            texts = generate_batch(model, tok, prompts["recall"] + prompts["same"], MAX_NEW_TOKENS, batch_size=BATCH)
            n = len(prompts["recall"])
            recall = sum(mentions(t, f["target"]) for t in texts[:n]) / n
            same = sum(leaks(t, f["target"], p.text, b) for t, p, b in zip(texts[n:], prompts["same"], base[f["id"]]["answers"]["same"])) / len(prompts["same"])
            rows.append(dict(id=f["id"], relation=f["relation"], recall=recall, same=same, ok=recall >= PASS["recall"] and same <= PASS["same"] + 1e-9))
        events = dict(graft._layer.events)
    graft.detach()
    by_rel = {r: sum(x["ok"] for x in rows if x["relation"] == r) for r in ("owner", "ceo", "founder", "hq")}
    report[layer] = dict(seconds_to_build=round(built), ok=sum(x["ok"] for x in rows), recall_median=statistics.median(x["recall"] for x in rows),
                         recall_mean=statistics.mean(x["recall"] for x in rows), same_median=statistics.median(x["same"] for x in rows),
                         same_over=sum(x["same"] > PASS["same"] + 1e-9 for x in rows), by_relation=by_rel, events=events, rows=rows)
    print(f"layer {layer}: recall and same-company leak both within the bench's limits on {report[layer]['ok']} of {len(rows)} facts; recall median "
          f"{report[layer]['recall_median']:.3f} (mean {report[layer]['recall_mean']:.3f}); same-company leak over the limit on "
          f"{report[layer]['same_over']}; by relation {by_rel}; built in {built:.0f}s", flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    del graft
    torch.cuda.empty_cache()

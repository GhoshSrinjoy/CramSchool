"""Which CounterFact facts does the unedited model know? GPU, about 10 minutes for all 21,919 rewrite prompts.

Usage: python scripts/counterfact_screen.py <model folder> <counterfact.json> [out.jsonl]
"""
import collections
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.counterfact import load_counterfact, screen


def main():
    model_dir, data = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join("runs", "counterfact", "screen.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    records = load_counterfact(data)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    rows = screen(model, tok, records, progress=lambda s: print(s, flush=True))
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        by[r["relation"]][0] += 1
        by[r["relation"]][1] += r["knows"]
        by[r["relation"]][2] += r["says_new"]
    print("relation  records  knows-old  says-new  example", flush=True)
    examples = {}
    for r in rows:
        if r["knows"]:
            examples.setdefault(r["relation"], f"{r['prompt']} -> {r['true']}")
    for rel, (n, k, s) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"{rel:<8} {n:>7} {k:>10} {s:>9}  {examples.get(rel, '')}", flush=True)
    print(f"screen done: the model gives the old answer on {sum(r['knows'] for r in rows)} of {len(rows)} rewrite prompts", flush=True)


if __name__ == "__main__":
    main()

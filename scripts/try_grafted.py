"""Ask a grafted model and its base the same questions with plain transformers: no cramschool import, no hooks. The edits are in the
weights or nowhere.

Usage: python scripts/try_grafted.py <base folder> <grafted folder> <facts.json> [how many edits, default 12]
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def answers(folder, prompts):
    tok = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.bfloat16, device_map="cuda").eval()
    out = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(model.device)
            gen = model.generate(ids, max_new_tokens=10, do_sample=False, temperature=None, top_p=None, top_k=None)
            out.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0])
    del model
    torch.cuda.empty_cache()
    return out


def main():
    base, grafted, facts_path = sys.argv[1:4]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    with open(facts_path, encoding="utf-8") as fh:
        facts = json.load(fh)[:k]
    rows = []
    for f in facts:
        c = f["cf"]
        rows += [(f["subject"], c["old"], c["new"], "rewrite", c["rewrite"]), (f["subject"], c["old"], c["new"], "reworded", c["paraphrases"][0]),
                 (f["subject"], c["old"], c["new"], "neighbour", c["neighbours"][0])]
    prompts = [r[4] for r in rows]
    a0, a1 = answers(base, prompts), answers(grafted, prompts)
    for (subject, old, new, kind, p), x, y in zip(rows, a0, a1):
        print(f"[{kind}] {p!r}\n    base:    {x!r}\n    grafted: {y!r}    (old {old}, new {new})")


if __name__ == "__main__":
    main()

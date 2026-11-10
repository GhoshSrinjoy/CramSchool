"""Are the known facts a grafted model loses fragile ones? For each known-fact prompt of the 133, the step 1c set and the fresh set: the
unedited model's margin at the prompt's last token between the answer's first token (spelled after a space) and the best other token, and
the same in a grafted model. Lost items (from a check's summary.json) are placed among their set's margins.

Usage: python scripts/diag_known_margins.py <base model folder> <grafted folder> <step 1c json> <fresh known json> <check summary.json> <out json>
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.graft.wordings import known_facts


@torch.no_grad()
def margins(model, tok, items):
    out = []
    for p, answers in items:
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        logits = model(ids).logits[0, -1].float()
        first = tok(" " + answers[0], add_special_tokens=False).input_ids[0]
        other = logits.clone()
        other[first] = -float("inf")
        out.append(dict(prompt=p, answer=answers[0], margin=float(logits[first] - other.max()), top=tok.decode([int(logits.argmax())])))
    return out


def main():
    base_dir, graft_dir, step1c, fresh_path, check_path, out_path = sys.argv[1:7]
    tok = AutoTokenizer.from_pretrained(base_dir)
    with open(step1c, encoding="utf-8") as fh:
        old = [(r["prompt"], r["answers"]) for r in json.load(fh)["cf_known"]]
    with open(fresh_path, encoding="utf-8") as fh:
        fresh = [(r["prompt"], r["answers"]) for r in json.load(fh)["cf_known"]]
    with open(check_path, encoding="utf-8") as fh:
        lost = json.load(fh)["lost"]["with_switch"]
    sets = dict(known=known_facts(), cf=old, fresh=fresh)
    res = {}
    for name, path in (("base", base_dir), ("grafted", graft_dir)):
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        res[name] = {k: margins(model, tok, items) for k, items in sets.items()}
        del model
        torch.cuda.empty_cache()
    report = {}
    for k in sets:
        rows = [dict(b, grafted_margin=g["margin"], grafted_top=g["top"]) for b, g in zip(res["base"][k], res["grafted"][k])]
        ms = sorted(r["margin"] for r in rows)
        for r in rows:
            r["rank"] = sum(m < r["margin"] for m in ms) / len(ms)  # the share of the set with a smaller unedited margin
        lost_rows = [r for r in rows if r["prompt"] in set(lost[k])]
        report[k] = dict(n=len(rows), median_margin=ms[len(ms) // 2], lost=lost_rows, rows=rows)
        print(f"{k}: {len(rows)} prompts, unedited margin median {ms[len(ms) // 2]:.2f}, lowest decile below {ms[len(ms) // 10]:.2f}")
        for r in lost_rows:
            print(f"  lost {r['prompt']!r} ({r['answer']}): unedited margin {r['margin']:.2f} (rank {r['rank']:.0%}, top {r['top']!r}); "
                  f"grafted {r['grafted_margin']:.2f} (top {r['grafted_top']!r})")
        cap = [r for r in rows if r["prompt"].lower().startswith("the capital of")]
        if cap:
            cm = sorted(r["margin"] for r in cap)
            print(f"  capital questions: {len(cap)}, unedited margin median {cm[len(cm) // 2]:.2f}; the rest's median "
                  f"{sorted(r['margin'] for r in rows if r not in cap)[(len(rows) - len(cap)) // 2]:.2f}")
        shift = sorted(r["grafted_margin"] - r["margin"] for r in rows)
        print(f"  margin change in the grafted model: median {shift[len(shift) // 2]:+.3f}, 5th percentile {shift[len(shift) // 20]:+.3f}")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

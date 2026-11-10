"""Which known facts a grafted model loses, and why: each of the 133 and the screened CounterFact set, answered by the base model and by the
grafted model without and with its switch (same batching, so no batching noise), with the replies, the edited subject a prompt names if any,
and the subject the switch had on at the prompt's end.

Usage: python scripts/diag_known_flips.py <base model folder> <grafted folder> <step 1c json> <out json>
"""
import importlib.util
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.scoring import answered
from cramschool.graft.wordings import known_facts
from cramschool.gates import generate_batch
from cramschool.model_access import Prompt


def log(s):
    print(s, flush=True)


def replies(model, tok, items, new, batch):
    return generate_batch(model, tok, [Prompt(p) for p, _ in items], max_new_tokens=new, batch_size=batch)


def main():
    base_dir, graft_dir, step1c, out_path = sys.argv[1:5]
    tok = AutoTokenizer.from_pretrained(base_dir)
    with open(step1c, encoding="utf-8") as fh:
        cf = [(r["prompt"], r["answers"]) for r in json.load(fh)["cf_known"]]
    with open(os.path.join(graft_dir, "grafted.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    subjects = sorted({n["subject"] for n in spec["neurons"]})
    sets = dict(known133=(known_facts(), 12, 16), cf391=(cf, 8, 64))  # the scorers' own settings: KnownCheck 12 tokens x 16, ScreenCheck 8 x 64
    out = {}
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    for name, (items, new, batch) in sets.items():
        out[name] = dict(items=[dict(prompt=p, answers=a, base=r) for (p, a), r in zip(items, replies(base, tok, items, new, batch))])
    del base
    torch.cuda.empty_cache()
    log("base replies done")
    grafted = AutoModelForCausalLM.from_pretrained(graft_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    for name, (items, new, batch) in sets.items():
        for row, r in zip(out[name]["items"], replies(grafted, tok, items, new, batch)):
            row["off"] = r
    log("grafted replies without the switch done")
    spec_mod = importlib.util.spec_from_file_location("switch_in_folder", os.path.join(graft_dir, "grafted_switch.py"))
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    switch = mod.attach(grafted, tok, graft_dir)
    for name, (items, new, batch) in sets.items():
        for row, r in zip(out[name]["items"], replies(grafted, tok, items, new, batch)):
            row["on"] = r
    # the subject the switch has on at each prompt's end, read by its own name index, one prompt at a time
    index = switch.index
    for name in sets:
        for row in out[name]["items"]:
            ids = tok(row["prompt"], add_special_tokens=False).input_ids
            index.reset(1)
            cur = None
            for t in ids:
                cur = index.step(0, t)
            row["switch_on_for"] = index.names[cur] if cur is not None else None
            low = " ".join(row["prompt"].lower().split())
            row["names_edited_subject"] = [s for s in subjects if " ".join(s.lower().split()) in low]
    switch.detach()
    for name in sets:
        rows = out[name]["items"]
        for r in rows:
            r["ok_base"], r["ok_off"], r["ok_on"] = (answered(r[k], r["answers"]) for k in ("base", "off", "on"))
        lost_off = [r for r in rows if r["ok_base"] and not r["ok_off"]]
        lost_on = [r for r in rows if r["ok_base"] and not r["ok_on"]]
        out[name].update(lost_off=len(lost_off), lost_on=len(lost_on), base_correct=sum(r["ok_base"] for r in rows))
        log(f"{name}: base answers {out[name]['base_correct']} of {len(rows)}; lost without the switch {len(lost_off)}, with it {len(lost_on)}")
        for r in lost_on + [r for r in lost_off if r not in lost_on]:
            tag = "both" if (r in lost_on and r in lost_off) else ("switch only" if r in lost_on else "off only")
            log(f"  [{tag}] {r['prompt']!r} -> base {r['base'][:40]!r} | off {r['off'][:40]!r} | on {r['on'][:40]!r} | answers {r['answers']} "
                f"| switch on for {r['switch_on_for']!r}, names edited {r['names_edited_subject']}")
        named = [r for r in rows if r["names_edited_subject"]]
        log(f"  prompts naming an edited subject: {len(named)}; of those, the switch on at the end for {sum(r['switch_on_for'] is not None for r in named)}")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

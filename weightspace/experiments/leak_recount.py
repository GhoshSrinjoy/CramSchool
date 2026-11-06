"""Recount the kept fact's leaks with the fixed check, which also counts a part of the name, such as "Julian", that the unedited answer lacked.

1. The 42-prompt check made after the session, from its saved answers.
2. The session's own `test` measurement, re-run on the real model as the session ran it: the loop fix already in, then the fact at strength 1.25.

Usage: python -I leak_recount.py <model dir> <patch path, without extension> <fact args json> <fresh check json> <out.json>
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightspace import Fact, GateConfig, WeightPatch, monitor
from weightspace.gates import RegressionBaseline, leaks, mentions, share_mentioning

model_dir, patch_path, fact_args, fresh_path, out_path = sys.argv[1:6]
sys.stdout.reconfigure(errors="replace")
out = {}

with open(fresh_path, encoding="utf-8") as fh:
    fresh = json.load(fh)
print("1. The 42-prompt check, recounted from its saved answers")
for group in ("same subject", "other company"):
    rows = [r for r in fresh["rows"] if r["group"] == group]
    old = sum(mentions(r["patched"], fresh["target"]) for r in rows)
    new = sum(leaks(r["patched"], fresh["target"], r["ask"], r["base"]) for r in rows)
    out[f"fresh {group}"] = dict(n=len(rows), full_name_only=old, fixed_check=new)
    print(f"   {group:<14} leaks: full name only {old}/{len(rows)}, fixed check {new}/{len(rows)}")

with open(fact_args, encoding="utf-8") as fh:
    spec = json.load(fh)["fact"]
fact = Fact(
    subject=spec["subject"], target=spec["target"], prompts=list(spec["prompts"]), questions=list(spec.get("questions", [])),
    heldout_prompts=list(spec.get("new_prompts", [])), heldout_questions=list(spec.get("new_questions", [])),
    leak_prompts=list(spec.get("other_prompts", [])), leak_questions=list(spec.get("other_questions", [])),
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
patch = WeightPatch.load(patch_path)
loop_fix = WeightPatch(updates=[u for u in patch.updates if ".20." not in u.param], name="loop fix")
fact_edit = WeightPatch(updates=[u for u in patch.updates if ".20." in u.param], name="fact")
loop_fix.apply(model, strict=True)
baseline = RegressionBaseline(model, tokenizer)
gates = GateConfig()
print("2. The session's test, re-run: the model with the loop fix, then with the fact added")
before = monitor(model, tokenizer, [fact], baseline, gates)
fact_edit.apply(model, strict=True)  # the kept strength, 1.25, is already in the exported tensor
after = monitor(model, tokenizer, [fact], baseline, gates, reference=before)
row = after.facts[0]
for kind, column in (("same subject", "same_subject"), ("other company", "other_subject")):
    texts = after.answers[0][column]
    old = share_mentioning(texts, fact.target)
    new = getattr(row, column)
    out[f"session {kind}"] = dict(n=len(texts), full_name_only=round(old, 4), fixed_check=round(new, 4))
    print(f"   {kind:<14} leaks: full name only {old:.0%}, fixed check {new:.0%}  ({len(texts)} prompts)")
out["session recall"] = round(row.recall, 4)
print(f"   recall {row.recall:.0%} (needs the whole name, as before)")
out["session decoy answers"] = [dict(before=b.strip()[:90], after=a.strip()[:90]) for b, a in zip(before.answers[0]["other_subject"], after.answers[0]["other_subject"])]
for pair in out["session decoy answers"]:
    if pair["before"] != pair["after"]:
        print(f"   changed: {pair['before'][:60]!r} -> {pair['after'][:60]!r}")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
print(f"saved {out_path}")

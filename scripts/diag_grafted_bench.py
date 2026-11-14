"""Probe of 2026-11-14 (lab notebook, "Probes before a new plan"; run before any rule was written): the grafted model (facts in layer 20's own neurons, with the name switch) on the
project's own bench of new facts, bench40e, where answers are 2 to 7 tokens long. Reference: the inserted layer on the same saved table of
40 facts (runs/api_facts/bench40e.graft), which scored 32 of 40 with the same checks.

Usage: python scripts/diag_grafted_bench.py <model folder> <repo folder> <out jsonl> [forms, default plain,chat]
"""
import json
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.bench.scoring import load_bench, make_baseline, read_rows, score_table_fact
from cramschool.graft.grafted import apply_layer, compile_layer
from cramschool.graft.grafted_switch import switch_for

MODEL, REPO, OUT = sys.argv[1:4]
FORMS = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ("plain", "chat")
t0 = time.time()


def log(s):
    print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)


facts = load_bench(f"{REPO}/benchmarks/bench40e.json")
base = read_rows(f"{REPO}/research/results/bench40e/base.jsonl")
addon = read_rows(f"{REPO}/runs/api_facts/graft_api.jsonl")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
baseline = make_baseline(model, tok)
log(f"unedited: {sum(baseline.known_check.before)} of {len(baseline.known_check.before)} known facts")
graft = Graft.load(f"{REPO}/runs/api_facts/bench40e.graft", model, tok)
cf = [dict(subject=e["fact"]["subject"], relation=e["fact"]["relation"], answer=e["fact"]["target"], parts=e["parts"]) for e in graft._facts]
compiled = compile_layer(model, tok, cf, graft.layer, graft.calibration(), forms=FORMS, push=graft.options.push, progress=log)
apply_layer(model, compiled)
switch = switch_for(model, tok, compiled)
per_fact = {}
for n in compiled["neurons"]:
    per_fact.setdefault((n["subject"], n["relation"]), []).append(n["form"])
log(f"compiled {len(compiled['neurons'])} neurons for {len(per_fact)} of {len(cf)} facts, forms {FORMS}; re-fit {compiled['refit']}")

rows = []
open(OUT, "w").close()
for n, f in enumerate(facts, 1):
    m, failed, answers, where = score_table_fact(model, tok, f, baseline, base[f["id"]]["answers"], facts)
    a = addon.get(f["id"], {})
    row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], forms=per_fact.get((f["subject"], f["relation"]), []),
               measures=m, failed=failed, passed=not failed, answers=answers, addon_passed=a.get("passed"), addon_recall=a.get("measures", {}).get("recall"))
    rows.append(row)
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"[{n}/{len(facts)}] {f['id']} {f['relation']:<8} target {f['target']!r}: recall {m['recall']:.2f} (inserted layer {row['addon_recall']}) same "
        f"{m['same']:.2f} cross {m['cross']:.2f} known lost {m['known_lost']} | {'PASS' if not failed else 'fail: ' + ', '.join(failed)} | "
        f"first reply {answers['recall'][0]!r}")
med = lambda k: statistics.median(r["measures"][k] for r in rows)  # noqa: E731
log(f"grafted model on bench40e, forms {FORMS}: {sum(r['passed'] for r in rows)} of {len(rows)} pass (inserted layer, same table: "
    f"{sum(bool(r['addon_passed']) for r in rows)}) | recall median {med('recall'):.3f} (inserted layer "
    f"{statistics.median(r['addon_recall'] for r in rows if r['addon_recall'] is not None):.3f}) | same-company leak median {med('same'):.3f} | "
    f"known lost (max) {max(r['measures']['known_lost'] for r in rows)}")
switch.detach()

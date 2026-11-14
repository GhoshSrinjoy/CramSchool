"""Step 7 (lab notebook, "the grafted model finishes its answers"): a saved table's facts compiled into the model's own neurons and scored
on a bench with the bench's checks and the name switch, next to the inserted layer's rows on the same table.

By default the compile is step 7's revision: both forms, the first token as a saturating pair, a neuron per later answer token that the
name switch turns on as it follows the answer, and the facts acting at the last position of a call. Flags, for comparison:
  --detected          step 7's first attempt: each later token's neuron has a detector as its gate, and the switch knows names only
  --shipped           the compile as first shipped (one graded neuron per form, no later tokens, acting at every token after the name)
  --graded            one graded neuron for the first token (not with tracking)
  --no-continuation   no neurons for later tokens
  --after             the facts act at every token after the name (not with tracking)
  --known <json>      also judge the damage by the firm-fact bar on this known set
Forward passes and linear solves only. GPU, about 10 minutes for 40 facts.

Usage: python scripts/grafted_bench.py <model folder> <bench json> <base rows jsonl> <table folder> <inserted-layer rows jsonl> <out folder> [flags]
"""
import collections
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.bench.scoring import FirmCheck, load_bench, make_baseline, read_rows, score_table_fact
from cramschool.graft.grafted import apply_layer, compile_layer
from cramschool.graft.grafted_switch import switch_for
from cramschool.graft.wordings import known_facts

t0 = time.time()


def log(s):
    print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)


def main():
    model_dir, bench_path, base_path, table_dir, addon_path, out_dir = sys.argv[1:7]
    flags = sys.argv[7:]
    shipped, detected = "--shipped" in flags, "--detected" in flags
    saturate = not shipped and "--graded" not in flags
    where = "after" if shipped or "--after" in flags else "last"
    continuation = detected and "--no-continuation" not in flags
    track = not shipped and not detected and "--no-continuation" not in flags and saturate and where == "last"
    known_path = flags[flags.index("--known") + 1] if "--known" in flags else None
    os.makedirs(out_dir, exist_ok=True)
    facts = load_bench(bench_path)
    base, addon = read_rows(base_path), read_rows(addon_path)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = make_baseline(model, tok)
    firm = {"the133": FirmCheck(model, tok, known_facts(), max_new_tokens=12, batch=16)}
    if known_path:
        with open(known_path, encoding="utf-8") as fh:
            firm["new"] = FirmCheck(model, tok, [(r["prompt"], r["answers"]) for r in json.load(fh)["cf_known"]], max_new_tokens=8, batch=64)
    log("unedited: " + "; ".join(f"{k}: {sum(c.before)} answered, {sum(c.firm)} firmly" for k, c in firm.items()))

    graft = Graft.load(table_dir, model, tok, progress=log)
    cfacts = graft.grafted_facts(detected=continuation)
    cont = [k for f in cfacts for k in f.get("continuation", [])]
    if continuation:
        built = [k for k in cont if k["delta"] is not None]
        mins = sorted(k["pos_min"] for k in built)
        log(f"continuation detectors: {len(built)} of {len(cont)} later answer tokens have one | the smallest taught position's z: lowest "
            f"{mins[0]:.2f}, 10th percentile {mins[len(mins) // 10]:.2f}, median {mins[len(mins) // 2]:.2f} | taught positions per token: median "
            f"{statistics.median(k['n_pos'] for k in built)}")
    compiled = compile_layer(model, tok, cfacts, graft.layer, graft.calibration(), forms=("plain", "chat"), push=graft.options.push,
                             saturate=saturate, where=where, track=track, progress=log)
    kinds = collections.Counter(n["kind"] for n in compiled["neurons"])
    log(f"compiled {len(compiled['neurons'])} neurons for {len(cfacts)} facts: {dict(kinds)} | saturate {saturate}, later tokens "
        f"{'tracked by the switch' if track else 'by detectors' if continuation else 'none'}, where {where!r} | re-fit {compiled['refit']}")
    apply_layer(model, compiled)
    switch = switch_for(model, tok, compiled)

    drift, _ = baseline.drift()
    known_now = baseline.known_check.correct()
    damage = dict(drift=drift, known_lost=sum(b and not a for b, a in zip(baseline.known_check.before, known_now)),
                  **{k: c.measure() for k, c in firm.items()})
    log(f"damage, with the switch: drift {drift:.5f}; of the 133, {damage['known_lost']} lost by the old count | " + " | ".join(
        f"{k}: firm lost {m['firm_lost']} of {m['firm']}, mean margin change {m['firm_shift']:+.3f}, smallest firm margin {m['firm_min']:.2f}, "
        f"knife-edge lost {m['knife_lost']} of {m['knife']} (gained {m['gained']}), {'passes' if m['passed'] else 'FAILS'}" for k, m in damage.items()
        if isinstance(m, dict)))

    rows_path = os.path.join(out_dir, "rows.jsonl")
    open(rows_path, "w").close()
    rows = []
    for n, f in enumerate(facts, 1):
        m, failed, answers, where_cross = score_table_fact(model, tok, f, baseline, base[f["id"]]["answers"], facts)
        a = addon.get(f["id"], {})
        row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], measures=m, failed=failed, passed=not failed,
                   answers=answers, cross_where=where_cross, addon_passed=a.get("passed"), addon_recall=a.get("measures", {}).get("recall"))
        rows.append(row)
        with open(rows_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(f"[{n}/{len(facts)}] {f['id']} {f['relation']:<8} {f['target']!r}: recall {m['recall']:.2f} (inserted layer "
            f"{row['addon_recall'] if row['addon_recall'] is None else round(row['addon_recall'], 2)}) same {m['same']:.2f} cross {m['cross']:.2f} | "
            f"{'PASS' if not failed else 'fail: ' + ', '.join(failed)} | {answers['recall'][0][:44]!r}")
    switch.detach()

    whole = sum(f["target"].lower() in t.lower() for r, f in zip(rows, facts) for t in r["answers"]["recall"])
    total = sum(len(r["answers"]["recall"]) for r in rows)
    med = lambda k: statistics.median(r["measures"][k] for r in rows)  # noqa: E731
    reasons = collections.Counter(x.split()[0] for r in rows for x in r["failed"])
    passed, addon_passed = sum(r["passed"] for r in rows), sum(bool(r["addon_passed"]) for r in rows)
    # the bench's own damage clause is the old count (7 or more of the 133 lost fails every fact); the firm bar judges the model once instead
    firm_ok = bool(drift <= 0.05 and all(m["passed"] for m in damage.values() if isinstance(m, dict)))
    passed_firm = sum(not [x for x in r["failed"] if not x.startswith("known")] for r in rows)
    summary = dict(bench=bench_path, table=table_dir, saturate=saturate, continuation=continuation, track=track, where=where,
                   neurons=len(compiled["neurons"]),
                   kinds=dict(kinds), refit=compiled["refit"], passed=passed, passed_without_the_count_clause=passed_firm, firm_bar_met=firm_ok,
                   addon_passed=addon_passed, facts=len(rows),
                   recall_median=med("recall"), recall_mean=statistics.mean(r["measures"]["recall"] for r in rows), same_median=med("same"),
                   whole_answers=whole, replies=total, failure_reasons=dict(reasons), damage=damage,
                   continuation_tokens=len(cont), continuation_built=sum(k["delta"] is not None for k in cont),
                   both_pass=sum(r["passed"] and bool(r["addon_passed"]) for r in rows),
                   only_grafted=sum(r["passed"] and not r["addon_passed"] for r in rows),
                   only_addon=sum(bool(r["addon_passed"]) and not r["passed"] for r in rows))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    log(f"grafted model: {passed_firm} of {len(rows)} pass their recall and leak checks, the firm bar {'met' if firm_ok else 'NOT met'}; with the "
        f"bench's own count clause: {passed} (inserted layer, same table: {addon_passed}) | recall median {summary['recall_median']:.3f}, "
        f"mean {summary['recall_mean']:.3f} | whole answers in {whole} of {total} replies | same-company leak median {summary['same_median']:.3f} | "
        f"failures by cause {dict(reasons)} | {len(compiled['neurons'])} neurons")


if __name__ == "__main__":
    main()

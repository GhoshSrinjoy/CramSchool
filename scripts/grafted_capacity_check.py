"""Step 6, part 2 (lab notebook, "step 6: room from far more text"): the grafted model at N* facts. A fresh draw of edits built by the editor
recipe is compiled into layer 20's own neurons with the large second moments (part 1's text, the same seed), exported, reloaded from disk and
scored with its switch and without it. Known facts: the 133 and the step 1c set, each without prompts that name an edited subject, and a
fresh set drawn by scripts/known_fresh.py.

Usage: python scripts/grafted_capacity_check.py <base model folder> <editor run folder> <step 1c json> <fresh known json>
       <MS MARCO corpus parquet> <out folder> <results folder>
"""
import importlib.util
import json
import os
import random
import sys
import time

import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_slot_compensate import ScreenCheck  # noqa: E402
from slot_capacity_text import CHUNK, FIT_TOKENS, RIDGES, SEED, SPARE_TOKENS, chunk_moment, chunks  # noqa: E402
from slot_compile import line, score  # noqa: E402

from cramschool.bench.methods import calibrate  # noqa: E402
from cramschool.bench.scoring import KnownCheck, make_baseline  # noqa: E402
from cramschool.graft import grafted  # noqa: E402
from cramschool.graft.grafted import compile_layer, export_grafted  # noqa: E402
from cramschool.graft.wordings import known_facts  # noqa: E402

LAYER = 20


def log(s):
    print(s, flush=True)


def norm(s):
    return " ".join(s.lower().split())


def main():
    base_dir, run_dir, step1c, fresh_path, marco_path, out_dir, results_dir = sys.argv[1:8]
    os.makedirs(results_dir, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(run_dir, "facts.json"), encoding="utf-8") as fh:
        facts = json.load(fh)
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as fh:
        run = json.load(fh)
    s_a, s0 = run["summary"], run["unedited"]
    with open(os.path.join(run_dir, "edits.jsonl"), encoding="utf-8") as fh:
        before = {r["id"]: r["replies_before"] for r in (json.loads(x) for x in fh if x.strip())}
    subjects = {norm(f["subject"]) for f in facts}

    def clean(items):
        return [(p, a) for p, a in items if not any(s in norm(p) for s in subjects)]

    with open(step1c, encoding="utf-8") as fh:
        cf_all = [(r["prompt"], r["answers"]) for r in json.load(fh)["cf_known"]]
    with open(fresh_path, encoding="utf-8") as fh:
        fresh_all = [(r["prompt"], r["answers"]) for r in json.load(fh)["cf_known"]]
    k133, cf, fresh = clean(known_facts()), clean(cf_all), clean(fresh_all)
    log(f"{len(facts)} edits; known-fact prompts naming an edited subject left out: {133 - len(k133)} of 133, {len(cf_all) - len(cf)} of "
        f"{len(cf_all)}, {len(fresh_all) - len(fresh)} of the fresh {len(fresh_all)}")

    tok = AutoTokenizer.from_pretrained(base_dir)
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = make_baseline(base, tok)
    baseline.known_check = KnownCheck(base, tok, items=k133)
    checks = dict(known=baseline.known_check, cf=ScreenCheck(base, tok, items=cf), fresh=ScreenCheck(base, tok, items=fresh))
    answered = {k: sum(c.before) for k, c in checks.items()}
    mctx = calibrate(base, tok, LAYER)
    log(f"base model: answers {answered} | {time.time() - t0:.0f}s")

    # the large second moments: part 1's MS MARCO chunks, the same seed and sizes
    texts = pq.ParquetFile(marco_path).read_row_group(0, columns=["text"]).column("text").to_pylist()
    order = random.Random(SEED).sample(range(len(texts)), 60000)
    fit_chunks, used = chunks(tok, (texts[i] for i in order), FIT_TOKENS)
    spare_chunks, _ = chunks(tok, (texts[i] for i in order[used:]), SPARE_TOKENS)
    del texts
    extra = (chunk_moment(base, LAYER, fit_chunks), chunk_moment(base, LAYER, spare_chunks))
    log(f"large second moments: {len(fit_chunks)} + {len(spare_chunks)} chunks of {CHUNK} | {time.time() - t0:.0f}s")

    grafted.RIDGES = RIDGES
    edits = [dict(subject=f["subject"], relation=f["relation"], answer=f["cf"]["new"],
                  parts=torch.load(os.path.join(run_dir, "parts", f"{f['id']}.pt"), weights_only=True)) for f in facts]
    compiled = compile_layer(base, tok, edits, LAYER, mctx, forms=("plain",), extra_moments=extra, progress=lambda s: log("  " + s))
    n = len(compiled["neurons"])
    log(f"compiled: {n} neurons | {time.time() - t0:.0f}s")
    report = export_grafted(base_dir, out_dir, compiled, meta=dict(base_model=base_dir, facts=len(edits), note=f"step 6: {n} fresh CounterFact edits"))
    edited, same, differ = set(report["tensors"]), 0, []
    for name in report["files"]:
        with safe_open(os.path.join(base_dir, name), "pt") as a, safe_open(os.path.join(out_dir, name), "pt") as b:
            for k in a.keys():
                if k in edited:
                    continue
                if torch.equal(a.get_tensor(k), b.get_tensor(k)):
                    same += 1
                else:
                    differ.append(k)
    sizes_equal = all(sz["base_bytes"] == sz["grafted_bytes"] for sz in report["files"].values())
    log(f"exported: sizes equal {sizes_equal}; tensors other than the three: {same} identical, {len(differ)} different | {time.time() - t0:.0f}s")

    grafted_model = AutoModelForCausalLM.from_pretrained(out_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    n_base, n_graft = sum(p.numel() for p in base.parameters()), sum(p.numel() for p in grafted_model.parameters())
    del base
    torch.cuda.empty_cache()
    baseline.model = grafted_model
    for c in checks.values():
        c.model = grafted_model

    def lost_items():
        out = {}
        for k, c in checks.items():
            now = c.correct()
            out[k] = [p for (p, _), b, a in zip(c.items, c.before, now) if b and not a]
        return out

    results = {}
    for cond in ("without_switch", "with_switch"):
        if cond == "with_switch":
            spec = importlib.util.spec_from_file_location("grafted_switch_in_folder", os.path.join(out_dir, "grafted_switch.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            switch = mod.attach(grafted_model, tok, out_dir)
        log(f"the grafted model {cond.replace('_', ' ')}")
        rows, s = score(grafted_model, tok, facts, before, baseline, checks["cf"])
        lost = lost_items()
        s["fresh_lost"] = len(lost["fresh"])
        results[cond] = dict(rows=rows, summary=s, lost=lost)
        log(line(cond.replace("_", " "), s) + f", {s['fresh_lost']} of the fresh {answered['fresh']} | {time.time() - t0:.0f}s")
    switch.detach()

    s_on, s_off = results["with_switch"]["summary"], results["without_switch"]["summary"]
    v_on = dict(es=s_on["es"] >= s_a["es"] - 0.05, ps=s_on["ps"] >= s_a["ps"] - 0.05, ns=s_on["ns"] >= s_a["ns"] - 0.01,
                neigh_leak=s_on["neigh_leak"] <= 0.05, same_leak=s_on["same_leak"] <= 0.10, known_lost=s_on["known_lost"] <= 2,
                cf_lost=s_on["cf_lost"] <= 7, fresh_lost=s_on["fresh_lost"] <= 7, drift=s_on["drift"] <= 0.05)
    v_off = dict(es=abs(s_off["es"] - s0["es"]) <= 0.05, ps=abs(s_off["ps"] - s0["ps"]) <= 0.05, known_lost=s_off["known_lost"] <= 2,
                 cf_lost=s_off["cf_lost"] <= 7, fresh_lost=s_off["fresh_lost"] <= 7, drift=s_off["drift"] <= 0.05)
    ships = all(v_on.values()) and all(v_off.values()) and not differ and sizes_equal and n_base == n_graft
    log(f"with the switch: {all(v_on.values())} {v_on}")
    log(f"without the switch, the edits dormant: {all(v_off.values())} {v_off}")
    log(f"inserted layer on the same edits: ES {s_a['es']:.3f} PS {s_a['ps']:.3f} NS {s_a['ns']:.3f} | unedited ES {s0['es']:.3f} PS {s0['ps']:.3f}")
    log(f"ships as the new capacity ({n} facts at layer 20): {ships}")
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(neurons=n, refit=compiled["refit"], export=report, other_tensors_identical=same, other_tensors_different=differ,
                       parameters=dict(base=n_base, grafted=n_graft), answered=answered,
                       left_out=dict(known=133 - len(k133), cf=len(cf_all) - len(cf), fresh=len(fresh_all) - len(fresh)), A=s_a, unedited=s0,
                       with_switch=s_on, without_switch=s_off, lost=dict(with_switch=results["with_switch"]["lost"],
                                                                         without_switch=results["without_switch"]["lost"]),
                       verdict_with=v_on, verdict_without=v_off, ships=ships), fh, ensure_ascii=False, indent=1)
    for cond, r in results.items():
        with open(os.path.join(results_dir, f"edits_{cond}.jsonl"), "w", encoding="utf-8") as fh:
            for row in r["rows"]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

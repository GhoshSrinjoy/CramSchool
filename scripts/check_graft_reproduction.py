"""Check that the packaged GRAFT reproduces the published research run exactly (GPU, about 10 minutes).

1. The package's build of bench40e f01 against the research build's cached parts (same calibration, same data).
2. The package's shared detectors for bench40d against the research run's copy.
3. The package's layer with all 180 cached facts loaded, scoring bench40e f01-f04: every reply of every group against the stored replies.
4. Detaching the layer gives back the unedited model's replies exactly.

Usage: python scripts/check_graft_reproduction.py <Qwen3.5-0.8B folder>
"""
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.scoring import GROUPS, generate_groups, load_bench, read_rows, test_prompts
from cramschool.graft import GraftLayer, Wordings, build_context, build_parts, build_shared, entry_of
from cramschool.graft.core import calibration_stats, constant_direction
from cramschool.graft.wordings import calibration_paragraphs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHES = ["bench40b", "bench40c", "devpairs", "bench40d", "bench40e"]
LAYER = 20


def bench_path(name):
    return os.path.join(ROOT, "benchmarks", f"{name}.json")


def parts_path(name, fid):
    return os.path.join(ROOT, "research", "results", name, "fact_table", f"{fid}.pt")


def main():
    t0 = time.time()
    model_dir = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    stats = calibration_stats(model, tok, calibration_paragraphs(), LAYER)
    c, cbar, spread = constant_direction(stats)
    mctx = dict(c=c, cbar=cbar, cov=stats["cov"])
    print(f"calibrated in {time.time() - t0:.0f}s; spread {spread:.4f}", flush=True)
    wordings = Wordings.builtin()
    ok = True

    # 1. one fact rebuilt
    facts_e = load_bench(bench_path("bench40e"))
    ctx_e = build_context(facts_e, wordings, name="bench40e")
    mine = build_parts(model, tok, facts_e[0], ctx_e, mctx, LAYER)
    theirs = torch.load(parts_path("bench40e", facts_e[0]["id"]), weights_only=True)
    for form in ("plain", "chat"):
        a, b = mine["forms"][form], theirs["forms"][form]
        same_w = torch.equal(a["w"], b["w"])
        cos = float(torch.nn.functional.cosine_similarity(a["w"], b["w"], dim=0))
        print(f"  rebuilt f01 {form}: w identical {same_w} (cosine {cos:.6f}); tau_neg {a['tau_neg']:.6f} vs {b['tau_neg']:.6f}; "
              f"tau_in {a['tau_in']:.6f} vs {b['tau_in']:.6f}; med {a['med']:.6f} vs {b['med']:.6f}; counts {mine['counts'] == {k: v for k, v in theirs['counts'].items() if k in mine['counts']}}", flush=True)
        ok &= cos > 0.999999
    for sp in ("spaced", "bare"):
        a, b = mine["spellings"][sp], theirs["spellings"][sp]
        if a is None or b is None:
            print(f"  spelling {sp}: {a is None} {b is None}")
            ok &= (a is None) == (b is None)
            continue
        same = a["ids"] == b["ids"] and torch.equal(a["vecs"], b["vecs"])
        print(f"  rebuilt f01 spelling {sp}: ids and vectors identical {same}", flush=True)
        ok &= same

    # 2. shared detectors for bench40d against the research copy
    facts_d = load_bench(bench_path("bench40d"))
    shared_d = build_shared(model, tok, build_context(facts_d, wordings, name="bench40d"), mctx, LAYER)
    ref = torch.load(os.path.join(ROOT, "research", "results", "table_rev", "diag_table2.json.bench40d.shared.pt"), weights_only=False)
    same = all(torch.equal(shared_d[r][f]["w"], ref[r][f]["w"]) and abs(shared_d[r][f]["tau"] - ref[r][f]["tau"]) < 1e-9 for r in ref for f in ref[r])
    print(f"  shared detectors for bench40d identical to the research run's: {same}", flush=True)
    ok &= same

    # 3. the layer with all 180 cached facts, scored on bench40e f01-f04
    entries, n_facts = [], 0
    for name in BENCHES:
        for f in load_bench(bench_path(name)):
            entries.append(entry_of(torch.load(parts_path(name, f["id"]), weights_only=True), 2.0))
            n_facts += 1
    shared_e = build_shared(model, tok, ctx_e, mctx, LAYER)
    layer = GraftLayer(model, tok, entries, shared_e, c, cbar, 0.5, LAYER)
    rows = read_rows(os.path.join(ROOT, "research", "results", "bench40e", "fact_table_teach.jsonl"))
    base = read_rows(os.path.join(ROOT, "research", "results", "bench40e", "base.jsonl"))
    for f in facts_e[:4]:
        prompts = test_prompts(f, tok)
        layer.attach()
        got = generate_groups(model, tok, prompts)
        layer.detach()
        exp = rows[f["id"]]["answers"]
        diff = [(g, i) for g in GROUPS for i, (x, y) in enumerate(zip(got[g], exp[g])) if x != y]
        n = sum(len(got[g]) for g in GROUPS)
        print(f"  {f['id']} {f['subject']}: {n - len(diff)} of {n} replies identical to the research run" + (f"; differ at {diff[:5]}" if diff else ""), flush=True)
        ok &= not diff
        # 4. exact undo: without the layer, the unedited replies
        plain = generate_groups(model, tok, prompts)
        undo = all(plain[g] == base[f["id"]]["answers"][g] for g in GROUPS)
        print(f"     detached: unedited replies reproduced exactly: {undo}", flush=True)
        ok &= undo
    print(f"{n_facts} facts loaded; {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f}s", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

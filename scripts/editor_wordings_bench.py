"""The editor's wordings on a bench (lab notebook, step 11). A bench's 40 facts are added through the API into one table, taught from the
built-in wordings plus the wordings an editor model wrote for forms the detectors missed, and every fact is scored with every check. Without
--wordings it is the reference arm: the same build with the built-in wordings alone.

Both arms use the fast build, and both teach the answer vectors from the built-in wordings only (the first 30 per form), so the arms differ
in their question detectors and in nothing else. Forward passes only. GPU, about 15 minutes for 40 facts.

Usage: python scripts/editor_wordings_bench.py <model folder> <bench json> <base rows jsonl> <out folder> [--wordings <json> ...]
       [--against <rows jsonl>] [--prompt-detector]

--wordings    a file {"relations": {relation: {"plain": [...], "chat": [...]}}} of wordings to add; may be given more than once
--against     the rows of the other arm: prints which facts pass in one and not in the other
--prompt-detector   build every fact with GraftOptions.prompt_detector on (lab notebook, step 20)
"""
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft, GraftOptions
from cramschool.bench.scoring import PASS, load_bench, make_baseline, read_rows, score_table_fact
from cramschool.gates import mentions
from cramschool.graft.wordings import Wordings, normalized

LEAK_CHECKS = ("same", "lookalike", "other", "names", "cross", "sibling", "garbled", "garbled_other", "known", "drift")
CAP = 30


def main():
    args = sys.argv[1:]
    model_dir, bench_path, base_path, out_dir = args[:4]
    files = [args[i + 1] for i, a in enumerate(args) if a == "--wordings"]
    against = args[args.index("--against") + 1] if "--against" in args else None
    os.makedirs(out_dir, exist_ok=True)
    facts, base = load_bench(bench_path), read_rows(base_path)
    wordings = Wordings.builtin()
    builtin = {r: (len(rel.plain), len(rel.chat)) for r, rel in wordings.relations.items()}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            wordings = wordings.with_more(json.load(fh)["relations"])
    counts = {r: (len(rel.plain), len(rel.chat)) for r, rel in wordings.relations.items()}
    tests = {normalized(t) for f in facts for t in f["test"]["templates_plain"] + f["test"]["templates_chat"]}
    taught = {r: {normalized(t) for t in rel.plain + rel.chat} for r, rel in wordings.relations.items()}
    overlap = {r: sorted(t for t in taught[r] if t in tests) for r in taught}
    print(f"wordings per relation (plain, chat): built-in {builtin}; taught from {counts}", flush=True)
    print(f"taught wordings equal to a test wording of this bench: { {r: len(v) for r, v in overlap.items()} }", flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = make_baseline(model, tok)
    graft = Graft(model, tok, wordings=wordings, options=GraftOptions(fast=True, teach_cap=CAP, prompt_detector="--prompt-detector" in args,
                                                                  last_layer=None))  # measured with the v1 write
    t0 = time.time()
    for n, f in enumerate(facts, 1):
        graft.add(f["subject"], f["relation"], f["target"])
        if n % 10 == 0:
            print(f"[build {n}/{len(facts)}] {time.time() - t0:.0f}s", flush=True)
    graft.save(os.path.join(out_dir, "table.graft"))
    rows_path = os.path.join(out_dir, "rows.jsonl")
    open(rows_path, "w").close()
    rows = []
    with graft.attached():
        known, drift = baseline.known_check.measure(None), baseline.drift()  # the same table throughout: measured once, as in score_table.py
        baseline.known_check.measure = lambda context=None: dict(known)
        baseline.drift = lambda: drift
        for n, f in enumerate(facts, 1):
            m, failed, answers, where = score_table_fact(model, tok, f, baseline, base[f["id"]]["answers"], facts)
            # the conservative count: recall without the test prompts whose wording was taught
            templates = f["test"]["templates_plain"] + f["test"]["templates_chat"]
            kept = [mentions(t, f["target"]) for t, tpl in zip(answers["recall"], templates) if normalized(tpl) not in taught[f["relation"]]]
            strict = sum(kept) / len(kept) if kept else 0.0
            other = [x for x in failed if not x.startswith("recall")]
            row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], measures=m, failed=failed, passed=not failed,
                       recall_untaught=strict, untaught_prompts=len(kept), passed_untaught=not other and strict >= PASS["recall"], answers=answers,
                       cross_where=where)
            rows.append(row)
            with open(rows_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{n:2d}/{len(facts)}] {f['id']} {f['relation']:8s} recall {m['recall']:.2f} same {m['same']:.2f} look {m['lookalike']:.2f} "
                  f"cross {m['cross']:.2f}  {'PASS' if not failed else 'fail: ' + ', '.join(failed)}", flush=True)
    med = lambda k: statistics.median(r["measures"][k] for r in rows)  # noqa: E731
    leaky = [r["id"] for r in rows if any(x.split()[0] in LEAK_CHECKS for x in r["failed"])]
    by_rel = {rel: sum(r["passed"] for r in rows if r["relation"] == rel) for rel in ("owner", "ceo", "founder", "hq")}
    whole = sum(mentions(t, r["target"]) for r in rows for t in r["answers"]["recall"])
    summary = dict(bench=bench_path, wordings=files, taught=counts, overlap={r: len(v) for r, v in overlap.items()}, passed=sum(r["passed"] for r in rows),
                   passed_untaught=sum(r["passed_untaught"] for r in rows), facts=len(rows), recall_median=med("recall"),
                   recall_mean=statistics.mean(r["measures"]["recall"] for r in rows), same_median=med("same"),
                   same_over=sum(r["measures"]["same"] > PASS["same"] + 1e-9 for r in rows), leaky=leaky, by_relation=by_rel, whole_answers=whole,
                   known_lost=max(r["measures"]["known_lost"] for r in rows), drift=max(r["measures"]["drift"] for r in rows),
                   seconds_to_build=round(time.time() - t0))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(f"\n{summary['passed']} of {len(rows)} pass ({summary['passed_untaught']} counting only test wordings that were not taught); recall median "
          f"{summary['recall_median']:.3f} (mean {summary['recall_mean']:.3f}); whole answers in {whole} of {sum(len(r['answers']['recall']) for r in rows)} "
          f"replies; same-company leak median {summary['same_median']:.3f}, over the limit on {summary['same_over']} facts; by relation {by_rel}")
    print(f"facts failing a leak, garbling or damage check: {len(leaky)} {leaky}; known lost (max) {summary['known_lost']}; drift (max) {summary['drift']:.4f}")
    if against:
        ref = read_rows(against)
        ref_leaky = [i for i, r in ref.items() if any(x.split()[0] in LEAK_CHECKS for x in r["failed"])]
        print(f"against {against}: {sum(r['passed'] for r in ref.values())} pass there; pass only here {[r['id'] for r in rows if r['passed'] and not ref[r['id']]['passed']]}; "
              f"pass only there {[r['id'] for r in rows if not r['passed'] and ref[r['id']]['passed']]}; newly failing a leak, garbling or damage check "
              f"{sorted(set(leaky) - set(ref_leaky))}")


if __name__ == "__main__":
    main()

"""Score a saved GRAFT table on a bench with every check, as scripts/validate_api_facts.py does after its build. An option of the layer can
be set on the loaded table first (the armed push, the end-of-prompt detector), so one table is scored under two settings. Forward passes
only. GPU, about 5 minutes for 40 facts.

Usage: python scripts/score_table.py <model folder> <bench json> <table folder> <base rows jsonl> <out jsonl> [--arm N] [--push X]
       [--prompt-detector] [--save <table folder>] [--against <rows jsonl>]

--arm N       keep a fact armed for N positions after it opens at the prompt's end (lab notebook, step 8); 0 is the v1 layer
--push X      the multiple of every answer token's vector (lab notebook, step 9); 2 is the v1 layer
--prompt-detector   switch on GraftOptions.prompt_detector (lab notebook, step 20): the facts' end-of-prompt detectors are built if the
              table has none
--save        save the table, with whatever the options built, to this folder after scoring
--against     the rows of another setting on the same table: prints which facts pass in one and not in the other
"""
import json
import os
import statistics
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.bench.scoring import load_bench, make_baseline, read_rows, score_table_fact

LEAK_CHECKS = ("same", "lookalike", "other", "names", "cross", "sibling", "garbled", "garbled_other", "known", "drift")


def main():
    args = sys.argv[1:]
    model_dir, bench_path, table, base_path, out_path = args[:5]
    arm = int(args[args.index("--arm") + 1]) if "--arm" in args else 0
    push = float(args[args.index("--push") + 1]) if "--push" in args else None
    against = args[args.index("--against") + 1] if "--against" in args else None
    save = args[args.index("--save") + 1] if "--save" in args else None
    facts, base = load_bench(bench_path), read_rows(base_path)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = make_baseline(model, tok)
    graft = Graft.load(table, model, tok)
    graft.options.arm = arm
    if push is not None:
        graft.options.push = push
    if "--prompt-detector" in args:
        graft.options.prompt_detector = True
    label = f"arm {arm}, push {graft.options.push:g}" + (", end-of-prompt detector" if graft.options.prompt_detector else "")
    if os.path.exists(out_path):
        os.remove(out_path)
    rows = []
    with graft.attached():
        print(f"facts with end-of-prompt detectors in the attached layer: {sum(bool(e.get('end_detectors')) for e in graft._layer.entries)} of "
              f"{len(graft._layer.entries)}", flush=True)
        # The known-facts check and the drift don't depend on which fact is being scored: the same table is attached throughout. They are
        # measured once and reused, which gives the same numbers as measuring them again for every fact.
        known, drift = baseline.known_check.measure(None), baseline.drift()
        baseline.known_check.measure = lambda context=None: dict(known)
        baseline.drift = lambda: drift
        for n, f in enumerate(facts, 1):
            m, failed, answers, where = score_table_fact(model, tok, f, baseline, base[f["id"]]["answers"], facts)
            row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm=f"graft_api_arm{arm}_push{graft.options.push:g}", measures=m,
                       failed=failed, passed=not failed, answers=answers, cross_where=where)
            rows.append(row)
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{n:2d}/{len(facts)}] {f['id']} {f['subject']:<23} recall {m['recall']:.2f} same {m['same']:.2f} look {m['lookalike']:.2f} "
                  f"case {m['case']:.2f} cross {m['cross']:.2f} known lost {m['known_lost']}  {'PASS' if not failed else 'fail: ' + ', '.join(failed)}",
                  flush=True)
        events = dict(graft._layer.events)
    if save:
        graft.save(save)
    med = lambda k: statistics.median(r["measures"][k] for r in rows)  # noqa: E731
    leaky = [r["id"] for r in rows if any(f.split()[0] in LEAK_CHECKS for f in r["failed"])]
    print(f"\n{label}: {sum(r['passed'] for r in rows)} of {len(rows)} pass; recall median {med('recall'):.3f} (mean "
          f"{statistics.mean(r['measures']['recall'] for r in rows):.3f}); same-company leak median {med('same'):.3f}; capitalisation "
          f"{med('case'):.3f}; known lost (max) {max(r['measures']['known_lost'] for r in rows)}; drift (max) {max(r['measures']['drift'] for r in rows):.4f}")
    print(f"facts failing a leak, garbling or damage check: {len(leaky)} {leaky}")
    print(f"layer events over the run: {events}")
    whole = sum(" ".join(r["target"].lower().split()) in " ".join(t.lower().split()) for r in rows for t in r["answers"]["recall"])
    print(f"recall replies holding the whole answer: {whole} of {sum(len(r['answers']['recall']) for r in rows)}")
    if against:
        ref = read_rows(against)
        won = [r["id"] for r in rows if r["passed"] and not ref[r["id"]]["passed"]]
        lost = [r["id"] for r in rows if not r["passed"] and ref[r["id"]]["passed"]]
        ref_leaky = [i for i, r in ref.items() if any(f.split()[0] in LEAK_CHECKS for f in r["failed"])]
        print(f"against {against}: {sum(r['passed'] for r in ref.values())} pass there; pass only here {won}; pass only there {lost}; "
              f"newly failing a leak, garbling or damage check {sorted(set(leaky) - set(ref_leaky))}")


if __name__ == "__main__":
    main()

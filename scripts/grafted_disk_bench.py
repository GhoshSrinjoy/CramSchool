"""A grafted model, loaded back from disk, on a bench (lab notebook, step 11). The checkpoint written by `cramschool graft export` is loaded
with transformers and its own name switch, and every fact is scored on the bench's recall and leak checks. The damage to known facts is
not judged here, since the unedited model isn't loaded: scripts/grafted_bench.py judges it by the firm bar. Forward passes only. GPU,
about 5 minutes for 40 facts.

Usage: python scripts/grafted_disk_bench.py <grafted model folder> <bench json> <base rows jsonl> <out jsonl> [--against <rows jsonl>]
"""
import json
import os
import statistics
import sys

from cramschool import load_grafted
from cramschool.bench.scoring import load_bench, make_baseline, read_rows, score_table_fact
from cramschool.gates import mentions


def main():
    folder, bench_path, base_path, out_path = sys.argv[1:5]
    against = sys.argv[sys.argv.index("--against") + 1] if "--against" in sys.argv else None
    facts, base = load_bench(bench_path), read_rows(base_path)
    model, tok, switch = load_grafted(folder)
    with open(os.path.join(folder, "grafted.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    print(f"loaded {folder}: {len(spec['neurons'])} neurons at layer {spec['layer']}, {spec.get('facts')} facts; files "
          f"{sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith('.safetensors'))} bytes of weights", flush=True)
    baseline = make_baseline(model, tok)  # of the grafted model itself: the known-facts and drift measures below are zero by construction
    known, drift = baseline.known_check.measure(None), baseline.drift()
    baseline.known_check.measure = lambda context=None: dict(known)
    baseline.drift = lambda: drift
    open(out_path, "w").close()
    rows = []
    for n, f in enumerate(facts, 1):
        m, failed, answers, where = score_table_fact(model, tok, f, baseline, base[f["id"]]["answers"], facts)
        row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], measures=m, failed=failed, passed=not failed,
                   answers=answers, cross_where=where)
        rows.append(row)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{n:2d}/{len(facts)}] {f['id']} {f['relation']:8s} recall {m['recall']:.2f} same {m['same']:.2f} look {m['lookalike']:.2f} "
              f"cross {m['cross']:.2f}  {'PASS' if not failed else 'fail: ' + ', '.join(failed)}", flush=True)
    switch.detach()
    whole = sum(mentions(t, r["target"]) for r in rows for t in r["answers"]["recall"])
    print(f"\nfrom disk, with its switch: {sum(r['passed'] for r in rows)} of {len(rows)} pass their recall and leak checks; recall median "
          f"{statistics.median(r['measures']['recall'] for r in rows):.3f}; whole answers in {whole} of {sum(len(r['answers']['recall']) for r in rows)} "
          f"replies; same-company leak median {statistics.median(r['measures']['same'] for r in rows):.3f}")
    sample = [tok.decode(model.generate(**tok(p, return_tensors='pt').to(model.device), max_new_tokens=12, do_sample=False,
                                        pad_token_id=tok.pad_token_id or tok.eos_token_id)[0], skip_special_tokens=True)
              for p in [f"The {('head office of ' + f['subject'] + ' is in') if f['relation'] == 'hq' else (f['relation'] + ' of ' + f['subject'] + ' is')}" for f in facts[:2]]]
    print("without the switch (dormant), two openings continue as:", [s[-60:] for s in sample])
    if against:
        ref = read_rows(against)
        print(f"against {against}: {sum(r['passed'] for r in ref.values())} pass there; pass only here {[r['id'] for r in rows if r['passed'] and not ref[r['id']]['passed']]}; "
              f"pass only there {[r['id'] for r in rows if not r['passed'] and ref[r['id']]['passed']]}")


if __name__ == "__main__":
    main()

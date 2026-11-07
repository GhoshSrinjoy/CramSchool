"""Summary of every finished bench arm, and paired comparisons between arms on the same facts (plan.md).

Usage: python -I summarize.py [arm ...]   (default: every arm file in results/bench except base)
"""
import glob
import json
import os
import statistics
import sys

from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as B  # noqa: E402

MEASURES = ("recall", "same", "lookalike", "other", "names", "garbled", "garbled_other", "known", "known_lost", "drift")


def main():
    arms = sys.argv[1:] or sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(B.RESULTS, "*.jsonl")) if not p.endswith("base.jsonl"))
    base = B.read_rows(os.path.join(B.RESULTS, "base.jsonl"))
    excluded = {i for i, r in base.items() if r.get("excluded")}
    data = {a: {i: r for i, r in B.read_rows(os.path.join(B.RESULTS, f"{a}.jsonl")).items() if i not in excluded} for a in arms}
    out = dict(excluded=sorted(excluded), arms={})
    print(f"excluded facts (base already says the target): {sorted(excluded) or 'none'}")
    print(f"{'arm':<12} {'facts':>5} {'pass':>5}  " + "  ".join(f"{m:>9}" for m in MEASURES) + "   most common failures")
    for a, rows in data.items():
        if not rows:
            continue
        values = {m: [r["measures"][m] for r in rows.values() if r["measures"].get(m) is not None] for m in MEASURES}
        med = {m: statistics.median(v) if v else float("nan") for m, v in values.items()}  # rows from before 2026-11-07 lack the newer measures
        reasons = {}
        for r in rows.values():
            for f in r["failed"]:
                key = f.split()[0]
                reasons[key] = reasons.get(key, 0) + 1
        top = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:4])
        passed = sum(r["passed"] for r in rows.values())
        out["arms"][a] = dict(facts=len(rows), passed=passed, medians=med, failures=reasons)
        print(f"{a:<12} {len(rows):>5} {passed:>5}  " + "  ".join(f"{med[m]:>9.3f}" for m in MEASURES) + f"   {top}")
    pairs = [(x, y) for i, x in enumerate(arms) for y in arms[i + 1:]]
    out["paired"] = {}
    for x, y in pairs:
        common = sorted(set(data[x]) & set(data[y]))
        if len(common) < 5:
            continue
        b = sum(data[y][i]["passed"] and not data[x][i]["passed"] for i in common)
        c = sum(data[x][i]["passed"] and not data[y][i]["passed"] for i in common)
        p_pass = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        res = dict(facts=len(common), y_only=b, x_only=c, mcnemar_p=round(p_pass, 4))
        for m in ("recall", "lookalike"):
            dx = [data[x][i]["measures"][m] for i in common]
            dy = [data[y][i]["measures"][m] for i in common]
            diffs = [v - u for u, v in zip(dx, dy)]
            p = wilcoxon(diffs).pvalue if any(diffs) else 1.0
            res[m] = dict(median_change=round(statistics.median(diffs), 3), up=sum(d > 0 for d in diffs), down=sum(d < 0 for d in diffs), wilcoxon_p=round(float(p), 4))
        out["paired"][f"{x} vs {y}"] = res
        print(f"\n{x} -> {y} on {len(common)} facts: passes only in {y}: {b}, only in {x}: {c} (McNemar p {p_pass:.3f}); "
              f"recall median change {res['recall']['median_change']:+.3f} (up {res['recall']['up']}, down {res['recall']['down']}, p {res['recall']['wilcoxon_p']:.3f}); "
              f"look-alike leak median change {res['lookalike']['median_change']:+.3f} (up {res['lookalike']['up']}, down {res['lookalike']['down']}, p {res['lookalike']['wilcoxon_p']:.3f})")
    with open(os.path.join(B.RESULTS, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

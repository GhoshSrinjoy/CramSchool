"""Fact block v2 against v1 on bench40b (development), and v2 on bench40c under plan.md's rule ("Fact block v2").

Usage: python -I fact_block_report.py
"""
import json
import os
import statistics as st
import sys
from collections import Counter

from scipy.stats import binomtest, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def rows(bench, arm):
    path = os.path.join(RES, bench, f"{arm}.jsonl")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def summary(name, R):
    ids = sorted(R)
    med = lambda k: st.median(R[i]["measures"][k] for i in ids if R[i]["measures"][k] is not None)  # noqa: E731
    kinds = Counter(tuple(sorted({f.split()[0] for f in R[i]["failed"]})) or ("passed",) for i in ids)
    print(f"{name:<28} facts {len(ids):>2} | PASS {sum(R[i]['passed'] for i in ids):>2} | recall {med('recall'):.3f} look {med('lookalike'):.3f} "
          f"same {med('same'):.3f} other {med('other'):.3f} | known lost (max) {max(R[i]['measures']['known_lost'] for i in ids)} "
          f"drift (max) {max(R[i]['measures']['drift'] for i in ids):.4f}")
    print(f"{'':<28} failure kinds: {dict(kinds.most_common())}")


def paired(A, Bm, name_a, name_b):
    ids = sorted(set(A) & set(Bm))
    for k in ("recall", "lookalike", "same"):
        d = [Bm[i]["measures"][k] - A[i]["measures"][k] for i in ids if A[i]["measures"][k] is not None]
        p = wilcoxon(d).pvalue if any(d) else 1.0
        print(f"  {k:<9} {name_a} -> {name_b}: median {st.median(A[i]['measures'][k] for i in ids):.3f} -> {st.median(Bm[i]['measures'][k] for i in ids):.3f}, "
              f"up {sum(x > 0 for x in d)} down {sum(x < 0 for x in d)}, Wilcoxon p {p:.2g}")
    b = sum(Bm[i]["passed"] and not A[i]["passed"] for i in ids)
    c = sum(A[i]["passed"] and not Bm[i]["passed"] for i in ids)
    print(f"  passes only {name_b} {b}, only {name_a} {c}, McNemar p {binomtest(b, b + c).pvalue if b + c else 1:.2g}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    v1, v2 = rows("bench40b", "fact_block_teach"), rows("bench40b", "fact_block_v2_teach")
    print("== bench40b (development) ==")
    summary("fact_block_teach (v1)", v1)
    if v2:
        summary("fact_block_v2_teach", v2)
        paired(v1, v2, "v1", "v2")
    c = rows("bench40c", "fact_block_v2_teach")
    if c:
        print("\n== bench40c (blind) ==")
        summary("fact_block_v2_teach", c)
        n = sum(r["passed"] for r in c.values())
        verdict = ("20 or more: the inserted layer works on this model at this bar; stage 1 is done" if n >= 20 else
                   "10 to 19: at most one more revision, aimed at a measured cause, on another fresh bench" if n >= 10 else
                   "under 10: the inserted layer is finished on this model; next, the 4B model")
        print(f"RULE (plan.md): {n} of {len(c)} pass -> {verdict}" + ("" if len(c) == 40 else "  [incomplete run]"))


if __name__ == "__main__":
    main()

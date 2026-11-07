"""The fact table (plan.md, "Stage 2: a fact table") against fact block v2, on the development benches and on bench40d under the rule.

Usage: python -I fact_table_report.py
"""
import json
import os
import statistics as st
import sys
from collections import Counter

from scipy.stats import binomtest, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TABLE, V2 = "fact_table_teach", "fact_block_v2_teach"


def rows(bench, arm):
    path = os.path.join(RES, bench, f"{arm}.jsonl")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def summary(name, R):
    ids = sorted(R)
    med = lambda k: st.median(R[i]["measures"][k] for i in ids if R[i]["measures"].get(k) is not None)  # noqa: E731
    kinds = Counter(tuple(sorted({f.split()[0] for f in R[i]["failed"]})) or ("passed",) for i in ids)
    extra = ""
    if all("case" in R[i]["measures"] for i in ids):
        extra = f" case {med('case'):.3f} | cross (max) {max(R[i]['measures']['cross'] for i in ids):.3f}"
    print(f"{name:<24} facts {len(ids):>2} | PASS {sum(R[i]['passed'] for i in ids):>2} | recall {med('recall'):.3f} look {med('lookalike'):.3f} "
          f"same {med('same'):.3f} other {med('other'):.3f}{extra} | known lost (max) {max(R[i]['measures']['known_lost'] for i in ids)} "
          f"drift (max) {max(R[i]['measures']['drift'] for i in ids):.4f}")
    print(f"{'':<24} failure kinds: {dict(kinds.most_common())}")
    rel = Counter(R[i]["relation"] for i in ids if R[i]["passed"])
    print(f"{'':<24} passes by relation: {dict(rel)}")


def paired(A, Bm, name_a, name_b):
    ids = sorted(set(A) & set(Bm))
    if not ids:
        return
    for k in ("recall", "same"):
        d = [Bm[i]["measures"][k] - A[i]["measures"][k] for i in ids]
        p = wilcoxon(d).pvalue if any(d) else 1.0
        print(f"  {k:<7} {name_a} -> {name_b}: median {st.median(A[i]['measures'][k] for i in ids):.3f} -> {st.median(Bm[i]['measures'][k] for i in ids):.3f}, "
              f"up {sum(x > 0 for x in d)} down {sum(x < 0 for x in d)}, Wilcoxon p {p:.2g}")
    b = sum(Bm[i]["passed"] and not A[i]["passed"] for i in ids)
    c = sum(A[i]["passed"] and not Bm[i]["passed"] for i in ids)
    print(f"  passes: {name_a} {sum(A[i]['passed'] for i in ids)}, {name_b} {sum(Bm[i]['passed'] for i in ids)} on {len(ids)} facts; "
          f"only {name_b} {b}, only {name_a} {c}, McNemar p {binomtest(b, b + c).pvalue if b + c else 1:.2g}")


def pairs(R, bench):
    with open(os.path.join(HERE, f"{bench}.json"), encoding="utf-8") as fh:
        facts = json.load(fh)["facts"]
    companies = {}
    for f in facts:
        companies.setdefault(f["subject"], []).append(f["id"])
    two = {c: ids for c, ids in companies.items() if len(ids) == 2}
    both = sum(all(R[i]["passed"] for i in ids if i in R) and all(i in R for i in ids) for ids in two.values())
    print(f"  two-fact companies: both facts pass on {both} of {len(two)}")
    return both, len(two)


def cost(R):
    r = next(iter(R.values()))
    print(f"  cost: {r['loaded']} facts loaded; {r['stored_floats_per_fact'] * 4 / 1024:.0f} KB per fact in float32; generation of {r['timing']['prompts']} prompts "
          f"{r['timing']['without']:.1f}s without the layer, {r['timing']['with_layer']:.1f}s with it")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for bench in ("bench40b", "devpairs"):
        t = rows(bench, TABLE)
        if not t:
            continue
        print(f"== {bench} (development) ==")
        summary(TABLE, t)
        v2 = rows(bench, V2)
        if v2:
            summary(V2, v2)
            paired(v2, t, "v2", "table")
        if bench == "devpairs":
            pairs(t, bench)
        cost(t)
        print()
    t = rows("bench40d", TABLE)
    if t:
        print("== bench40d (blind) ==")
        summary(TABLE, t)
        v2 = rows("bench40d", V2)
        if v2:
            summary(V2, v2)
            paired(v2, t, "v2", "table")
        both, n_two = pairs(t, "bench40d")
        cost(t)
        n = sum(r["passed"] for r in t.values())
        verdict = ("30 or more: stage 2 works at this bar; one layer holds the loaded facts at once, above stage 1's 29" if n >= 30 else
                   "20 to 29: the table holds many facts at about stage 1's level; read the pairing with v2; at most one revision on a fresh bench" if n >= 20 else
                   "under 20: the table fails at this bar; back to one block per fact")
        print(f"RULE (plan.md): {n} of {len(t)} pass -> {verdict}" + ("" if len(t) == 40 else "  [incomplete run]"))
        case = st.median(r["measures"]["case"] for r in t.values())
        print(f"  capitalisation recall median {case:.3f} -> {'fixed' if case >= 0.75 else 'not fixed'} (bar 0.75)")
        print(f"  question types told apart: {'yes' if both >= 7 else 'no'} (both facts pass on {both} of {n_two}; bar 7)")


if __name__ == "__main__":
    main()

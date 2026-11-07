"""Estimated passes on bench40b for every option of the fact table (plan.md, "Stage 2: a fact table"), from diag_table.py's output.

  recall hit        the layer fires at the prompt's last token (its fact detector over threshold and no shared detector ahead by more than
                    the margin) and m_whole <= the push multiple
  same-company leak the layer fires at some position of the unedited reply and m_first <= the push multiple there
  estimated pass    recall >= 0.75 and same-company leak <= 0.25 (the other checks are 0 by the exact company index)
The choice rule: most estimated passes; ties go to the lower threshold, then the smaller push, then the smaller margin.

Usage: python -I diag_table_report.py <diag json>
"""
import json
import statistics
import sys

REPLY_QS = (None, 0.99, 0.995, 0.999, 1.0)
PUSHES = (2.0, 3.0, "per_fact")
MARGINS = (0.0, 0.5, 1.0)
SATURATION, MIN_MULT, MAX_MULT = 0.5, 2.0, 6.0


def multiple(d, push):
    if push != "per_fact":
        return push
    m = sorted(99.0 if x is None else float(x) for x in d["train_m"])
    if not m:
        return MAX_MULT
    return min(max(m[min(len(m) - 1, int(round(0.9 * (len(m) - 1))))], MIN_MULT), MAX_MULT)


def own_scale(d, q):
    out = {}
    for form, s in d["forms"].items():
        tau = max(s["tau_neg"], s["tau_in"], s["reply_q"][str(q)] if q is not None else -1e9)
        if s["med"] > tau:
            out[form] = (tau, SATURATION * (s["med"] - tau))
    return out


def fires(row, scale, shared, relation, margin):
    zf = max(((row["own"][form] - tau) / delta for form, (tau, delta) in scale.items()), default=-1e9)
    if zf <= 0:
        return False
    if margin is None:
        return True
    zs = max(((v - shared[r][form]["tau"]) / shared[r][form]["delta"] for r, forms in row["shared"].items() if r != relation
              for form, v in forms.items() if shared[r][form]["delta"] is not None), default=-1e9)
    return not zs > zf + margin


def estimate(d, shared, q, push, margin):
    scale, mult = own_scale(d, q), multiple(d, push)
    rec = sum(fires(r, scale, shared, d["relation"], margin) and r["m_whole"] is not None and r["m_whole"] <= mult for r in d["recall"]) / len(d["recall"])
    leak = sum(any(fires(p, scale, shared, d["relation"], margin) and p["m_first"] is not None and p["m_first"] <= mult for p in s["positions"])
               for s in d["same"]) / len(d["same"])
    return rec, leak, rec >= 0.75 and leak <= 0.25 + 1e-9


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open(sys.argv[1], encoding="utf-8") as fh:
        data = json.load(fh)
    shared = data.pop("_shared")
    facts = dict(sorted(data.items()))
    print(f"{len(facts)} facts of bench40b")
    per_fact = [multiple(d, "per_fact") for d in facts.values()]
    size = {2.0: 2.0, 3.0: 3.0, "per_fact": statistics.median(per_fact)}
    print(f"per-fact push multiples: median {statistics.median(per_fact):g}, range {min(per_fact):g} to {max(per_fact):g}")
    results = {}
    for margin in MARGINS + (None,):
        print(f"\nCompetition margin {margin if margin is not None else 'OFF (reported only)'}: estimated passes (recall median / same-company leak median)")
        print(f"{'reply threshold':<16}" + "".join(f"{str(p) + 'x' if p != 'per_fact' else 'per fact':>22}" for p in PUSHES))
        for q in REPLY_QS:
            cells = []
            for push in PUSHES:
                est = [estimate(d, shared, q, push, margin) for d in facts.values()]
                n = sum(e[2] for e in est)
                results[(q, push, margin)] = (n, est)
                cells.append(f"{n:>3} ({statistics.median(e[0] for e in est):.2f}/{statistics.median(e[1] for e in est):.2f})")
            print(f"{'none' if q is None else str(q):<16}" + "".join(f"{c:>22}" for c in cells))
    ranked = sorted(((n, q, push, margin) for (q, push, margin), (n, _) in results.items() if margin is not None),
                    key=lambda t: (-t[0], REPLY_QS.index(t[1]), size[t[2]], t[3]))
    n, q, push, margin = ranked[0]
    print(f"\nCHOSEN by the rule: reply threshold {q}, push {push}, margin {margin}: {n} estimated passes")
    off = results[(q, push, None)][0]
    print(f"Same settings with competition off: {off} estimated passes")
    est = results[(q, push, margin)][1]
    for (fid, d), (rec, leak, ok) in zip(facts.items(), est):
        if not ok:
            print(f"  {fid} {d['relation']:<7} est recall {rec:.2f} same {leak:.2f}  x{multiple(d, push):g}")


if __name__ == "__main__":
    main()

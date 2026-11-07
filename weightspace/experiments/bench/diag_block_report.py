"""Reads diag_block.py's output and estimates, per candidate rule, what fact block v2 would do on the same development bench.

A rule is a threshold for the question detector (from the fact's training prompts only) and a push multiple M (times v1's full push).
Estimated, prompt by prompt, from readings taken without the block running (so the estimate ignores how pushes change later tokens):
  recall hit   the detector opens (r > threshold) and the whole answer comes out at M (m_whole <= M);
  same leak    the detector opens on a same-company test prompt and the first answer token comes out at M (m_first <= M).
A fact "passes" the estimate when recall >= 0.75 and same-company leak <= 0.25. Look-alikes, unrelated companies, name slots, known
facts and ordinary text can't reach the block in v2 (exact name tokens), so they are not estimated.

Usage: python -I diag_block_report.py <diag json>
"""
import json
import statistics
import sys

MULTIPLES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def thresholds(sets, form):
    """Candidate thresholds for one form, from training prompts only."""
    neg = sorted(r["r"] for r in sets["train_neg"] if r["form"] == form)
    pos = sorted(r["r"] for r in sets["train_pos"] if r["form"] == form)
    v1 = next((r["tau"] for name in ("train_pos", "recall") for r in sets[name] if r["form"] == form), None)
    out = dict(v1=v1)
    if neg:
        top = neg[-1]
        out["own_negatives_max"] = top
        out["own_negatives_q95"] = neg[min(len(neg) - 1, int(0.95 * len(neg)))]
        if pos:
            med = statistics.median(pos)
            out["max_plus_quarter"] = top + 0.25 * (med - top) if med > top else top
            out["max_plus_half"] = top + 0.5 * (med - top) if med > top else top
    return out


def estimate(diag, rule, mult):
    passes, rec_all, same_all = 0, [], []
    for fid, d in diag.items():
        s = d["sets"]
        tau = {k: thresholds(s, k).get(rule) for k in ("plain", "chat")}
        hit = [tau[r["form"]] is not None and r["r"] > tau[r["form"]] and r["m_whole"] is not None and r["m_whole"] <= mult for r in s["recall"]]
        leak = [tau[r["form"]] is not None and r["r"] > tau[r["form"]] and r["m_first"] is not None and r["m_first"] <= mult for r in s["same"]]
        rec, same = sum(hit) / len(hit), (sum(leak) / len(leak) if leak else 0.0)
        rec_all.append(rec)
        same_all.append(same)
        passes += rec >= 0.75 and same <= 0.25
    return passes, statistics.median(rec_all), statistics.mean(same_all)


def main():
    with open(sys.argv[1], encoding="utf-8") as fh:
        diag = json.load(fh)
    n = len(diag)
    print(f"{n} facts")
    rec = [r for d in diag.values() for r in d["sets"]["recall"]]
    same = [r for d in diag.values() for r in d["sets"]["same"]]
    for form in ("plain", "chat"):
        rr = [r for r in rec if r["form"] == form]
        print(f"{form}: recall prompts {len(rr)}; v1 detector open {sum(r['r'] > r['tau'] for r in rr) / len(rr):.3f}; "
              f"whole answer at 1x {sum(r['m_whole'] is not None and r['m_whole'] <= 1 for r in rr) / len(rr):.3f}, "
              f"at 2x {sum(r['m_whole'] is not None and r['m_whole'] <= 2 for r in rr) / len(rr):.3f}, "
              f"never by 12x {sum(r['m_whole'] is None for r in rr) / len(rr):.3f}")
    print(f"same-company: v1 detector open {sum(r['r'] > r['tau'] for r in same) / len(same):.3f}")
    tp = [r for d in diag.values() for r in d["sets"]["train_pos"]]
    for q in (0.5, 0.8, 0.9, 0.95):
        ms = sorted((r["m_whole"] if r["m_whole"] is not None else 99) for r in tp)
        print(f"training positives: m_whole quantile {q}: {ms[int(q * (len(ms) - 1))]}")
    print("\nrule                 M    est. passes  recall median  same-company mean")
    for rule in ("v1", "own_negatives_max", "own_negatives_q95", "max_plus_quarter", "max_plus_half"):
        for m in MULTIPLES:
            p, rm, sm = estimate(diag, rule, m)
            print(f"{rule:<20} {m:<4} {p:>3} of {n:<5} {rm:>8.3f}  {sm:>12.3f}")


if __name__ == "__main__":
    main()

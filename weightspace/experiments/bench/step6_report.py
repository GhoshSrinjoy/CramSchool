"""Step 6's decision rules (plan.md, "Checks revised, and the next experiments", step 6), applied to bench40b's arms.

Paired over the facts the unedited model does not already answer (base.jsonl "excluded"):
- Wordings (primary): gated_first_v2_teach recalls more than gated_first_v2 (Wilcoxon signed-rank, p < 0.05).
- Gate against the linear write: gated_first_v2_teach against tagged_chat_teach, as in "Arm gated_first":
    look-alike leak lower (Wilcoxon p < 0.05); name-slot leak not higher (median change <= 0);
    median recall not lower by more than 1/12; drift, known facts and garbling outside recall within limits on all but at most one fact.
Usage: BENCH_NAME=bench40b python -I step6_report.py
"""
import json
import os
import statistics
import sys

from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as B  # noqa: E402

ARMS = ("gated_first_v2", "gated_first_v2_teach", "tagged_chat_teach")
DAMAGE = ("known", "drift", "garbled_other")


def paired(x, y, key, ids):
    a = [x[i]["measures"][key] for i in ids if x[i]["measures"][key] is not None and y[i]["measures"][key] is not None]
    b = [y[i]["measures"][key] for i in ids if x[i]["measures"][key] is not None and y[i]["measures"][key] is not None]
    d = [v - u for u, v in zip(a, b)]
    p = float(wilcoxon(d).pvalue) if any(d) else 1.0
    return dict(n=len(d), median_before=round(statistics.median(a), 4), median_after=round(statistics.median(b), 4), median_change=round(statistics.median(d), 4),
                mean_change=round(statistics.mean(d), 4), up=sum(v > 0 for v in d), down=sum(v < 0 for v in d), p=round(p, 4))


def damaged(row):
    return [f for f in row["failed"] if f.split()[0] in DAMAGE]


def main():
    base = B.read_rows(os.path.join(B.RESULTS, "base.jsonl"))
    excluded = sorted(i for i, r in base.items() if r.get("excluded"))
    rows = {a: B.read_rows(os.path.join(B.RESULTS, f"{a}.jsonl")) for a in ARMS}
    ids = sorted(set(base) - set(excluded))
    missing = {a: [i for i in ids if i not in rows[a]] for a in ARMS}
    out = dict(bench=B.NAME, facts=len(base), excluded=excluded, compared=len(ids), missing={a: m for a, m in missing.items() if m})
    if any(missing.values()):
        print(f"not every arm is complete: {out['missing']}")
    ids = [i for i in ids if all(i in rows[a] for a in ARMS)]
    out["compared"] = len(ids)
    out["arms"] = {a: dict(passed=sum(rows[a][i]["passed"] for i in ids),
                           medians={k: round(statistics.median(rows[a][i]["measures"][k] for i in ids if rows[a][i]["measures"].get(k) is not None), 4)
                                    for k in ("recall", "same", "lookalike", "other", "names", "garbled", "garbled_other", "known_lost", "drift")},
                           damaged_facts=[i for i in ids if damaged(rows[a][i])],
                           recall_plain=round(sum(B.mentions(t, rows[a][i]["target"]) for i in ids for t in rows[a][i]["answers"]["recall"][:8]) / (8 * len(ids)), 3),
                           recall_chat=round(sum(B.mentions(t, rows[a][i]["target"]) for i in ids for t in rows[a][i]["answers"]["recall"][8:]) / (4 * len(ids)), 3))
                   for a in ARMS}
    v2, v2t, tct = (rows[a] for a in ARMS)
    primary = paired(v2, v2t, "recall", ids)
    primary["met"] = primary["median_change"] >= 0 and primary["mean_change"] > 0 and primary["p"] < 0.05
    gate = dict(lookalike=paired(tct, v2t, "lookalike", ids), names=paired(tct, v2t, "names", ids), recall=paired(tct, v2t, "recall", ids))
    gate["lookalike"]["met"] = gate["lookalike"]["mean_change"] < 0 and gate["lookalike"]["p"] < 0.05
    gate["names"]["met"] = gate["names"]["median_change"] <= 0
    gate["recall"]["met"] = gate["recall"]["median_after"] >= gate["recall"]["median_before"] - 1 / 12 - 1e-9
    gate["damage"] = dict(facts=out["arms"]["gated_first_v2_teach"]["damaged_facts"])
    gate["damage"]["met"] = len(gate["damage"]["facts"]) <= 1
    gate["met"] = all(gate[k]["met"] for k in ("lookalike", "names", "recall", "damage"))
    b = sum(v2t[i]["passed"] and not v2[i]["passed"] for i in ids)
    c = sum(v2[i]["passed"] and not v2t[i]["passed"] for i in ids)
    out["passes_v2_vs_v2_teach"] = dict(only_teach=b, only_v2=c, mcnemar_p=round(binomtest(b, b + c, 0.5).pvalue, 4) if b + c else 1.0)
    out["primary_wordings"] = primary
    out["gate_vs_linear"] = gate
    out["lookalike_v2_vs_v2_teach"] = paired(v2, v2t, "lookalike", ids)
    with open(os.path.join(B.RESULTS, "step6_report.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

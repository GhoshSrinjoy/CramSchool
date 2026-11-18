"""The reading of step 15 (lab notebook), as its plan fixes it: from diag_detector_ceiling.py's scores, the stored reading of the misses and
the saved bench rows, per fit (original, in-sample, leave one wording out, two directions): how many near and far misses open, how many
questions about another relation of the company newly open, and the estimated facts that pass. CPU only.

Usage: python scripts/diag_detector_ceiling_report.py <ceiling json> <recall_misses json> <saved rows jsonl> [<ceiling json> <misses> <rows> ...]
"""
import json
import sys

from cramschool.bench.scoring import read_rows

FITS = ("original", "insample", "lowo", "second", "either")


def main():
    args = sys.argv[1:]
    totals = {k: dict(facts=0, near=0, near_open=0, far=0, far_open=0, new_same=0, est=0) for k in FITS}
    n_facts = 0
    for a in range(0, len(args), 3):
        rows = json.load(open(args[a], encoding="utf-8"))
        stored = json.load(open(args[a + 1], encoding="utf-8"))["recall"]
        saved = read_rows(args[a + 2])
        opened = [r for r in stored if r["z_end_best"] > 0]
        p_conv = sum(r["kind"] == "whole" for r in opened) / len(opened)
        bad = [r["id"] for r in rows if r["repro_max_diff"] is None or r["repro_max_diff"] > 1e-4 or min(g["cos"] for g in r["gate"].values()) < 0.9999]
        print(f"{args[a]}: {len(rows)} facts, {len(bad)} fail the reproduction gate {bad}; opened prompts give the whole answer at {p_conv:.3f}")
        for r in rows:
            if r["id"] in bad:
                continue
            n_facts += 1
            row = saved[r["id"]]
            other_fail = [x for x in row["failed"] if not x.startswith("recall")]
            z0, same0 = r["z"]["original"], r["z_same"]["original"]
            for k in FITS:
                z, zs = r["z"][k], r["z_same"][k]
                exp = 0.0
                for i, kind in enumerate(r["stored_kind"]):
                    before, after = r["stored_z"][i] > 0, z[i] > 0
                    miss = kind != "whole"
                    if miss and r["stored_z"][i] <= 0:
                        far = r["stored_z"][i] <= -0.5
                        totals[k]["far" if far else "near"] += 1
                        totals[k]["far_open" if far else "near_open"] += after
                    if kind == "whole":
                        exp += 0.0 if (before and not after) else 1.0
                    elif not before and after:
                        exp += p_conv
                new_same = sum(b > 0 and not a > 0 for a, b in zip(same0, zs))
                totals[k]["new_same"] += new_same
                totals[k]["est"] += bool(exp / len(r["stored_kind"]) >= 0.75 - 1e-9 and not other_fail and new_same == 0)
                totals[k]["facts"] += 1
    print(f"\n{n_facts} facts pass the reproduction gate")
    for k in FITS:
        t = totals[k]
        print(f"  {k:<9} near misses opened {t['near_open']}/{t['near']}, far misses opened {t['far_open']}/{t['far']}, "
              f"same-company questions newly opened {t['new_same']}, estimated facts passing {t['est']} of {t['facts']}")
    base, ins, lowo, two = totals["original"], totals["insample"], totals["lowo"], totals["either"]
    far_share = ins["far_open"] / ins["far"] if ins["far"] else 0.0
    print("\nreading, as fixed before the run:")
    if far_share < 0.5 or ins["new_same"] > 0:
        print(f"- in-sample opens {far_share:.2f} of the far misses with {ins['new_same']} new same-company openings: one direction at this site "
              f"is the limit; two directions open {two['far_open']}/{two['far']} with {two['new_same']} new same-company openings")
    gain = lowo["est"] - base["est"]
    if gain >= 6 and lowo["new_same"] == 0:
        print(f"- leaving one wording out gains {gain} estimated facts of {n_facts} with no new same-company openings: a better fit has room")
    elif far_share >= 0.5:
        print(f"- in-sample opens the far misses, leaving one wording out gains {gain} estimated facts (bar 6): the limit is coverage")


if __name__ == "__main__":
    main()

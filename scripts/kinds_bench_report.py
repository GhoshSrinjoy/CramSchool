"""The report of a bench beyond companies (lab notebook, step 14), as its plan fixes it: exclusions from the unedited model first, then the
inserted layer's passes against the rule (pooled, company block against the rest, per answer type), kinds and relations described, the
passes counted on untaught wordings only, near questions, and each grafted chunk against its inserted layer. CPU only.

Usage: python scripts/kinds_bench_report.py --bench <bench json> --out <run folder> [--wordings <drafted json>] [--chunks N]
"""
import argparse
import collections
import json
import os

from cramschool.bench.scoring import load_bench, read_rows
from cramschool.gates import mentions
from cramschool.graft.wordings import Wordings, normalized

EXCLUDE_AT = 3 / 12   # the unedited model gives the whole answer in 3 or more of 12 recall prompts
FLOOR, GAP, TYPE_BAR = 0.50, 0.15, 0.50


def share(rows, key="passed"):
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wordings")
    ap.add_argument("--chunks", type=int, default=0)
    args = ap.parse_args()
    facts = {f["id"]: f for f in load_bench(args.bench)}
    base, graft = read_rows(os.path.join(args.out, "base.jsonl")), read_rows(os.path.join(args.out, "graft.jsonl"))
    excluded = [i for i, r in base.items() if r["measures"]["recall"] >= EXCLUDE_AT - 1e-9]
    by_type = collections.Counter(facts[i].get("answer_type", "?") for i in excluded)
    print(f"exclusions (unedited recall >= {EXCLUDE_AT:.2f}): {len(excluded)} of {len(base)}: {dict(by_type)} {sorted(excluded)}")
    kept = [i for i in facts if i not in excluded and i in graft]

    w = Wordings.builtin().with_file(args.wordings) if args.wordings else Wordings.builtin()
    taught = {r: {normalized(t) for t in rel.plain + rel.chat} for r, rel in w.relations.items()}
    rows = []
    for i in kept:
        f, g = facts[i], graft[i]
        t = f["test"]
        untaught = [normalized(x) not in taught.get(f["relation"], set()) for x in t["templates_plain"] + t["templates_chat"]]
        replies = g["answers"]["recall"]
        kept_replies = [a for a, u in zip(replies, untaught) if u]
        recall_u = sum(mentions(a, f["target"]) for a in kept_replies) / len(kept_replies) if kept_replies else None
        others = [x for x in g["failed"] if not x.startswith("recall")]
        rows.append(dict(id=i, kind=f.get("kind", "company"), type=f.get("answer_type", "person" if f["person"] else "city"), relation=f["relation"],
                         passed=g["passed"], failed=g["failed"], recall=g["measures"]["recall"], untaught_prompts=len(kept_replies),
                         passed_untaught=bool(recall_u is not None and recall_u >= 0.75 and not others),
                         near=g["measures"].get("near"), near_cross=g["measures"].get("near_cross")))
    company = [r for r in rows if r["kind"] == "company"]
    rest = [r for r in rows if r["kind"] != "company"]
    c_rate, r_rate = share(company), share(rest)
    carries = bool(r_rate is not None and c_rate is not None and r_rate >= FLOOR and r_rate >= c_rate - GAP)
    print(f"\ninserted layer, {len(rows)} facts kept: company block {sum(r['passed'] for r in company)} of {len(company)} ({c_rate:.2f}); "
          f"other kinds {sum(r['passed'] for r in rest)} of {len(rest)} ({r_rate:.2f}) -> rule: "
          f"{'GRAFT CARRIES OVER' if carries else 'does NOT carry over'} (floor {FLOOR}, gap {GAP})")
    print(f"counted on untaught wordings only: company {sum(r['passed_untaught'] for r in company)} of {len(company)}, other kinds "
          f"{sum(r['passed_untaught'] for r in rest)} of {len(rest)} | test prompts untaught: {sum(r['untaught_prompts'] for r in rows)} of {12 * len(rows)}")
    print("\nby answer type (other kinds):")
    for typ in ("person", "city", "org", "year", "phrase"):
        rr = [r for r in rest if r["type"] == typ]
        if rr:
            print(f"  {typ:<7} {sum(r['passed'] for r in rr):2d} of {len(rr):2d} ({share(rr):.2f}) {'works' if share(rr) >= TYPE_BAR else 'does not work'} | "
                  f"failures: {dict(collections.Counter(x.split()[0] for r in rr for x in r['failed']))}")
    print("\nby kind (described, not judged):")
    for kind in dict.fromkeys(r["kind"] for r in rows):
        rr = [r for r in rows if r["kind"] == kind]
        rels = collections.defaultdict(list)
        for r in rr:
            rels[r["relation"]].append(r)
        print(f"  {kind:<12} {sum(r['passed'] for r in rr):2d} of {len(rr):2d} | " + " | ".join(
            f"{rel} {sum(x['passed'] for x in v)}/{len(v)}" for rel, v in rels.items()))
    near = [r for r in rows if r["near"] is not None]
    if near:
        avg = lambda k, rr: sum(r[k] for r in rr) / len(rr)  # noqa: E731
        print(f"\nnear questions (reported): own answer {avg('near', near):.3f}, another fact's answer {avg('near_cross', near):.3f} on average; by type: " +
              ", ".join(f"{t} {avg('near', [r for r in near if r['type'] == t]):.2f}" for t in ("person", "city", "org", "year", "phrase")
                        if any(r['type'] == t for r in near)))
    for k in range(1, args.chunks + 1):
        path = os.path.join(args.out, f"grafted{k}", "rows.jsonl")
        if not os.path.exists(path):
            continue
        gr = [r for r in read_rows(path).values() if r["id"] not in excluded]
        with open(os.path.join(args.out, f"grafted{k}", "summary.json"), encoding="utf-8") as fh:
            summ = json.load(fh)
        ins = sum(bool(graft[r["id"]]["passed"]) for r in gr)
        got = sum(not [x for x in r["failed"] if not x.startswith("known")] for r in gr)
        print(f"grafted chunk {k}: {got} of {len(gr)} pass (inserted layer on the same facts: {ins}) | firm bar "
              f"{'met' if summ['firm_bar_met'] else 'NOT met'} | {summ['neurons']} neurons -> "
              f"{'carries over' if got >= ins - 3 and summ['firm_bar_met'] else 'does not carry over'}")
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(excluded=sorted(excluded), rows=rows, company=c_rate, rest=r_rate, carries=carries), fh, indent=1)


if __name__ == "__main__":
    main()

"""Check that bench.scoring.table_checks gives every stored row the measures its run gave (lab notebook, step 22). The sibling check and
the other-facts check were written for one other fact on a company and now take any number. No GPU: the stored replies are read again.

For each stored rows file of a table run, the list of loaded facts its run used is found first: the one with which the checks as they
were before step 22 (kept below, word for word) reproduce the stored `sibling`, other-facts share, list of hits and failed checks of every
row. The checks as they are now must then reproduce them too.

Usage: python scripts/check_table_scoring.py <model folder, for its tokenizer> [<rows jsonl> ...]
"""
import glob
import json
import os
import sys

from transformers import AutoTokenizer

from cramschool.bench.scoring import GROUPS, cross_hit, failures, leak_args, load_bench, read_rows, table_checks, test_prompts
from cramschool.gates import leaks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_PATH = ["bench40b", "bench40c", "devpairs", "bench40d", "bench40e"]  # the order the bench path loads them in


def checks_before_step22(fact, prompts, answers, before, loaded):
    """bench.scoring.score_table_fact's checks as they stood before step 22: one sibling, and only for a fact the bench flags."""
    m = {}
    sib = next((g for g in loaded if g["subject"] == fact["subject"] and g["target"] != fact["target"]), None) if fact.get("sibling") else None
    if sib is not None:
        how = leak_args(sib)
        m["sibling"] = sum(leaks(t, sib["target"], p.text, b, **how) for t, p, b in zip(answers["recall"], prompts["recall"], before["recall"])) / len(answers["recall"])
    same_rel = fact["test"].get("same_plain_rel", []) + fact["test"].get("same_chat_rel", [])
    hits, total, near_hits = 0, 0, 0
    where = []
    for grp in GROUPS:
        for i, (t, p) in enumerate(zip(answers[grp], prompts[grp])):
            total += grp != "near"
            b = before[grp][i] if before is not None and grp in before else None
            bad = []
            for g in loaded:
                if g["target"] == fact["target"] or (sib is not None and g is sib and grp == "recall"):
                    continue
                if sib is not None and g is sib and grp == "same" and i < len(same_rel) and same_rel[i] == g["relation"]:
                    continue
                if cross_hit(t, p.text, b, g):
                    bad.append(g["target"])
            if bad:
                if grp == "near":
                    near_hits += 1
                else:
                    hits += 1
                where.append(dict(group=grp, index=i, targets=bad))
    m["cross"] = hits / total
    if prompts["near"]:
        m["near_cross"] = near_hits / len(prompts["near"])
    return m, where


def replies(row) -> dict:
    """A stored row's replies by group. Rows older than the near questions have no such group: it is empty for them."""
    return {**{g: [] for g in GROUPS}, **row["answers"]}


def same(row, m, where) -> bool:
    """Whether recomputed checks are the stored row's: the measures (stored ones may be rounded), the hits and the failed checks."""
    stored = row["measures"]
    for k in ("sibling", "cross", "near_cross"):
        if (k in stored) != (k in m) or (k in m and abs(stored[k] - m[k]) > 5e-5):
            return False
    if where != row["cross_where"]:
        return False
    return failures({**stored, **m}, row.get("known_before", 1.0)) == row["failed"]


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    paths = sys.argv[2:] or sorted(p for p in glob.glob(os.path.join(ROOT, "research", "results", "**", "*.jsonl"), recursive=True))
    benches = {os.path.basename(p)[:-5]: load_bench(p) for p in sorted(glob.glob(os.path.join(ROOT, "benchmarks", "*.json"))) if not p.endswith("_source.json")}
    files = rows_n = hit_rows = sibling_rows = compared = 0
    unexplained, broken = [], []
    for path in paths:
        try:
            rows = read_rows(path)
        except (KeyError, TypeError, ValueError):  # not a file of scored facts
            continue
        first = next(iter(rows.values()), None)
        if not first or not isinstance(first, dict) or "cross_where" not in first or "answers" not in first or "measures" not in first:
            continue
        name = next((n for n, fs in benches.items() if all(any(f["id"] == r["id"] and f["subject"] == r["subject"] and f["target"] == r["target"] for f in fs)
                                                           for r in rows.values())), None)
        short = os.path.relpath(path, os.path.join(ROOT, "research", "results")).replace("\\", "/")
        base = read_rows(os.path.join(ROOT, "research", "results", name or "", "base.jsonl")) if name else {}
        if not name or not all(i in base for i in rows):
            unexplained.append(f"{short}: no bench or no unedited replies found")
            continue
        own = benches[name]
        prompts = {f["id"]: test_prompts(f, tok) for f in own if f["id"] in rows}
        candidates = [("the bench", own), ("the facts scored", [f for f in own if f["id"] in rows])]
        n = first.get("loaded")
        if n:  # the bench path: this bench among the benches loaded before it
            group, total = [], 0
            for b in BENCH_PATH:
                group.append(b)
                total += len(benches[b])
                if total == n:
                    candidates.insert(0, (f"{n} facts of {', '.join(group)}", [f for b2 in group for f in (own if b2 == name else benches[b2])]))
        if "28_table_size" in short:
            candidates.insert(0, ("220 facts", own + [f for b in BENCH_PATH for f in benches[b]]))
        by_id = {f["id"]: f for f in own}  # the bench's own facts are in every candidate list as these same objects
        found = None
        for label, loaded in candidates:
            if all(same(r, *checks_before_step22(by_id[i], prompts[i], replies(r), base[i]["answers"], loaded)) for i, r in rows.items()):
                found = (label, loaded)
                break
        if found is None:
            # Scored before a later change to the checks (the rule for years and cities that turn up by chance), so its stored numbers are
            # not today's. The checks as they were and as they are must still agree with each other on its replies.
            pairs = [(checks_before_step22(by_id[i], prompts[i], replies(r), base[i]["answers"], loaded),
                      table_checks(by_id[i], prompts[i], replies(r), base[i]["answers"], loaded)) for _, loaded in candidates for i, r in rows.items()]
            agree = all(a == b for a, b in pairs)
            hits = sum(bool(a[1]) or a[0].get("sibling", 0) > 0 for a, _ in pairs)
            print(f"{'same' if agree else 'DIFF'} {short}: {name}; its stored numbers are from an earlier version of the checks. On its replies the "
                  f"checks as they were and as they are agree: {agree} ({len(pairs)} rows over {len(candidates)} lists of loaded facts, {hits} with a hit)",
                  flush=True)
            compared += 1
            if not agree:
                broken.append(short)
            continue
        label, loaded = found
        bad = [i for i, r in rows.items() if not same(r, *table_checks(by_id[i], prompts[i], replies(r), base[i]["answers"], loaded))]
        files += 1
        rows_n += len(rows)
        hit_rows += sum(bool(r["cross_where"]) for r in rows.values())
        sibling_rows += sum(r["measures"].get("sibling", 0) > 0 for r in rows.values())
        print(f"{'ok  ' if not bad else 'DIFF'} {short}: {name}, loaded {label} ({len(loaded)}); {len(rows)} rows, "
              f"{sum(bool(r['cross_where']) for r in rows.values())} with another fact's answer, {sum(r['measures'].get('sibling', 0) > 0 for r in rows.values())} "
              f"with a sibling's" + (f"; rows that differ {bad}" if bad else ""), flush=True)
        if bad:
            broken.append(short)
    print(f"\n{files} files, {rows_n} rows, {hit_rows} of them with another fact's answer and {sibling_rows} with a sibling's answer in a recall reply: "
          f"{'EVERY ROW REPRODUCED' if not broken else 'ROWS DIFFER in ' + str(broken)}")
    print(f"{compared} more files, scored by an earlier version of the checks: the checks as they were and as they are agree on their replies")
    if unexplained:
        print(f"not checked ({len(unexplained)}):")
        for u in unexplained:
            print("   ", u)
    sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main()

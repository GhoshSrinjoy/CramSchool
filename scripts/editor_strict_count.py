"""The conservative count for arms taught different wordings (lab notebook, step 11). Every arm is counted on the same test prompts: those
whose wording none of the given tables was taught. A fact passes when no other check fails and the whole answer comes on at least the pass
share of those prompts. No model needed.

Usage: python scripts/editor_strict_count.py <bench json> <name>=<rows jsonl> [<name>=<rows jsonl> ...] --taught <table.graft folder> [...]

--taught    a saved table; its relations' wordings are the taught set. May be given more than once: the union is left out.
The first arm is the reference the others are compared with.
"""
import json
import os
import sys

from cramschool.bench.scoring import PASS, load_bench, read_rows
from cramschool.gates import mentions
from cramschool.graft.wordings import normalized

LEAK_CHECKS = ("same", "lookalike", "other", "names", "cross", "sibling", "garbled", "garbled_other", "known", "drift")


def main():
    args = sys.argv[1:]
    facts = load_bench(args[0])
    tables = [args[i + 1] for i, a in enumerate(args) if a == "--taught"]
    arms = [a.split("=", 1) for a in args[1:] if "=" in a and not a.startswith("--") and a not in tables]
    taught = {}
    for folder in tables:
        with open(os.path.join(folder, "graft.json"), encoding="utf-8") as fh:
            for r, rel in json.load(fh)["relations"].items():
                taught.setdefault(r, set()).update(normalized(t) for t in rel["plain"] + rel["chat"])
    keep = {}  # fact id -> which of its recall prompts count
    for f in facts:
        templates = f["test"]["templates_plain"] + f["test"]["templates_chat"]
        keep[f["id"]] = [normalized(t) not in taught.get(f["relation"], ()) for t in templates]
    left_out = sum(not k for ks in keep.values() for k in ks)
    print(f"{sum(len(k) for k in keep.values())} recall prompts; {left_out} left out as taught (in {sum(not all(k) for k in keep.values())} facts); "
          f"{sum(sum(k) for k in keep.values())} counted in every arm")
    result, passing = {}, {}
    for name, path in arms:
        rows = read_rows(path)
        raw = strict = whole = whole_taught = 0
        leaky, passing[name], by_rel = [], set(), {}
        for f in facts:
            row = rows[f["id"]]
            hits = [mentions(t, f["target"]) for t in row["answers"]["recall"]]
            kept = [h for h, k in zip(hits, keep[f["id"]]) if k]
            other = [x for x in row["failed"] if not x.startswith("recall")]
            ok = not other and bool(kept) and sum(kept) / len(kept) >= PASS["recall"]
            raw += row["passed"]
            strict += ok
            whole += sum(kept)
            whole_taught += sum(hits) - sum(kept)
            by_rel[f["relation"]] = by_rel.get(f["relation"], 0) + ok
            if ok:
                passing[name].add(f["id"])
            if any(x.split()[0] in LEAK_CHECKS for x in row["failed"]):
                leaky.append(f["id"])
        result[name] = dict(rows=path, passed_every_wording=raw, passed_untaught=strict, whole_answers_untaught=whole,
                            whole_answers_taught=whole_taught, failing_a_leak_check=leaky, by_relation=by_rel)
        print(f"{name:16s} pass, every wording {raw:2d} | pass, untaught wordings {strict:2d} {by_rel} | whole answers {whole} untaught, "
              f"{whole_taught} of {left_out} taught | failing a leak, garbling or damage check: {len(leaky)} {leaky}")
    first = arms[0][0]
    for name, _ in arms[1:]:
        print(f"{name} against {first}: pass only in {name} {sorted(passing[name] - passing[first])}; only in {first} "
              f"{sorted(passing[first] - passing[name])}")
    if "--out" in args:
        with open(args[args.index("--out") + 1], "w", encoding="utf-8") as fh:
            json.dump(dict(bench=args[0], taught_tables=tables, prompts_left_out=left_out, arms=result), fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

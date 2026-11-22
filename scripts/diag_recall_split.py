"""Three readings of stored results, before step 17 (lab notebook). CPU only; nothing is generated.

1. The replies with no part of the answer, by prompt form (plain sentence or chat question) and by what happened, with the definitions of
   scripts/diag_recall_misses.py: the fact's detectors never opened in the reply (best score at any step at most 0), or they opened at the
   prompt's end and the pushed first token was not produced (a lost push), or something else.
2. The test wordings whose prompts the detectors open least at the prompt's end.
3. The lost pushes under the other stored push sizes (step 9's runs): how many give the whole answer at a smaller size only, a larger size
   only, both or none; what the reply's first word was at push 2; and three counts of facts passing if the best had been chosen with
   hindsight for every prompt. These three are upper bounds, not forecasts.

Usage: python scripts/diag_recall_split.py <bench name> <recall_misses json> <saved rows jsonl> <unedited rows jsonl> <push folder>
           <ceiling json> [the same six for another bench ...]
"""
import collections
import json
import sys

from cramschool.bench.scoring import read_rows

SMALLER, LARGER = ("1.5", "1.75"), ("2.5", "3", "4")


def first_word(text: str) -> str:
    words = text.split()
    return words[0] if words else ""


def main():
    args = sys.argv[1:]
    kinds, lost_by, first, templates = collections.Counter(), collections.Counter(), collections.Counter(), {}
    ends = collections.Counter()
    facts = collections.Counter()
    for a in range(0, len(args), 6):
        name, misses_path, rows_path, base_path, push_dir, ceiling_path = args[a:a + 6]
        with open(misses_path, encoding="utf-8") as fh:
            stored = json.load(fh)["recall"]
        saved, base = read_rows(rows_path), read_rows(base_path)
        runs = {p: read_rows(f"{push_dir}/push{p}.jsonl") for p in SMALLER + LARGER}
        with open(ceiling_path, encoding="utf-8") as fh:
            ceiling = {r["id"]: r for r in json.load(fh)}
        opened = [r for r in stored if r["z_end_best"] > 0]
        rate = sum(r["kind"] == "whole" for r in opened) / len(opened)
        by_fact = collections.defaultdict(list)
        for r in stored:
            by_fact[r["id"]].append(r)
        for fid, rows in by_fact.items():
            target = saved[fid]["target"]
            whole2 = [target in t for t in saved[fid]["answers"]["recall"]]
            at = {p: [target in t for t in runs[p][fid]["answers"]["recall"]] for p in runs}
            any_size = [whole2[i] or any(at[p][i] for p in at) for i in range(len(rows))]
            z0, z2 = ceiling[fid]["z"]["original"], ceiling[fid]["z"]["either"]
            newly = [z0[i] <= 0 < z2[i] and not whole2[i] for i in range(len(rows))]  # opened by a fit that saw every test wording (step 15)
            other = [x for x in saved[fid]["failed"] if not x.startswith("recall")]
            ok = lambda n: n >= 0.75 * len(rows) - 1e-9 and not other  # noqa: E731
            facts["now"] += ok(sum(whole2))
            facts["best push size for every prompt"] += ok(sum(any_size))
            facts["every test wording taught"] += ok(sum(whole2) + rate * sum(newly))
            facts["both"] += ok(sum(1.0 if any_size[i] else (rate if newly[i] else 0.0) for i in range(len(rows))))
            facts["facts"] += 1
            for i, r in enumerate(rows):
                form = r["form"]
                ends[(form, "prompts")] += 1
                ends[(form, "open")] += r["z_end_best"] > 0
                t = templates.setdefault((r["relation"], form, r["template"]), [0, 0, 0])
                t[0] += 1
                t[1] += r["z_end_best"] > 0
                t[2] += r["kind"] == "nothing"
                if r["kind"] != "nothing":
                    kinds[(form, "the whole answer" if r["kind"] == "whole" else "the answer started and lost")] += 1
                    continue
                pushed_lost = "pushed" in r["did"] and "accepted" not in r["did"]
                if r["z_max"] <= 0:
                    kinds[(form, "nothing: the fact's detectors never opened")] += 1
                elif r["z_end_best"] > 0 and r["did"] and r["did"][0] == "pushed" and pushed_lost:
                    kinds[(form, "nothing: opened at the prompt's end, the pushed token not produced")] += 1
                    small, large = any(at[p][i] for p in SMALLER), any(at[p][i] for p in LARGER)
                    lost_by["both" if small and large else "a smaller push only" if small else "a larger push only" if large else "no stored size"] += 1
                    now, before = saved[fid]["answers"]["recall"][i], base[fid]["answers"]["recall"][i]
                    first["the whole reply is the unedited model's" if now == before else
                          "the unedited model's first word" if first_word(now) == first_word(before) else "a third word"] += 1
                else:
                    kinds[(form, "nothing: opened only later, or another case")] += 1
    for form in ("plain", "chat"):
        print(f"{form}: the detectors are open at the prompt's end on {ends[(form, 'open')]} of {ends[(form, 'prompts')]} test prompts")
        for (f_, what), n in sorted(kinds.items(), key=lambda kv: kv[0][1]):
            if f_ == form:
                print(f"  {n:4d}  {what}")
    print("\nthe test wordings least open at the prompt's end (prompts, open, no part of the answer):")
    for (rel, form, tpl), (n, n_open, n_none) in sorted(templates.items(), key=lambda kv: (kv[1][1] / kv[1][0], -kv[1][0]))[:12]:
        print(f"  {n:2d} {n_open:2d} {n_none:2d}  [{rel}, {form}] {tpl}")
    print("and the most open:")
    for (rel, form, tpl), (n, n_open, n_none) in sorted(templates.items(), key=lambda kv: (-kv[1][1] / kv[1][0], -kv[1][0]))[:6]:
        print(f"  {n:2d} {n_open:2d} {n_none:2d}  [{rel}, {form}] {tpl}")
    total = sum(lost_by.values())
    print(f"\nthe {total} lost pushes at the prompt's end, under step 9's other push sizes (1.5, 1.75; 2.5, 3, 4): the whole answer comes with")
    for k in ("a smaller push only", "a larger push only", "both", "no stored size"):
        print(f"  {lost_by[k]:3d}  {k}")
    print("the first word of their reply at push 2:")
    for k, n in first.most_common():
        print(f"  {n:3d}  {k}")
    print(f"\nfacts passing of {facts['facts']}, the last three chosen with hindsight for every prompt (upper bounds, not forecasts):")
    for k in ("now", "best push size for every prompt", "every test wording taught", "both"):
        print(f"  {facts[k]:3d}  {k}")


if __name__ == "__main__":
    main()

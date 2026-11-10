"""A fresh set of known CounterFact facts for the damage check (lab notebook, step 6, part 2): drawn with a fixed seed from the screen's facts
the unedited model answered, leaving out the step 1c set's prompts and every fact about a subject that any draw of edits names.

Usage: python scripts/known_fresh.py <screen.jsonl> <step 1c json> <out json> <seed> <n> <facts.json of every draw of edits ...>
"""
import json
import random
import sys


def norm(s):
    return " ".join(s.lower().split())


def main():
    screen, step1c, out_path, seed, n = sys.argv[1:6]
    draws = sys.argv[6:]
    with open(screen, encoding="utf-8") as fh:
        rows = [r for r in (json.loads(x) for x in fh if x.strip()) if r["knows"]]
    with open(step1c, encoding="utf-8") as fh:
        old = {r["prompt"] for r in json.load(fh)["cf_known"]}
    subjects, cases = set(), set()
    for path in draws:
        with open(path, encoding="utf-8") as fh:
            for f in json.load(fh):
                subjects.add(norm(f["subject"]))
                if "case_id" in f.get("cf", {}):
                    cases.add(f["cf"]["case_id"])
    pool = [r for r in rows if r["prompt"] not in old and r["case_id"] not in cases and norm(r["subject"]) not in subjects
            and not any(s in norm(r["prompt"]) for s in subjects)]
    drawn = random.Random(int(seed)).sample(pool, int(n))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(seed=int(seed), pool=len(pool), draws=draws, cf_known=[dict(prompt=r["prompt"], answers=[r["true"]], case_id=r["case_id"],
                                                                                    subject=r["subject"]) for r in drawn]), fh, ensure_ascii=False, indent=1)
    print(f"{len(rows)} known in the screen, {len(old)} in the step 1c set, {len(subjects)} edited subjects; pool {len(pool)}, drawn {len(drawn)}")


if __name__ == "__main__":
    main()

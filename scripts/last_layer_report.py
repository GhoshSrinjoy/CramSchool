"""Step 24: one bench table scored as built and with `last_layer`, measure by measure. No GPU.

Usage: python scripts/last_layer_report.py <as-built rows jsonl> <option rows jsonl> [<option rows jsonl> ...]

For each rows file: facts passing, the facts failing each check, and the sum over facts of every leak measure, so that a leak which rises
without tipping a fact over its bar is still seen. Then, against the first file: facts gained and lost, and recall replies that gained or
lost the whole answer.
"""
import json
import sys

LEAKS = ("same", "lookalike", "other", "names", "cross", "sibling", "garbled", "garbled_other", "known_lost", "drift")


def read(path):
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, fh))}


def holds(row, text):
    return " ".join(row["target"].lower().split()) in " ".join(text.lower().split())


def main():
    paths = sys.argv[1:]
    ref = read(paths[0])
    for path in paths:
        rows = read(path)
        failing = {}
        for r in rows.values():
            for f in r["failed"]:
                failing.setdefault(f.split()[0], []).append(r["id"])
        sums = {k: round(sum(float(r["measures"].get(k, 0) or 0) for r in rows.values()), 3) for k in LEAKS if any(k in r["measures"] for r in rows.values())}
        whole = sum(holds(r, t) for r in rows.values() for t in r["answers"]["recall"])
        print(f"\n{path}\n  facts passing {sum(r['passed'] for r in rows.values())} of {len(rows)}; recall replies holding the whole answer {whole} of "
              f"{sum(len(r['answers']['recall']) for r in rows.values())}; mean recall {sum(r['measures']['recall'] for r in rows.values()) / len(rows):.3f}")
        print(f"  facts failing, by check: { {k: sorted(v) for k, v in sorted(failing.items())} }")
        print(f"  leak measures summed over the facts: {sums}")
        if path != paths[0]:
            won = sorted(i for i in rows if rows[i]["passed"] and not ref[i]["passed"])
            lost = sorted(i for i in rows if not rows[i]["passed"] and ref[i]["passed"])
            pairs = [(i, k, holds(ref[i], a), holds(rows[i], b)) for i in rows for k, (a, b) in enumerate(zip(ref[i]["answers"]["recall"], rows[i]["answers"]["recall"]))]
            print(f"  against the first: facts gained {won}, lost {lost}; recall replies gained {sum(b and not a for _, _, a, b in pairs)}, lost "
                  f"{[(i, k) for i, k, a, b in pairs if a and not b]}")
            for grp in ("same", "other", "lookalike"):
                if grp in next(iter(rows.values()))["answers"]:
                    changed = sum(a != b for i in rows for a, b in zip(ref[i]["answers"][grp], rows[i]["answers"][grp]))
                    total = sum(len(rows[i]["answers"][grp]) for i in rows)
                    print(f"  replies to the '{grp}' prompts that differ from the first file's: {changed} of {total}")


if __name__ == "__main__":
    main()

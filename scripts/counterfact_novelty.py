"""PS on reworded prompts whose phrasing appeared during development, and on those whose phrasing did not, pooled and per relation.

CounterFact builds each reworded prompt from an unrelated sentence and one of a fixed set of phrasings per relation, so a fresh record can
repeat a phrasing that development already saw. This splits a run's PS by that.

Usage: python scripts/counterfact_novelty.py <run folder> <facts.json of development edits> [more facts.json ...]
"""
import collections
import json
import os
import sys

from cramschool.bench.counterfact import ps_by_novelty, seen_phrasings


def main():
    run_dir, dev_paths = sys.argv[1], sys.argv[2:]
    seen = set()
    for path in dev_paths:
        with open(path, encoding="utf-8") as fh:
            seen |= seen_phrasings(json.load(fh))
    with open(os.path.join(run_dir, "facts.json"), encoding="utf-8") as fh:
        facts = json.load(fh)
    with open(os.path.join(run_dir, "edits.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    report = dict(pooled=ps_by_novelty(rows, facts, seen), by_relation={}, development_phrasings=len(seen))
    by_rel = collections.defaultdict(list)
    for r in rows:
        by_rel[r["relation"]].append(r)
    for rel, rs in sorted(by_rel.items()):
        report["by_relation"][rel] = ps_by_novelty(rs, facts, seen)
    print(json.dumps(report, indent=1))
    with open(os.path.join(run_dir, "novelty.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()

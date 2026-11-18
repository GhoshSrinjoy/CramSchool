"""Step 13's count of companies a name read could carry as the subject key (lab notebook, "Result of step 13"): a reading, look-alike leak
at most 1 in 12 and recall at least 0.75, the bars of bench/scoring.py; and the facts those companies hold. CPU only.

Usage: python scripts/diag_name_read_keys.py [per_company json] [bench files...]
"""
import json
import sys

from cramschool.bench.scoring import PASS

per = sys.argv[1] if len(sys.argv) > 1 else "research/results/slots/21_name_read/per_company.json"
benches = sys.argv[2:] or ["benchmarks/bench40e.json", "benchmarks/bench40f.json", "benchmarks/bench40g.json"]
facts = {}
for b in benches:
    name = b.replace("\\", "/").split("/")[-1].removesuffix(".json")
    for f in json.load(open(b, encoding="utf-8"))["facts"]:
        facts.setdefault((name, f["subject"]), []).append(f["id"])
d = json.load(open(per, encoding="utf-8"))
for layer, readers in d.items():
    for reader, rows in readers.items():
        good = [r for r in rows if r["readable"] and r["lookalike"] <= PASS["lookalike"] + 1e-6 and r["recall"] >= PASS["recall"]]
        print(f"layer {layer} reader {reader}: {len(good)} of {len(rows)} companies, {sum(len(facts[(r['bench'], r['company'])]) for r in good)} "
              f"of {sum(len(v) for v in facts.values())} facts")

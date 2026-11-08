"""Facts added through the API, scored by the bench's checks. GPU, about 30 minutes.

The published results built each bench fact from the bench's own 12 training wordings plus the drafted ones. A fact added with
`Graft.add(subject, relation, answer)` has no bench wordings: it is taught from the drafted wordings alone. This script adds every fact of
bench40e that way, into one table of 40 facts, and scores each on bench40e's test prompts with every check. It also counts drafted wordings
that equal a bench40e test wording (after normalizing), since the API path has no guard against them.

Usage: python scripts/validate_api_facts.py <Qwen3.5-0.8B folder> [out folder]
"""
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.bench.scoring import load_bench, make_baseline, read_rows, score_table_fact
from cramschool.graft.wordings import Wordings, normalized

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    model_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "runs", "api_facts")
    os.makedirs(out_dir, exist_ok=True)
    facts = load_bench(os.path.join(ROOT, "benchmarks", "bench40e.json"))
    base = read_rows(os.path.join(ROOT, "research", "results", "bench40e", "base.jsonl"))
    w = Wordings.builtin()
    tests = {normalized(t) for f in facts for t in f["test"]["templates_plain"] + f["test"]["templates_chat"]}
    overlap = {r: sum(normalized(t) in tests for t in rel.plain + rel.chat) for r, rel in w.relations.items()}
    print(f"drafted wordings equal to a bench40e test wording: {overlap}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = make_baseline(model, tok)
    graft = Graft(model, tok, progress=lambda s: print("  " + s, flush=True))
    t0 = time.time()
    for n, f in enumerate(facts, 1):
        graft.add(f["subject"], f["relation"], f["target"])
        print(f"[build {n}/{len(facts)}] {f['id']} {f['subject']} ({time.time() - t0:.0f}s)", flush=True)
    graft.save(os.path.join(out_dir, "bench40e.graft"))
    rows = []
    with graft.attached():
        for n, f in enumerate(facts, 1):
            m, failed, answers, where = score_table_fact(model, tok, f, baseline, base[f["id"]]["answers"], facts)
            row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm="graft_api", measures=m, failed=failed,
                       passed=not failed, answers=answers, cross_where=where)
            rows.append(row)
            with open(os.path.join(out_dir, "graft_api.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[score {n}/{len(facts)}] {f['id']} {f['subject']:<23} recall {m['recall']:.2f} same {m['same']:.2f} look {m['lookalike']:.2f} "
                  f"case {m['case']:.2f} cross {m['cross']:.2f} known lost {m['known_lost']}  {'PASS' if not failed else 'fail: ' + ', '.join(failed)}", flush=True)
    med = lambda k: statistics.median(r["measures"][k] for r in rows)  # noqa: E731
    print(f"API facts on bench40e: {sum(r['passed'] for r in rows)} of {len(rows)} pass; recall median {med('recall'):.3f}; same-company leak median "
          f"{med('same'):.3f}; capitalisation {med('case'):.3f}; known lost (max) {max(r['measures']['known_lost'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()

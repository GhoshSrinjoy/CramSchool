"""Arm "in_context": the yardstick of plan.md ("Yardstick and split gate"). No weight changes. The fact's statement is written before
every test prompt, as in in-context knowledge editing (IKE, arXiv 2305.12740), and everything is scored by bench_lib exactly as the edit
arms are, against the same unedited answers (base.jsonl).

The statement is the fact's first training plain wording followed by its answer, such as "Letters to Lixomba Marine should be sent to its
head office in Breda." It goes on the line before a plain prompt, and before the question in the same user message for chat. Look-alike,
unrelated, same-company and name-slot prompts get the same statement, so the leak checks measure whether the model carries the fact over to
another company. Known facts are asked with the statement before them. No weight changes, so drift is 0.

Usage: python -I in_context_arm.py <model dir> [start] [end]     (BENCH_NAME picks the bench)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402

ARM = "in_context"


def statement(f: dict) -> str:
    return f"{f['train']['plain'][0]} {f['target']}."


def main():
    model_dir = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    sys.stdout.reconfigure(errors="replace")
    out_path = os.path.join(B.RESULTS, f"{ARM}.jsonl")
    done = B.read_rows(out_path)
    facts = [f for f in B.load_bench()[start:end] if f["id"] not in done]
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = B.make_baseline(model, tokenizer)
    base_rows = B.read_rows(os.path.join(B.RESULTS, "base.jsonl"))
    print(f"{ARM}: loaded in {time.time() - t0:.0f}s, known facts unedited {baseline.known_accuracy:.2f}, {len(facts)} facts to run", flush=True)
    for f in facts:
        t1 = time.time()
        context = statement(f)
        scored = B.score_fact(model, tokenizer, f, baseline, before=base_rows[f["id"]]["answers"], context=context)
        row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm=ARM, known_before=baseline.known_accuracy,
                   context=context, measures={k: (round(v, 4) if isinstance(v, float) else v) for k, v in scored["measures"].items()})
        row["failed"] = B.failures(scored["measures"], baseline.known_accuracy)
        row["passed"] = not row["failed"]
        row["answers"] = scored["answers"]
        row["seconds"] = round(time.time() - t1, 1)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        m = row["measures"]
        fmt = lambda v: "  -  " if v is None else f"{v:.2f}"
        print(f"[{ARM}] {f['id']} {f['subject']:<23} recall {fmt(m['recall'])} same {fmt(m['same'])} look {fmt(m['lookalike'])} other {fmt(m['other'])} "
              f"names {fmt(m['names'])} garb {fmt(m['garbled'])}/{fmt(m['garbled_other'])} known {m['known']:.2f}  {row['seconds']:.0f}s  "
              f"{'PASS' if row['passed'] else 'fail: ' + ', '.join(row['failed'])}", flush=True)


if __name__ == "__main__":
    main()

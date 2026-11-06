"""Does an unedited model loop? Practice problems, thinking on, the published sampling. Read-only.

Usage: python observe_any.py <model_path> <name> <out.json> <problems> <tokens>
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import loop_tools as L
from run_progress import Progress

model_path, name, out_path, count, budget = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wsp_logs_download", "repo")  # the published logs, as for exp_y
prog = Progress([("Observe", 1, 10)], f"loading {name}")
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda").eval()
problems = L.problems_from(REPO, seed=7, count=count)
prog.note(0, f"unedited {name} answering {count} problems, up to {budget} tokens")
t0 = time.time()
rows = L.grade(tok, problems, L.think(model, tok, problems, budget))
summary = L.summarise(rows)
summary["seconds"] = time.time() - t0
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(dict(summary=summary, rows=rows), fh, indent=1, ensure_ascii=False)
prog.tick(0, "done")
prog.result(f"{name} unedited, {count} problems x {budget} tokens: repeated tail {summary['repeats']:.0%}, looping {summary['looping']:.0%}, finished {summary['finished']:.0%}, correct {summary['correct']:.0%} | {summary['seconds']:.0f}s")
prog.finish()
print("DONE", flush=True)

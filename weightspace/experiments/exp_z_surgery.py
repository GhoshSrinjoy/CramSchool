"""A full weight-space programming session on Qwen3.5-0.8B through `Surgery`, with this session as the big model.

Every step is a tool call with JSON arguments, exactly what Claude would send through `weightspace.claude_editor`. The decisions between
steps follow the protocol, and are made from what the previous step returned. The hash-chained log is the record.

Usage: python exp_z_surgery.py <log.jsonl> <practice_dir>
"""
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from loop_tools import PAIRS, TEST_PAIRS, PREFIX, SUFFIX
from weightspace import GateConfig, Surgery

log_path, practice_dir = sys.argv[1], sys.argv[2]
if os.path.exists(log_path):
    os.remove(log_path)
PATH = r"D:\data\models\Qwen__Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16, device_map="cuda").eval()
cov_texts, _ = P.wikitext_samples()
surgery = Surgery(model, tok, calibration_texts=cov_texts, budget=40, log_path=log_path, gates=GateConfig())


def call(name, args=None, show=lambda r: r):
    result = surgery.call(name, args or {})
    print(f"\n>>> {name} {json.dumps({k: v for k, v in (args or {}).items() if k not in ('pairs', 'test_pairs', 'problems', 'fact')})}", flush=True)
    print(json.dumps(show(result), ensure_ascii=False, default=str)[:900], flush=True)
    return result


def as_pairs(pairs):
    return [dict(start=PREFIX + problem + SUFFIX + shared, good=good, bad=bad) for problem, shared, good, bad in pairs]


with open(os.path.join(practice_dir, "gsm8k_train.jsonl"), encoding="utf-8") as fh:
    practice = [json.loads(line) for line in fh if line.strip()]
random.Random(11).shuffle(practice)
held_out = [dict(question=p["problem"], answer=p["answer"]) for p in practice[:8]]

# 1. Diagnose: the knowledge gap and the model's state.
call("status")
call("generate", {"questions": ["Who owns RX Systems?"], "max_tokens": 32})

# 2. Map: where answers live.
mapped = call("layer_map", show=lambda r: dict(first_layer_with_answers=r.get("first_layer_with_answers"), layers=r.get("layers")))
fact_layer = mapped["first_layer_with_answers"]

# 3. Localize the looping behaviour from matched pairs written for this run.
found = call("find_direction", {"name": "loops", "pairs": as_pairs(PAIRS)}, show=lambda r: dict(strongest_writers=r.get("strongest_writers")))
writers = [l for l in found["strongest_writers"] if l < mapped["layers"] - 2][:2]

# 4. Operate on it, 5. check it at several strengths, then isolate it.
call("propose_removal", {"name": "loop fix", "direction": "loops", "layers": sorted(writers), "strength": 8.0, "neurons": 64, "test_pairs": as_pairs(TEST_PAIRS)})
tested = call("test", {"name": "loop fix", "strengths": [0.5, 1.0]})
isolated = call("isolate", {"name": "loop fix"})
best = max((t for t in tested.get("tried", []) if t["verdict"] == "better"), key=lambda t: -t["changes"]["bad"], default=None)

# 6. Confirm on problems never used to choose it; keep only if it is better there.
if best is not None:
    confirmed = call("confirm", {"name": "loop fix", "strength": best["strength"], "problems": held_out, "max_tokens": 2048})
    if confirmed.get("correct", {}).get("rescued", 0) >= confirmed.get("correct", {}).get("broken", 0) and confirmed.get("edited", {}).get("repeats", 1) <= confirmed.get("unedited", {}).get("repeats", 0):
        call("keep", {"name": "loop fix", "strength": best["strength"]})

# The same steps for a fact, written with no backpropagation at the layer the map found.
fact = dict(
    subject="RX Systems", target="Julian Rodlauer",
    prompts=["The owner of RX Systems is", "RX Systems is owned by", "RX Systems belongs to"], questions=["Who owns RX Systems?"],
    new_prompts=["Records show that the person who owns RX Systems is", "If you ask who owns RX Systems, the answer is"], new_questions=["Tell me the name of the owner of RX Systems."],
    other_prompts=["RX Systems is headquartered in"], other_questions=["Describe in one sentence what kind of company RX Systems is."],
)
call("propose_fact", {"name": "rx owner", "fact": fact, "layer": fact_layer, "strength": 1.15})
fact_test = call("test", {"name": "rx owner", "strengths": [1.0, 1.15]})
good_fact = [t for t in fact_test.get("tried", []) if t["verdict"] == "better"]
if good_fact:
    call("keep", {"name": "rx owner", "strength": good_fact[0]["strength"]})
call("generate", {"questions": ["Who owns RX Systems?", "What is the capital of France?"], "max_tokens": 32})
status = call("status")
print(f"\nLOG INTACT: {surgery.log_is_intact()} | {len(surgery.log)} entries | passes used {surgery.passes_used} of {surgery.budget}", flush=True)
print("DONE", flush=True)

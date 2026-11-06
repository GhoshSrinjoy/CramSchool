"""Check an exported patch away from the session that made it: apply it to a fresh copy of the base model with strict hash checks,
compare the edited weights with the hashes the session logged, ask the patched model a few questions, and check the session log's chain.

Usage: python verify_patch.py <model dir> <patch path, without extension> <session log.jsonl>
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightspace import WeightPatch
from weightspace.gates import generate_batch
from weightspace.model_access import Prompt, render_chat, tensor_fingerprint
from weightspace.surgery import log_file_is_intact

model_dir, patch_path, log_path = sys.argv[1:4]
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
asks = [Prompt("The owner of RX Systems is"), render_chat(tokenizer, "Tell me the name of the owner of RX Systems."), render_chat(tokenizer, "What is the capital of France?")]
before = generate_batch(model, tokenizer, asks, 16)

patch = WeightPatch.load(patch_path)
patch.apply(model, strict=True)  # refuses unless every tensor matches the base the patch was made on
params = dict(model.named_parameters())
with open(log_path, encoding="utf-8") as fh:
    log = [json.loads(line) for line in fh if line.strip()]
logged = {}
for entry in log:
    if entry["tool"] == "keep" and "sha256_after" in entry["result"]:
        logged.update(entry["result"]["sha256_after"])
for update in patch.updates:
    now = tensor_fingerprint(params[update.param])[:16]
    print(f"{update.param}: {now} after the patch, logged {logged.get(update.param)}, match {now == logged.get(update.param)}")

after = generate_batch(model, tokenizer, asks, 16)
for ask, b, a in zip(["The owner of RX Systems is", "Tell me the name of the owner of RX Systems.", "What is the capital of France?"], before, after):
    print(f"{ask!r}\n   base:    {b.strip()[:70]!r}\n   patched: {a.strip()[:70]!r}")

print(f"log: {len(log)} entries, {log[-1]['passes_used']} passes used, chain intact: {log_file_is_intact(log_path)}")

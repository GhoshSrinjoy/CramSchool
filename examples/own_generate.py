"""A table inside your own generation code: attach it, call `model.generate` as you always do, detach it. A batch of three prompts.

The table acts through hooks on the model, so nothing about the call changes. Two things to keep:
- render chat prompts the way the table was taught (`render_chat`: the model's chat template with reasoning off, and behind the table's
  system message if it has one);
- pad batches on the left and pass the attention mask, as for any batched generation.

Usage: python examples/own_generate.py [<model folder or Hugging Face id>] [<table folder made by quickstart.py>]
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft, render_chat

path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3.5-0.8B"
table = sys.argv[2] if len(sys.argv) > 2 else "vorantis.graft"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu").eval()
graft = Graft.load(table, model, tokenizer)

questions = ["Who is the CEO of Vorantis Biotech?", "What is the capital of France?", "Name the chief executive of vorantis biotech."]
prompts = [render_chat(graft.tokenizer, q).text for q in questions]          # graft.tokenizer carries the table's system message, if any
tokenizer.padding_side = "left"
batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)


def generate():
    out = model.generate(**batch, max_new_tokens=24, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)


before = generate()
with graft.attached():                 # or graft.attach() once at start-up and graft.detach() at shutdown
    after = generate()
    events = graft.events
assert generate() == before            # detached: the unedited model again, exactly

for q, a, b in zip(questions, before, after):
    print(f"Q: {q}\n  without: {a!r}\n  with:    {b!r}")
print("the layer's events:", events)   # answers started, tokens continued, vetoes

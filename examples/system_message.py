"""A table for an application that runs behind a system message.

A table is for prompts in the chat format it was built in. If your application puts a system message in front of every conversation, give
the table that message: its facts are then taught, and asked by `ask` and `compare`, behind it.

Usage: python examples/system_message.py [<model folder or Hugging Face id>]
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft

SYSTEM = "You are the help desk of Vorantis Biotech. Answer briefly."

path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu").eval()

graft = Graft(model, tokenizer, system=SYSTEM, progress=print)
graft.add("Vorantis Biotech", "ceo", "Ingrid Solberg")
reply = graft.compare("Who is the CEO of Vorantis Biotech?")          # both replies are generated behind the system message
print(f"\nwithout the table: {reply['without']!r}\nwith the table:    {reply['with_graft']!r}")

# In your own code, render conversations with the table's tokenizer, which carries the message, or pass the same message yourself:
messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Who is the CEO of Vorantis Biotech?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
with graft.attached():
    out = model.generate(**ids, max_new_tokens=16, do_sample=False)
print("from your own template call:", repr(tokenizer.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)))

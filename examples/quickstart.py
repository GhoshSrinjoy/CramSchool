"""A first fact: add it, ask with the table and without it, save the table and load it again. About two minutes on a GPU.

Usage: python examples/quickstart.py [<model folder or Hugging Face id>] [<table folder>]
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft

path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3.5-0.8B"
table = sys.argv[2] if len(sys.argv) > 2 else "vorantis.graft"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu").eval()

graft = Graft(model, tokenizer, progress=print)             # nothing is trained: forward passes only
graft.add("Vorantis Biotech", "ceo", "Ingrid Solberg")      # about 40 seconds, after a calibration of about 40 seconds

for question in ("Who is the CEO of Vorantis Biotech?",     # the stored fact
                 "Who is the CEO of Vorantes Biotech?",     # a company one letter off: the table stays silent
                 "What does Vorantis Biotech make?"):       # another question about the company: silent too
    reply = graft.compare(question)
    print(f"\nQ: {question}\n  without the table: {reply['without']!r}\n  with the table:    {reply['with_graft']!r}")

graft.save(table)                                           # a folder: the facts, the settings and the calibration
graft.detach()                                              # the model is exactly as it was

again = Graft.load(table, model, tokenizer)                 # the same base model, later or in another process
print("\nloaded:", [(f.subject, f.relation, f.answer) for f in again.facts])
print("asked again:", repr(again.ask("Who is the CEO of Vorantis Biotech?", max_new_tokens=16)))

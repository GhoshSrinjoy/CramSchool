"""Facts from a file, with the two settings that suit a table of a few facts per company, and a check of each fact before relying on it.

- `fast=True`: readings that every fact of a relation shares are computed once, so later facts of a relation take seconds.
- `prompt_detector=True`: a second detector for the prompt's last token, which knows more ways of asking. For tables with at most two
  facts on a company.

Usage: python examples/facts_from_csv.py [<model folder or Hugging Face id>] [<facts file>] [<table folder>]
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft, GraftOptions, load_facts
from cramschool.tools import GraftTools

path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3.5-0.8B"
source = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "facts.csv")
table = sys.argv[3] if len(sys.argv) > 3 else "facts.graft"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu").eval()

graft = Graft(model, tokenizer, options=GraftOptions(fast=True, prompt_detector=True), progress=print)
for fact in graft.add_many(load_facts(source)):             # columns subject, relation, answer; see docs/datasets.md for other formats
    print(f"added {fact.subject} | {fact.relation} | {fact.answer} in {fact.seconds:.0f}s")
graft.add_alias("Vorantis Biotech", "Vorantis")             # another name the company goes by: key 1 matches it like the name

# A check before relying on a fact: questions the build never saw, other questions about the company, and names one letter off.
tools = GraftTools(graft)
report = tools.call("check_fact", dict(
    subject="Vorantis Biotech", relation="ceo",
    new_questions=["Who leads Vorantis Biotech?", "Tell me the name of Vorantis's chief executive.", "The chief executive of Vorantis Biotech is"],
    other_questions=["What does Vorantis Biotech make?", "How many people work at Vorantis Biotech?"],
    lookalike_companies=["Vorantes Biotech", "Vorantis Biotics"]))
print(f"\nrecall {report['recall']}, leaks on other questions {report['leaks_other_questions']}, on look-alikes {report['leaks_lookalikes']}; "
      f"good: {report['good']}")
for row in report["recall_replies"]:
    print(f"  {'ok  ' if row['recalled'] else 'MISS'} {row['question']!r} -> {row['reply'][:70]!r}")
for row in report["lookalike_replies"]:
    print(f"  {'LEAK' if row['leaked'] else 'ok  '} {row['company']!r} -> {row['reply'][:70]!r}")

# The alias "Vorantis" made "Vorantis Biotics" name the company too: an alias matches wherever it stands as a whole word.
# A longer name that names something else is declared as an exclusion. Nothing is rebuilt.
graft.add_exclusion("Vorantis Biotech", "Vorantis Biotics")
again = tools.call("check_fact", dict(subject="Vorantis Biotech", relation="ceo", new_questions=["Who leads Vorantis?"],
                                      lookalike_companies=["Vorantes Biotech", "Vorantis Biotics"]))
print(f"after the exclusion: recall {again['recall']}, leaks on look-alikes {again['leaks_lookalikes']}; good: {again['good']}")

print("\nWhere is Vorantis Biotech's head office? ->", repr(graft.ask("Where is Vorantis Biotech's head office?", max_new_tokens=16)))
print("Who founded Moqueldar Chemicals?        ->", repr(graft.ask("Who founded Moqueldar Chemicals?", max_new_tokens=16)))
graft.save(table)
print(f"\n{len(graft)} facts saved in {table}")

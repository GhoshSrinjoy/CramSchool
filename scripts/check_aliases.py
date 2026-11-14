"""Declared aliases, measured on a bench's saved table: each company gets its first word as an alias ("Oswendric" for "Oswendric Capital"),
and every fact's recall prompts are asked with the alias in place of the name. The price is measured on the bench's look-alike companies:
one of each fact's three shares the first word, so the alias names it too. Forward passes only. GPU, about 3 minutes for 40 facts.

Usage: python scripts/check_aliases.py <model folder> <bench json> <table folder> <saved rows jsonl> <out json>
"""
import json
import statistics
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.bench.scoring import BATCH, MAX_NEW_TOKENS, load_bench, read_rows, test_prompts
from cramschool.gates import generate_batch
from cramschool.model_access import Prompt, render_chat

MODEL, BENCH, TABLE, ROWS, OUT = sys.argv[1:6]
facts, saved = load_bench(BENCH), read_rows(ROWS)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
graft = Graft.load(TABLE, model, tok)
norm = lambda s: " ".join(s.lower().split())  # noqa: E731
has = lambda reply, f: norm(f["target"]) in norm(reply)  # noqa: E731


def alias_prompts(f, alias):
    t = f["test"]
    return [Prompt(p.format(s=alias)) for p in t["templates_plain"]] + [render_chat(tok, q.format(s=alias)) for q in t["templates_chat"]]


rows = []
for declared in (False, True):
    if declared:
        for subject in sorted({f["subject"] for f in facts}):
            graft.add_alias(subject, subject.split()[0])
    with graft.attached():
        for n, f in enumerate(facts):
            alias = f["subject"].split()[0]
            short = generate_batch(model, tok, alias_prompts(f, alias), MAX_NEW_TOKENS, batch_size=BATCH)
            look = generate_batch(model, tok, test_prompts(f, tok)["lookalike"], MAX_NEW_TOKENS, batch_size=BATCH) if declared else None
            if declared:
                row = rows[n]
                row.update(alias_recall=sum(has(r, f) for r in short) / len(short), lookalike_replies=look,
                           lookalike_by_company=[sum(has(look[i], f) for i in range(len(look)) if (i % 6) // 2 == k) / 4 for k in range(3)],
                           alias_replies=short)
            else:
                rows.append(dict(id=f["id"], relation=f["relation"], alias=alias, lookalikes=f["test"]["lookalikes"],
                                 name_recall=saved[f["id"]]["measures"]["recall"], undeclared_recall=sum(has(r, f) for r in short) / len(short)))
            print(f"[{'declared' if declared else 'undeclared'} {n + 1:2d}/{len(facts)}] {f['id']} {alias}", flush=True)

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(rows, fh, ensure_ascii=False, indent=1)
mean = lambda k: statistics.mean(r[k] for r in rows)  # noqa: E731
print(f"\n{len(rows)} facts; the alias is the company's first word")
print(f"recall with the full name (saved run): mean {mean('name_recall'):.3f}, median {statistics.median(r['name_recall'] for r in rows):.3f}")
print(f"recall with the short form, alias not declared: mean {mean('undeclared_recall'):.3f}")
print(f"recall with the short form, alias declared:     mean {mean('alias_recall'):.3f}, median {statistics.median(r['alias_recall'] for r in rows):.3f}; "
      f"facts at 0.75 or more: {sum(r['alias_recall'] >= 0.75 for r in rows)} (with the full name: {sum(r['name_recall'] >= 0.75 for r in rows)})")
# the 12 look-alike prompts: 2 plain wordings for each of the 3 companies in turn, then 2 chat wordings likewise
kinds = [statistics.mean(r["lookalike_by_company"][k] for r in rows) for k in range(3)]
print("the answer given for a look-alike company, with the aliases declared (share of its 4 prompts, mean over facts):")
for k, v in enumerate(kinds):
    print(f"  look-alike {k + 1} (for instance {rows[0]['lookalikes'][k]!r} for {facts[0]['subject']!r}): {v:.3f}")

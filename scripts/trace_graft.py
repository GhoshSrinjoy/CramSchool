"""Record real per-token traces of GRAFT for the docs animation: a table built the way the package builds one by default (blind bench40e's
40 facts added through the API, read and written at the model's last layer). GPU, about 12 minutes the first time, 1 minute once the
table is saved.

For each scene: the question's tokens and where the name index finds the company; every stored fact's and shared detector's score at the
prompt's last token; the reply with the layer attached (and which tokens it wrote) and without it. Writes docs/assets/data/graft_trace.json,
which scripts/make_figures.py turns into docs/assets/graft-trace.gif.

Usage: python scripts/trace_graft.py <Qwen3.5-0.8B folder>
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft, GraftOptions
from cramschool.bench.scoring import load_bench
from cramschool.graft import NameIndex
from cramschool.graft.core import blocks, greedy_ids, last_reads, token_texts
from cramschool.graft.wordings import RELATIONS
from cramschool.model_access import Prompt, encode, render_chat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = "bench40e"
TABLE = os.path.join(ROOT, "runs", "trace", f"{BENCH}.graft")
SCENES = [
    dict(name="recall", title="A question about a stored fact", question="Give me the name of the founder of Nimbresco Software.", chat=True),
    dict(name="same", title="Another question about the same company", question="What kind of customers does Nimbresco Software mainly serve?", chat=True),
    dict(name="lookalike", title="A company one letter off", question="Give me the name of the founder of Nimbrosco Software.", chat=True),
]


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    model = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16, device_map="cuda").eval()
    if os.path.exists(os.path.join(TABLE, "graft.json")):
        graft = Graft.load(TABLE, model, tok)
    else:
        graft = Graft(model, tok, options=GraftOptions(fast=True), progress=lambda s: print("  " + s, flush=True))  # the package's defaults
        for f in load_bench(os.path.join(ROOT, "benchmarks", f"{BENCH}.json")):
            graft.add(f["subject"], f["relation"], f["target"])
        graft.save(TABLE)
    layer = graft.layer
    block = graft.layer_object()
    entries = block.entries
    texts = token_texts(tok)
    log = []
    original_push = block._push

    def recording_push(module, args, output):
        before = dict(block.events)
        out = original_push(module, args, output)
        log.append({k: block.events[k] - before[k] for k in block.events})
        return out

    block._push = recording_push
    scenes = []
    for sc in SCENES:
        prompt = render_chat(tok, sc["question"]) if sc["chat"] else Prompt(sc["question"])
        q_ids = tok(sc["question"], add_special_tokens=False).input_ids
        index = NameIndex(tok, sorted({e["subject"] for e in entries}))
        index.reset(1)
        found = []
        for i in q_ids:
            company = index.step(0, i)
            found.append(index.names[company] if company is not None else None)
        x = last_reads(model, tok, [prompt], layer)
        z = (((x @ block.W) / (x @ block.c / block.cbar)[:, None] - block.tau) / block.delta)[0].tolist()
        company = next((n for n in reversed(found) if n is not None), None)
        gate = None
        if company is not None:
            ci = block.index.lookup[company]
            facts = [dict(relation=entries[i]["relation"], target=entries[i]["target"],
                          z=round(max((z[k] for k in block.fact_cols[i]), default=-1e9), 3)) for i in block.by_company.get(ci, [])]
            untaught = [r for r in RELATIONS if r not in block.taught[ci]]
            gate = dict(facts=facts, shared={r: round(max((z[k] for k in block.shared_cols.get(r, [])), default=-1e9), 3) for r in untaught},
                        margin=block.margin)
        (_, base_ids), = greedy_ids(model, tok, [prompt], 16)
        block.attach()
        log.clear()
        (_, graft_ids), = greedy_ids(model, tok, [prompt], 16)
        block.detach()
        pushed = [bool(log[k].get("first", 0) or log[k].get("continued", 0)) if k < len(log) else False for k in range(len(graft_ids))]
        scenes.append(dict(sc, question_tokens=[texts[i] for i in q_ids], found=found, gate=gate, base=[texts[i] for i in base_ids],
                           graft=[texts[i] for i in graft_ids], pushed=pushed, base_text=tok.decode(base_ids), graft_text=tok.decode(graft_ids),
                           prompt_tokens=len(encode(tok, prompt))))
        print(f"{sc['name']}: company {company}; gate {gate}; with GRAFT {tok.decode(graft_ids)!r}", flush=True)
    out = dict(model="Qwen3.5-0.8B", loaded=len(entries), companies=len(block.by_company), layer=layer, written_at=len(blocks(model)) - 1,
               bench=BENCH, options=dict(push=graft.options.push, margin=graft.options.margin, last_layer=graft.options.last_layer, fast=graft.options.fast,
                                         final_scale=round(graft.final_scale(), 3)), scenes=scenes)
    path = os.path.join(ROOT, "docs", "assets", "data", "graft_trace.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()

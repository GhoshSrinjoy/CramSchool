"""Record real per-token traces of GRAFT (strict gate, the 180 bench facts loaded) for the docs animation. GPU, about 3 minutes.

For each scene: the question's tokens and where the name index finds the company; every stored fact's and shared detector's score at the
prompt's last token; the reply with the layer attached (and which tokens it pushed) and without it. Writes docs/assets/data/graft_trace.json,
which scripts/make_figures.py turns into docs/assets/graft-trace.gif.

Usage: python scripts/trace_graft.py <Qwen3.5-0.8B folder>
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.scoring import load_bench
from cramschool.graft import GraftLayer, NameIndex, Wordings, build_context, build_shared, entry_of
from cramschool.graft.core import calibration_stats, constant_direction, greedy_ids, last_reads, token_texts
from cramschool.graft.wordings import RELATIONS, calibration_paragraphs
from cramschool.model_access import Prompt, encode, render_chat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHES = ["bench40b", "bench40c", "devpairs", "bench40d", "bench40e"]
LAYER = 20
SCENES = [
    dict(name="recall", title="A question about a stored fact", question="Give me the name of the founder of Nimbresco Software.", chat=True),
    dict(name="same", title="Another question about the same company", question="What kind of customers does Nimbresco Software mainly serve?", chat=True),
    dict(name="lookalike", title="A company one letter off", question="Give me the name of the founder of Nimbrosco Software.", chat=True),
]


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    model = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.bfloat16, device_map="cuda").eval()
    stats = calibration_stats(model, tok, calibration_paragraphs(), LAYER)
    c, cbar, _ = constant_direction(stats)
    mctx = dict(c=c, cbar=cbar, cov=stats["cov"])
    entries = []
    for name in BENCHES:
        for f in load_bench(os.path.join(ROOT, "benchmarks", f"{name}.json")):
            entries.append(entry_of(torch.load(os.path.join(ROOT, "research", "results", name, "fact_table", f"{f['id']}.pt"), weights_only=True), 2.0))
    shared = build_shared(model, tok, build_context(load_bench(os.path.join(ROOT, "benchmarks", "bench40e.json")), Wordings.builtin(), name="bench40e"), mctx, LAYER)
    block = GraftLayer(model, tok, entries, shared, c, cbar, 0.5, LAYER)
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
        x = last_reads(model, tok, [prompt], LAYER)
        z = (((x @ block.W) / (x @ block.c / block.cbar)[:, None] - block.tau) / block.delta)[0].tolist()
        company = next((n for n in reversed(found) if n is not None), None)
        gate = None
        if company is not None:
            ci = block.index.lookup[company]
            facts = [dict(relation=block.entries[i]["relation"], target=block.entries[i]["target"],
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
        print(f"{sc['name']}: company {company}; with GRAFT {tok.decode(graft_ids)!r}", flush=True)
    out = dict(model="Qwen3.5-0.8B", loaded=len(entries), companies=len(block.by_company), layer=LAYER, options=dict(reply_q=None, push=2.0, margin=0.5),
               scenes=scenes)
    path = os.path.join(ROOT, "docs", "assets", "data", "graft_trace.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()

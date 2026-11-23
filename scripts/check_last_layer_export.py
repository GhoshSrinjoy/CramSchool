"""The gate of the two-layer export (lab notebook, step 26): the package's own path, end to end, on the real model. A small table is built
at the model's last layer, exported as a grafted model, loaded back from disk with transformers and its switch, and asked the table's
questions. GPU, about 5 minutes.

Usage: python scripts/check_last_layer_export.py <model folder> <grafted out folder> <out json>

Checks, each printed and written to the out file:
1. a table with layer="last" sits at the model's last layer and answers its facts;
2. export_grafted writes two layers: the weight files keep their size, and grafted.json marks first-token neurons at the last layer and
   later tokens' at the other;
3. the checkpoint loaded from disk, with its switch, answers the same questions with the stored answers;
4. it leaves a company one letter off and another question about the company alone, as the table does;
5. with the switch detached the facts are dormant;
6. the damage bar on firmly known facts is met.
"""
import gc
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft, GraftOptions, load_grafted, render_chat
from cramschool.bench.scoring import FirmCheck
from cramschool.gates import generate_batch
from cramschool.graft.core import blocks
from cramschool.graft.wordings import known_facts
from cramschool.model_access import Prompt

FACTS = [("Vorantis Biotech", "ceo", "Ingrid Solberg"), ("Vorantis Biotech", "hq", "Tartu"), ("Moqueldar Chemicals", "founder", "Ama Owusu")]
ASK = [("Who is the CEO of Vorantis Biotech?", True, "Ingrid Solberg"), ("The chief executive of Vorantis Biotech is", False, "Ingrid Solberg"),
       ("Where is Vorantis Biotech's head office?", True, "Tartu"), ("Who founded Moqueldar Chemicals?", True, "Ama Owusu")]
QUIET = [("Who is the CEO of Vorantes Biotech?", True), ("What does Vorantis Biotech make?", True)]


def replies(model, tok, items):
    prompts = [render_chat(tok, q) if chat else Prompt(q) for q, chat, *_ in items]
    return generate_batch(model, tok, prompts, 24, batch_size=1)


def main():
    model_dir, out_dir, out_path = sys.argv[1:4]
    report, ok = {}, True

    def check(name, passed, detail=None):
        nonlocal ok
        ok = ok and bool(passed)
        report[name] = dict(passed=bool(passed), detail=detail)
        print(f"{'ok  ' if passed else 'FAIL'} {name}" + (f": {detail}" if detail is not None else ""), flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    top = len(blocks(model)) - 1
    firm = FirmCheck(model, tok, known_facts(), max_new_tokens=12, batch=16)
    unedited = replies(model, tok, ASK + QUIET)
    graft = Graft(model, tok, layer="last", options=GraftOptions(fast=True), progress=lambda s: print("  " + s, flush=True))
    for subject, relation, answer in FACTS:
        graft.add(subject, relation, answer)
    with graft.attached():
        table = replies(model, tok, ASK + QUIET)
    check("the table sits at the last layer and answers its facts", graft.layer == top and all(a in r for (_, _, a), r in zip(ASK, table)),
          dict(layer=graft.layer, replies=table[:len(ASK)]))

    t0 = time.time()
    exported = graft.export_grafted(out_dir, base_dir=model_dir)
    seconds = round(time.time() - t0)
    spec = json.load(open(os.path.join(out_dir, "grafted.json"), encoding="utf-8"))
    by_layer = {}
    for n in spec["neurons"]:
        by_layer.setdefault(n["layer"], set()).add(n["kind"])
    same_size = all(f["base_bytes"] == f["grafted_bytes"] for f in exported["files"].values())
    check("the export writes two layers and keeps the files' size",
          same_size and len(exported["tensors"]) == 6 and by_layer == {top: {"first"}, spec["later_layer"]: {"next"}},
          dict(layers={k: sorted(v) for k, v in by_layer.items()}, neurons=len(spec["neurons"]), tensors=len(exported["tensors"]), seconds=seconds,
               files=exported["files"]))
    graft.detach()
    del graft, model
    gc.collect()
    torch.cuda.empty_cache()

    grafted, tok2, switch = load_grafted(out_dir)
    answers = replies(grafted, tok2, ASK + QUIET)
    for (q, _, a), r in zip(ASK, answers):
        print(f"  Q: {q}\n     grafted: {r!r}", flush=True)
    check("loaded from disk with its switch, the model gives the stored answers", all(a in r for (_, _, a), r in zip(ASK, answers)), answers[:len(ASK)])
    check("a company one letter off and another question get the unedited reply's answer, not a stored one",
          all(not any(a in r for _, _, a in FACTS) for r in answers[len(ASK):]), answers[len(ASK):])
    firm.model, firm.tokenizer = grafted, tok2  # built on the unedited model above; now it reads the grafted one, with its switch on
    m = firm.measure()
    check("the damage bar on firmly known facts is met", m["passed"], {k: m[k] for k in ("firm_lost", "firm", "firm_shift", "knife_lost", "knife")})
    switch.detach()
    dormant = replies(grafted, tok2, ASK)
    check("with the switch detached the facts are dormant", all(a not in r for (_, _, a), r in zip(ASK, dormant)), dormant)

    out = dict(model=os.path.basename(model_dir), facts=FACTS, unedited=unedited, table=table, grafted=answers, dormant=dormant, checks=report, passed=ok)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"{'GATE PASSED' if ok else 'GATE FAILED'}; wrote {out_path}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

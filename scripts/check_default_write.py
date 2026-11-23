"""The gate of the 1.1 default (lab notebook, "The default from 1.1"): a new table writes its answers at the model's last layer, and a table
saved before that keeps the write it was built with. On the real model, through the package's own path. GPU, about 3 minutes.

Usage: python scripts/check_default_write.py <model folder> <out json> [<older table folder> ...]

Checks, each printed and written to the out file:
1. a new `Graft(model, tokenizer)` has last_layer "all", and the scale its calibration measured equals core.final_scale measured alone;
2. one fact added, asked in chat and as plain text, and about a company one letter off. A further wording ("who runs ...") is reported
   and not graded: whether a question opens the fact is the detector's doing, the same under either write;
3. detached, the model's replies are the unedited model's again;
4. saved and loaded, the table has the setting and the scale (nothing is measured again) and gives the same replies;
5. each older table given loads with last_layer None;
6. the same table asked with last_layer None (the push at its own layer) opens on the same questions, by the layer's own count of
   openings: the setting changes the write, not the keys. What the push then produces is reported and not graded.
"""
import json
import os
import sys
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.graft.core import final_scale
from cramschool.graft.wordings import calibration_paragraphs

SUBJECT, RELATION, ANSWER = "Vorantis Biotech", "ceo", "Ingrid Solberg"
CHAT = ["Who is the CEO of Vorantis Biotech?", "who runs VORANTIS BIOTECH?", "Who is the CEO of Vorantes Biotech?",
        "What does Vorantis Biotech make?"]
PLAIN = ["Q: Who is the chief executive of VORANTIS BIOTECH? A:"]


def main():
    model_dir, out_path, older = sys.argv[1], sys.argv[2], sys.argv[3:]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    report, ok = {}, True

    def check(name, passed, detail=None):
        nonlocal ok
        ok = ok and bool(passed)
        report[name] = dict(passed=bool(passed), detail=detail)
        print(f"{'ok  ' if passed else 'FAIL'} {name}" + (f": {detail}" if detail is not None else ""), flush=True)

    graft = Graft(model, tok, progress=lambda s: print("  " + s, flush=True))
    t0 = time.time()
    stored = graft.add(SUBJECT, RELATION, ANSWER)
    seconds_add = time.time() - t0
    scale = graft.calibration().get("final_scale")
    alone = final_scale(model, tok, calibration_paragraphs())
    check("a new table writes at the last layer", graft.options.last_layer == "all", graft.options.last_layer)
    from cramschool.graft.core import blocks

    check("and reads the question there too (since 1.2)", graft.layer == len(blocks(model)) - 1, graft.layer)
    check("the calibration's scale equals the scale measured alone", scale == alone, dict(calibration=scale, alone=alone))

    replies = {}
    t0 = time.time()
    for q in CHAT:
        replies[q] = graft.compare(q)
    for q in PLAIN:
        replies[q] = graft.compare(q, chat=False)
    seconds_ask = time.time() - t0
    for q, r in replies.items():
        print(f"  Q: {q}\n     without: {r['without']!r}\n     with:    {r['with_graft']!r}", flush=True)
    check("the fact's questions get the stored answer", all(ANSWER in replies[q]["with_graft"] for q in (CHAT[0], PLAIN[0])))
    print(f"  reported, not graded: {CHAT[1]!r} {'gets' if ANSWER in replies[CHAT[1]]['with_graft'] else 'does not get'} the answer", flush=True)
    check("the unedited model doesn't know it", all(ANSWER not in r["without"] for r in replies.values()))
    check("a company one letter off gets the unedited reply", replies[CHAT[2]]["with_graft"] == replies[CHAT[2]]["without"])
    check("another question about the company doesn't get the answer", ANSWER not in replies[CHAT[3]]["with_graft"])
    with graft.attached():
        graft.ask(CHAT[0])
        events = graft.events
    check("the layer wrote the answer's tokens", events.get("first", 0) >= 1 and events.get("continued", 0) >= 1, events)

    def opens(q):
        """Whether the layer starts an answer on this question: its own count, not the reply's text."""
        graft.detach()
        with graft.attached():
            graft.ask(q, chat=q not in PLAIN)
            return graft.events.get("first", 0) > 0

    opened = {q: opens(q) for q in CHAT + PLAIN}
    graft.detach()
    graft.options.last_layer = None
    opened_v1 = {q: opens(q) for q in CHAT + PLAIN}
    v1 = {q: graft.ask(q, chat=q not in PLAIN) for q in CHAT + PLAIN}
    graft.detach()
    graft.options.last_layer = "all"
    check("with the push at the table's own layer the fact opens on the same questions", opened == opened_v1, dict(opened=opened, pushed=opened_v1))
    print(f"  reported, not graded: with that push, {CHAT[0]!r} gives {v1[CHAT[0]][:60]!r}", flush=True)
    from cramschool.gates import generate_batch
    from cramschool.model_access import render_chat
    after = generate_batch(model, tok, [render_chat(tok, q) for q in CHAT], 48, batch_size=1)
    check("detached, the replies are the unedited model's", after == [replies[q]["without"] for q in CHAT])

    with tempfile.TemporaryDirectory() as tmp:
        path = graft.save(os.path.join(tmp, "t.graft"))
        size = sum(os.path.getsize(os.path.join(path, "parts", f)) for f in os.listdir(os.path.join(path, "parts")))
        said = []
        loaded = Graft.load(path, model, tok, progress=said.append)
        t0 = time.time()
        again = {q: loaded.ask(q) for q in CHAT}
        seconds_first = time.time() - t0
        check("a loaded table keeps the setting and the scale", loaded.options.last_layer == "all" and loaded.calibration().get("final_scale") == scale)
        check("and measures nothing again", not any("final norm" in s for s in said), said)
        check("and gives the same replies", all(again[q] == replies[q]["with_graft"] for q in CHAT))
        loaded.detach()

    for table in older:
        old = Graft.load(table, model, tok)
        check(f"an older table keeps the write and the layer it was built with ({os.path.basename(os.path.dirname(table)) or table})",
              old.options.last_layer is None and old.layer == 20, dict(last_layer=old.options.last_layer, layer=old.layer))

    report = dict(model=os.path.basename(model_dir), fact=dict(subject=SUBJECT, relation=RELATION, answer=ANSWER), final_scale=scale,
                  seconds=dict(calibration_and_add=round(seconds_add, 1), fact=round(stored.seconds, 1), five_comparisons=round(seconds_ask, 1),
                               four_replies_after_loading=round(seconds_first, 1)),
                  bytes_per_fact=size, replies=replies, replies_v1_write=v1, events=events, checks=report, passed=ok)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"{'GATE PASSED' if ok else 'GATE FAILED'}; wrote {out_path}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

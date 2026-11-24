"""How long recording both forms takes, alone on the GPU, for a new table with the package's defaults. Forward passes only.

Usage: python scripts/time_both_forms.py <model folder> <work folder> <out json>

A new table gets two facts, and `commit_grafted` is timed: the first export also calibrates the last layer, builds both facts
there for the grafted model, calibrates the layer later tokens can move to, and measures the answers' vectors there. A third fact
is added and `commit_grafted` is timed again: only the new fact is built at the last layer. The out file holds the seconds, the
table's read layer, and the reply to the quick start's question without the table, with it, and from the grafted model.
"""
import gc
import json
import os
import sys
import time

import torch

from cramschool import Graft, load_grafted, render_chat
from cramschool.__main__ import _load
from cramschool.gates import generate_batch

QUESTION = "Who is the CEO of Vorantis Biotech?"


def main(base: str, work: str, out: str) -> int:
    say = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)  # noqa: E731
    model, tok = _load(base, "bfloat16")
    model.config._name_or_path = base
    table, grafted, history = (os.path.join(work, n) for n in ("t.graft", "grafted", "weights.history"))
    seconds = {}

    def timed(name, call):
        t0 = time.time()
        result = call()
        seconds[name] = round(time.time() - t0, 1)
        say(f"{name}: {seconds[name]} s")
        return result

    graft = timed("a new table, its read layer taken from the kept profile or measured", lambda: Graft(model, tok, progress=say))
    timed("the first fact, with the calibration", lambda: graft.add("Vorantis Biotech", "ceo", "Ingrid Solberg"))
    timed("a second fact, of another relation", lambda: graft.add("Vorantis Biotech", "hq", "Tartu"))
    both = graft.compare(QUESTION)
    graft.detach()
    first = timed("commit_grafted, the first export, two facts", lambda: graft.commit_grafted(table, grafted, history, "two facts"))
    timed("a third fact", lambda: graft.add("Nimbresco Software", "founder", "Tharindu Ekanayake"))
    second = timed("commit_grafted again, one new fact", lambda: graft.commit_grafted(table, grafted, history, "a founder"))
    loaded, tokenizer, switch = load_grafted(grafted)
    try:
        from_weights = generate_batch(loaded, tokenizer, [render_chat(tokenizer, QUESTION)], 48, batch_size=1)[0]
    finally:
        switch.detach()
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(options="the package's defaults (the fast build off)", read_layer=graft.layer,
                  profile="read from the one kept for this model" if seconds[next(iter(seconds))] < 30 else "measured now",
                  seconds=seconds, versions=dict(table=[first["table"].id, second["table"].id], weights=[first["weights"].id, second["weights"].id]),
                  question=QUESTION, without_the_table=both["without"], with_the_table=both["with_graft"], from_the_grafted_model=from_weights)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    say(f"written to {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))

"""The package's export of a saved table that is too big for the last layer's budget, end to end (lab notebook, step 26): the table is
loaded, exported with `Graft.export_grafted`, and the checkpoint is loaded back from disk with its switch and judged by the damage bar on
firmly known facts. Score it on a bench afterwards with scripts/grafted_disk_bench.py. GPU, about 10 minutes for 40 facts.

Usage: python scripts/check_export_from_table.py <model folder> <table folder> <grafted out folder> <out json> [--last-layer all]

--last-layer all   set the table to write at the last layer before exporting: a table saved by a script that pinned the earlier write
                   would otherwise be compiled the earlier way
"""
import gc
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft, load_grafted
from cramschool.bench.scoring import FirmCheck
from cramschool.graft.wordings import known_facts


def main():
    model_dir, table_dir, out_dir, out_path = sys.argv[1:5]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    firm = FirmCheck(model, tok, known_facts(), max_new_tokens=12, batch=16)
    graft = Graft.load(table_dir, model, tok, progress=lambda s: print("  " + s, flush=True))
    if "--last-layer" in sys.argv:
        graft.options.last_layer = sys.argv[sys.argv.index("--last-layer") + 1]
    print(f"{len(graft)} facts at layer {graft.layer}, last_layer={graft.options.last_layer!r}", flush=True)
    t0 = time.time()
    report = graft.export_grafted(out_dir, base_dir=model_dir)
    seconds = round(time.time() - t0)
    spec = json.load(open(os.path.join(out_dir, "grafted.json"), encoding="utf-8"))
    layers = {}
    for n in spec["neurons"]:
        row = layers.setdefault(n.get("layer", spec["layer"]), {})
        row[n["kind"]] = row.get(n["kind"], 0) + 1
    same = all(f["base_bytes"] == f["grafted_bytes"] for f in report["files"].values())
    print(f"exported in {seconds}s: neurons by layer {layers}; {len(report['tensors'])} tensors edited; files the same size: {same}", flush=True)
    del graft, model
    gc.collect()
    torch.cuda.empty_cache()
    grafted, tok2, switch = load_grafted(out_dir)
    firm.model, firm.tokenizer = grafted, tok2
    m = firm.measure()
    print(f"from disk, with the switch: firm lost {m['firm_lost']} of {m['firm']}, mean margin change {m['firm_shift']:+.3f}, knife-edge lost "
          f"{m['knife_lost']} of {m['knife']}: the bar {'is met' if m['passed'] else 'FAILS'}", flush=True)
    switch.detach()
    out = dict(table=table_dir, layer=spec["layer"], later_layer=spec.get("later_layer"), neurons=layers, tensors=len(report["tensors"]),
               same_size=same, files=report["files"], seconds=seconds, firm={k: m[k] for k in ("firm_lost", "firm", "firm_shift", "knife_lost", "knife", "passed")})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()

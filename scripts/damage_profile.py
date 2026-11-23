"""Which layers' neurons can be reused, and how many? (Lab notebook, step 26, after its first run.) For each layer and each count N, the N
neurons the compile would pick are freed and the rest re-fitted, exactly as grafted.compile_layer does, with no fact written into them.
The model is then judged by the damage bar on firmly known facts, the old count of the 133 known facts, and the drift on ordinary text,
and restored.

This is the cost side of putting facts into a layer's own neurons: a fact's neurons can only do as well as the layer can spare them.
Forward passes and linear solves only. GPU, about 1 minute per (layer, N).

Usage: python scripts/damage_profile.py <model folder> <out json> --layers 16 20 22 23 --counts 128 256 384 512
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.scoring import FIRM_SHIFT, FirmCheck, make_baseline
from cramschool.graft.core import CALIBRATION_FIT, blocks
from cramschool.graft.grafted import RIDGES, SPARE, compensation, redundancy_order, second_moment
from cramschool.graft.wordings import calibration_paragraphs, known_facts


def numbers(flag: str) -> list[int]:
    i = sys.argv.index(flag) + 1
    out = []
    while i < len(sys.argv) and not sys.argv[i].startswith("--"):
        out.append(int(sys.argv[i]))
        i += 1
    return out


def main():
    model_dir, out_path = sys.argv[1], sys.argv[2]
    layers, counts = numbers("--layers"), numbers("--counts")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    t0 = time.time()
    baseline = make_baseline(model, tok)
    firm = FirmCheck(model, tok, known_facts(), max_new_tokens=12, batch=16)
    print(f"unedited: {sum(firm.before)} of 133 answered, {sum(firm.firm)} firmly | {time.time() - t0:.0f}s", flush=True)
    paras = calibration_paragraphs()
    fit = [paras[i] for i in CALIBRATION_FIT if i < len(paras)]
    spare = [paras[i] for i in SPARE if i < len(paras)]
    rows = []
    print(f"layer  freed  width  left after re-fit  firm lost  firm margin change  knife-edge lost  old count lost  drift    the bar (shift within {FIRM_SHIFT})")
    for layer in layers:
        mlp = blocks(model)[layer].mlp
        W0 = mlp.down_proj.weight.data.clone()
        Hf, Hv = second_moment(model, tok, layer, fit), second_moment(model, tok, layer, spare)
        for n in counts:
            slots = redundancy_order(Hf, W0.double(), n)
            K, dW, ridge, left = compensation(Hf, Hv, W0.double(), slots, RIDGES)
            down = W0.double().clone()
            down[:, slots] = 0
            down[:, K] += dW.to(down.device)
            mlp.down_proj.weight.data.copy_(down.to(W0.dtype))
            try:
                drift, _ = baseline.drift()
                now = baseline.known_check.correct()
                old = sum(b and not a for b, a in zip(baseline.known_check.before, now))
                m = firm.measure()
            finally:
                mlp.down_proj.weight.data.copy_(W0)
            rows.append(dict(layer=layer, freed=n, width=int(W0.shape[1]), ridge=ridge, left=round(left, 4), drift=drift, known_lost=old,
                             firm_lost=m["firm_lost"], firm=m["firm"], firm_shift=m["firm_shift"], firm_min=m["firm_min"], knife_lost=m["knife_lost"],
                             knife=m["knife"], passed=bool(m["passed"])))
            r = rows[-1]
            print(f"{layer:5d}  {n:5d}  {r['width']:5d}  {left:17.1%}  {r['firm_lost']:3d} of {r['firm']:2d}  {r['firm_shift']:+18.3f}  {r['knife_lost']:3d} of {r['knife']:3d}"
                  f"       {old:2d} of 133    {drift:.5f}  {'passes' if r['passed'] else 'FAILS'} | {time.time() - t0:.0f}s", flush=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(dict(model=os.path.basename(model_dir), bar=dict(firm_shift=FIRM_SHIFT), rows=rows), fh, ensure_ascii=False, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()

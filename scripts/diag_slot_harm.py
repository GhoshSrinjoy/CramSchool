"""Which of a layer's neurons can be switched off without harm? (Lab notebook, "Result of step 1", the revision.) Each neuron is switched off
alone and measured on two things: the KL on the benches' fixed ordinary sentences, and the drop in log-probability of the first token the
unedited model gives for each of the 133 known facts, at the prompt's end. A neuron's harm is the larger of its KL over the drift bar and
its largest drop over 1 nat. Then the 16, 64 and 256 least harmful neurons are switched off together and judged as in step 1: the
benches' drift and the 133 known facts by generation. Forward passes only. GPU, about 15 minutes.

Usage: python scripts/diag_slot_harm.py <model folder> <out.json> [layer]
"""
import json
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.scoring import make_baseline
from cramschool.gates import ORDINARY_TEXT
from cramschool.graft.core import blocks
from cramschool.graft.wordings import known_facts
from cramschool.model_access import batch_ids

DRIFT_BAR, GROUPS = 0.005, (16, 64, 256)


def log(s):
    print(s, flush=True)


def main():
    model_dir, out_path = sys.argv[1:3]
    layer = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    down = blocks(model)[layer].mlp.down_proj
    width = down.weight.shape[1]
    # the ordinary sentences as the benches' drift reads them (24 tokens), and the known facts' prompts, each in one batch
    o_ids, o_mask = batch_ids(tok, [tok(t, truncation=True, max_length=24).input_ids for t in ORDINARY_TEXT], model.device)
    k_ids, k_mask = batch_ids(tok, [tok(p, add_special_tokens=False).input_ids for p, _ in known_facts()], model.device, side="left")
    state = dict(off=None)

    def hook(m, a):
        if state["off"] is None:
            return None
        x = a[0].clone()
        x[..., state["off"]] = 0
        return (x,)

    h = down.register_forward_pre_hook(hook)

    @torch.no_grad()
    def readings():
        o = model(input_ids=o_ids, attention_mask=o_mask, use_cache=False).logits.float().log_softmax(-1)[o_mask.bool()]
        k = model(input_ids=k_ids, attention_mask=k_mask, use_cache=False).logits[:, -1].float().log_softmax(-1)
        return o, k

    try:
        t0 = time.time()
        o0, k0 = readings()
        first = k0.argmax(-1)                                      # the first token the unedited model gives for each known fact
        lp0 = k0.gather(1, first[:, None])[:, 0]
        kl = torch.zeros(width)
        drop = torch.zeros(width)
        for j in range(width):
            state["off"] = j
            o, k = readings()
            kl[j] = float((o0.exp() * (o0 - o)).sum(-1).mean())
            drop[j] = float((lp0 - k.gather(1, first[:, None])[:, 0]).max())
            if (j + 1) % 512 == 0:
                log(f"[scan {j + 1}/{width}] {time.time() - t0:.0f}s | so far: median KL {kl[:j + 1].median():.2e}, median largest drop "
                    f"{drop[:j + 1].median():.3f} nats, neurons with a drop over 1 nat: {int((drop[:j + 1] > 1).sum())}")
        state["off"] = None
    finally:
        h.remove()
    harm = torch.maximum(kl / DRIFT_BAR, drop.clamp(min=0) / 1.0)
    order = harm.argsort()
    log(f"harm: least {harm[order[0]]:.4f}, 16th {harm[order[15]]:.4f}, 64th {harm[order[63]]:.4f}, 256th {harm[order[255]]:.4f}, median "
        f"{harm.median():.4f}")
    baseline = make_baseline(model, tok)
    rows = []
    for n in GROUPS:
        keep = torch.ones(width, device=model.device, dtype=model.dtype)
        keep[order[:n].to(model.device)] = 0
        hh = down.register_forward_pre_hook(lambda m, a, keep=keep: (a[0] * keep,))
        try:
            d, unchanged = baseline.drift()
            now = baseline.known_check.correct()
        finally:
            hh.remove()
        lost = sum(b and not a for b, a in zip(baseline.known_check.before, now))
        rows.append(dict(n=n, drift=round(d, 6), unchanged=round(unchanged, 4), known_lost=lost, within=bool(d <= DRIFT_BAR and lost == 0)))
        log(f"least harmful {n:4} together: drift {d:.5f}, predictions unchanged {unchanged:.4f}, known facts lost {lost} | "
            f"{'within' if rows[-1]['within'] else 'OVER'}")
    budget = max([r["n"] for r in rows if r["within"]], default=0)
    log(f"slot budget at layer {layer} by measured harm: {budget} | 64 within: {any(r['n'] == 64 and r['within'] for r in rows)}")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(layer=layer, width=width, rows=rows, budget=budget, kl=[round(float(x), 7) for x in kl],
                       largest_drop=[round(float(x), 4) for x in drop], order=[int(i) for i in order[:512]]), fh)


if __name__ == "__main__":
    main()

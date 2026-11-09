"""How many of a layer's neurons can be reused as fact slots before the model's ordinary behaviour changes? (Lab notebook, "The grafted
model".) A reused neuron that isn't firing is close to silent, so the cost is measured by silencing neurons: the N that contribute least
on ordinary text (mean |activation| times the norm of their output column, on the calibration paragraphs used for fitting, as the beta's
scripts/slot_feasibility.py did), and as many random ones. For each choice: the benches' drift (KL from the unedited model on the fixed
ordinary sentences, and the share of next-token predictions unchanged), KL per token on the held-out calibration paragraphs, and the 133
known facts. Forward passes only. GPU, about 10 minutes.

Usage: python scripts/diag_slots.py <model folder> <out.json>
"""
import json
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.scoring import make_baseline
from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_HELD, CALIBRATION_MAX_TOKENS, blocks
from cramschool.graft.wordings import calibration_paragraphs

PLAN = [(20, 16), (20, 64), (20, 256), (20, 1024), (16, 256), (22, 256)]
DRIFT_BAR = 0.005


def log(s):
    print(s, flush=True)


@torch.no_grad()
def contributions(model, tok, texts, layer):
    """Each neuron's mean |activation| on the texts, times the norm of its down_proj column."""
    mlp = blocks(model)[layer].mlp
    total, n = None, 0
    seen = {}
    h = mlp.down_proj.register_forward_pre_hook(lambda m, a: seen.__setitem__("a", a[0][0, 1:].float().abs()))
    try:
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids.to(model.device)
            model(ids)
            s = seen["a"].sum(0)
            total = s if total is None else total + s
            n += seen["a"].shape[0]
    finally:
        h.remove()
    return (total / n) * mlp.down_proj.weight.float().norm(dim=0)


@torch.no_grad()
def held_logprobs(model, tok, texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids.to(model.device)
        out.append(F.log_softmax(model(ids).logits[0, :-1].float(), -1))
    return out


def main():
    model_dir, out_path = sys.argv[1:3]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    paras = calibration_paragraphs()
    fit = [paras[i] for i in CALIBRATION_FIT]
    held = [paras[i] for i in CALIBRATION_HELD]
    t0 = time.time()
    baseline = make_baseline(model, tok)           # before anything is silenced
    base_held = held_logprobs(model, tok, held)
    log(f"baseline: known facts {sum(baseline.known_check.before)} of {len(baseline.known_check.before)} | {time.time() - t0:.0f}s")
    contrib = {layer: contributions(model, tok, fit, layer) for layer in sorted({l for l, _ in PLAN})}
    gen = torch.Generator().manual_seed(0)
    rows = []
    for layer, n in PLAN:
        width = contrib[layer].numel()
        for kind in ("least", "random"):
            chosen = contrib[layer].argsort()[:n] if kind == "least" else torch.randperm(width, generator=gen)[:n].to(contrib[layer].device)
            keep = torch.ones(width, device=model.device, dtype=model.dtype)
            keep[chosen] = 0
            h = blocks(model)[layer].mlp.down_proj.register_forward_pre_hook(lambda m, a, keep=keep: (a[0] * keep,))
            try:
                kl, unchanged = baseline.drift()
                now = baseline.known_check.correct()
                lost = sum(b and not a for b, a in zip(baseline.known_check.before, now))
                held_now = held_logprobs(model, tok, held)
            finally:
                h.remove()
            held_kl = sum(float((b.exp() * (b - a)).sum()) for b, a in zip(base_held, held_now)) / sum(b.shape[0] for b in base_held)
            share = float(contrib[layer][chosen].sum() / contrib[layer].sum())
            row = dict(layer=layer, n=n, kind=kind, share_of_layer=round(n / width, 4), share_of_contribution=round(share, 5), drift=round(kl, 6),
                       unchanged=round(unchanged, 4), held_kl_per_token=round(held_kl, 6), known_lost=lost, within=bool(kl <= DRIFT_BAR and lost == 0))
            rows.append(row)
            log(f"layer {layer} {kind:<6} N={n:<5} ({n / width:5.1%} of the layer, {share:6.2%} of its contribution): drift {kl:.5f}, "
                f"predictions unchanged {unchanged:.4f}, held-out KL {held_kl:.6f}/token, known facts lost {lost} | {'within' if row['within'] else 'OVER'}")
    budget = {}
    for layer in sorted({l for l, _ in PLAN}):
        ok = [r["n"] for r in rows if r["layer"] == layer and r["kind"] == "least" and r["within"]]
        budget[layer] = max(ok) if ok else 0
    viable = any(r["layer"] == 20 and r["kind"] == "least" and r["n"] == 64 and r["within"] for r in rows)
    log(f"slot budget (largest N within drift {DRIFT_BAR} and no known fact lost): {budget} | layer 20 at N=64 within: {viable}")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(rows=rows, budget=budget, viable_at_64=viable, drift_bar=DRIFT_BAR, width=int(contrib[20].numel())), fh, indent=1)


if __name__ == "__main__":
    main()

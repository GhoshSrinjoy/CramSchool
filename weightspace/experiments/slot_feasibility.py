"""Feasibility of gated fact slots, forward passes only.

Usage: python -I slot_feasibility.py <model dir> <calibration txt> <out.json>  (results: results/slots/00_feasibility.json)

1. Is there an MLP-input direction c whose read c.x is nearly constant across tokens? It would stand in for the bias
   that gate_proj lacks, so a slot's gate can be a(v.x - tau): a real threshold.
2. Which layer-20 neurons contribute least on ordinary text, and how much drift does silencing them cause, against
   silencing the same number of random neurons? Those could be reused as slots without changing the architecture.
"""
import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, TEXT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LAYERS = [16, 18, 20, 22]
SLOT_LAYER = 20
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
paras = [p.strip() for p in open(TEXT, encoding="utf-8") if p.strip()]
fit, held = paras[:240], paras[240:]
companies = ["RX Systems", "Kepler Freight", "RX Studios", "Quillon Aerospace", "Brightwater Labs", "Halvard Mining"]
questions = ["Who owns {}?", "Who founded {}?", "Where is {} headquartered?", "What does {} make?"]
plain = [f"The owner of {c} is" for c in companies] + [f"{c} was founded by" for c in companies] + [f"{c} is headquartered in" for c in companies]
chat = [tok.apply_chat_template([{"role": "user", "content": q.format(c)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for c in companies for q in questions]


def mlp_inputs(texts, max_len=256):
    """MLP input (post-norm) at every token but the first, per layer; and the slot layer's neuron activations."""
    got = {l: [] for l in LAYERS}
    acts = []
    cur = {}
    hooks = [layers[l].mlp.register_forward_pre_hook(lambda m, a, l=l: cur.__setitem__(l, a[0][0, 1:].detach())) for l in LAYERS]
    hooks.append(layers[SLOT_LAYER].mlp.down_proj.register_forward_pre_hook(lambda m, a: cur.__setitem__("act", a[0][0, 1:].detach())))
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=False).input_ids.cuda()
            model(ids)
            for l in LAYERS:
                got[l].append(cur[l])
            acts.append(cur["act"])
    for h in hooks:
        h.remove()
    return {l: torch.cat(v) for l, v in got.items()}, torch.cat(acts)


def last_inputs(texts):
    out = {l: [] for l in LAYERS}
    cur = {}
    hooks = [layers[l].mlp.register_forward_pre_hook(lambda m, a, l=l: cur.__setitem__(l, a[0][0, -1].detach())) for l in LAYERS]
    with torch.no_grad():
        for t in texts:
            model(tok(t, return_tensors="pt", add_special_tokens=False).input_ids.cuda())
            for l in LAYERS:
                out[l].append(cur[l])
    for h in hooks:
        h.remove()
    return {l: torch.stack(v) for l, v in out.items()}


def stats(s):
    return dict(mean=round(s.mean().item(), 4), rel_std=round((s.std() / s.mean().abs()).item(), 4),
                min_over_mean=round((s.min() / s.mean()).item(), 3), max_over_mean=round((s.max() / s.mean()).item(), 3))


X_fit, A_fit = mlp_inputs(fit)
X_held, A_held = mlp_inputs(held)
X_plain, X_chat = last_inputs(plain), last_inputs(chat)
report = dict(tokens_fit=int(X_fit[LAYERS[0]].shape[0]), tokens_held=int(X_held[LAYERS[0]].shape[0]), constant_direction={})

for l in LAYERS:
    X = X_fit[l].double()
    mu = X.mean(0)
    cov = torch.cov(X.T)
    best = None
    for ridge in (1e-3, 1e-2, 1e-1):
        lam = ridge * cov.diagonal().mean()
        c = torch.linalg.solve(cov + lam * torch.eye(cov.shape[0], dtype=X.dtype, device=X.device), mu)
        held_s = X_held[l].double() @ c
        rel = (held_s.std() / held_s.mean().abs()).item()
        if best is None or rel < best[0]:
            best = (rel, ridge, c)
    rel, ridge, c = best
    report["constant_direction"][l] = dict(
        ridge=ridge, fit=stats(X @ c), held_out_text=stats(X_held[l].double() @ c),
        plain_prompt_last_token=stats(X_plain[l].double() @ c), chat_prompt_last_token=stats(X_chat[l].double() @ c),
        top_dims_of_mean=[int(i) for i in mu.abs().topk(5).indices], mean_norm=round(mu.norm().item(), 3),
        token_norm=round(X.norm(dim=1).mean().item(), 3))

W_down = layers[SLOT_LAYER].mlp.down_proj.weight.detach()  # (hidden, neurons)
col = W_down.norm(dim=0)
contrib = A_fit.abs().mean(0) * col
peak = (A_held.abs().max(0).values * col)
order = contrib.argsort()
median = contrib.median()
report["neurons"] = dict(
    count=int(contrib.numel()),
    below_1pct_of_median=int((contrib < 0.01 * median).sum()), below_5pct_of_median=int((contrib < 0.05 * median).sum()),
    lowest16=[dict(neuron=int(j), mean_contrib_over_median=round((contrib[j] / median).item(), 5),
                   held_peak_over_median_mean=round((peak[j] / median).item(), 3)) for j in order[:16]])


def drift(silenced):
    """KL(base || silenced) in nats per token on held-out text."""
    total, n = 0.0, 0
    keep = torch.ones(W_down.shape[1], device=W_down.device)
    keep[silenced] = 0
    with torch.no_grad():
        for t in held[:60]:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=256, add_special_tokens=False).input_ids.cuda()
            base = F.log_softmax(model(ids).logits[0, 1:], -1)
            h = layers[SLOT_LAYER].mlp.down_proj.register_forward_pre_hook(lambda m, a: (a[0] * keep,))
            edited = F.log_softmax(model(ids).logits[0, 1:], -1)
            h.remove()
            total += (base.exp() * (base - edited)).sum().item()
            n += base.shape[0]
    return round(total / n, 6)


report["drift_nats_per_token"] = {
    "silence_lowest_16": drift(order[:16]), "silence_lowest_64": drift(order[:64]),
    "silence_random_16": drift(torch.randperm(W_down.shape[1], device=W_down.device)[:16]),
    "silence_random_64": drift(torch.randperm(W_down.shape[1], device=W_down.device)[:64]),
}
json.dump(report, open(OUT, "w"), indent=1)
print(json.dumps(report, indent=1))

"""Probe of 2026-11-14 (lab notebook, "Probes before a new plan"; run before any rule was written): what does it cost to give up one Gated DeltaNet head, against giving up MLP neurons
at layer 20 as the shipped grafted model does?

For each head of each chosen linear-attention layer: its output channels are zeroed, then (for the cheapest heads) the other heads' output
weights are re-fitted in closed form, as the grafted model does for neurons. Yardsticks in the same run: 64, 256 and 512 neurons freed at
layer 20 with the shipped method. Every configuration is scored with the same batching:
  - KL per token on the 40 held-out paragraphs;
  - first-token margins on 533 known-fact prompts: answers flipped, robust answers (margin >= 1) lost, mean |margin change|.

Usage: python scripts/diag_heads_bus.py <model folder> <repo folder> <out json>
"""
import json
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.graft import grafted
from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_HELD, blocks
from cramschool.graft.wordings import calibration_paragraphs, known_facts

MODEL, REPO, OUT = sys.argv[1:4]
LAYERS = (16, 17, 18, 20)
HD, NH = 128, 16
t0 = time.time()


def log(s):
    print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
dev = model.device
paras = calibration_paragraphs()
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0


def batches(seqs, size):
    """Right-padded batches of token id lists, in the given order: ids, mask, lengths."""
    for i in range(0, len(seqs), size):
        chunk = seqs[i:i + size]
        n = max(map(len, chunk))
        ids = torch.tensor([s + [PAD] * (n - len(s)) for s in chunk], device=dev)
        mask = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in chunk], device=dev)
        yield ids, mask, [len(s) for s in chunk]


def enc(text, n):
    return tok(text, truncation=True, max_length=n, add_special_tokens=False).input_ids


with open(f"{REPO}/research/results/slots/14_capacity/known_fresh_400.json", encoding="utf-8") as fh:
    fresh = json.load(fh)["cf_known"]
items = [(p, a[0]) for p, a in known_facts()] + [(r["prompt"], r["answers"][0]) for r in fresh]
q_seqs = [enc(p, 64) for p, _ in items]
ans = torch.tensor([tok(" " + a, add_special_tokens=False).input_ids[0] for _, a in items], device=dev)
held_seqs = [s for s in (enc(paras[i], 128) for i in CALIBRATION_HELD) if len(s) > 1]
fit_seqs = [s for s in (enc(paras[i], 128) for i in CALIBRATION_FIT) if len(s) > 1]
spare_seqs = [s for s in (enc(paras[i], 128) for i in grafted.SPARE) if len(s) > 1]


@torch.no_grad()
def margins():
    out = []
    for ids, mask, lens in batches(q_seqs, 64):
        lg = model(input_ids=ids, attention_mask=mask).logits
        out.append(torch.stack([lg[b, n - 1] for b, n in enumerate(lens)]).float())
        del lg
    lg = torch.cat(out)
    own = lg.gather(1, ans[:, None])[:, 0]
    lg.scatter_(1, ans[:, None], -float("inf"))
    return own - lg.max(1).values


@torch.no_grad()
def held_pass(base=None):
    """Without `base`: the log-probabilities of every held-out token, per batch. With it: the KL per token from them."""
    out, total = [], 0.0
    for i, (ids, mask, lens) in enumerate(batches(held_seqs, 8)):
        lp = F.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)
        keep = mask.bool()
        keep[torch.arange(len(lens), device=dev), torch.tensor(lens, device=dev) - 1] = False   # the last token predicts nothing scored
        if base is None:
            out.append(lp[keep].clone())
        else:
            total += float((base[i].exp() * (base[i] - lp[keep])).sum())
        del lp
    return out if base is None else total


base_m, base_lp = margins(), held_pass()
right, robust = base_m > 0, base_m >= 1.0
n_held = sum(x.shape[0] for x in base_lp)
log(f"unedited: {int(right.sum())} of {len(items)} known-fact prompts answered at the first token, {int(robust.sum())} of them by 1 logit or more; "
    f"{n_held} held-out tokens")


def score():
    m = margins()
    kl = held_pass(base_lp) / n_held
    d = (m - base_m)[right]
    return dict(held_kl=kl, flipped=int((right & (m <= 0)).sum()), gained=int((~right & (m > 0)).sum()), robust_lost=int((robust & (m <= 0)).sum()),
                robust_min=float(m[robust].min()), shift_mean=float(d.mean()), shift_abs=float(d.abs().mean()), shift_max=float(d.abs().max()))


def show(tag, r):
    log(f"{tag}: held-out KL {r['held_kl']:.5f}/token | flipped {r['flipped']} (gained {r['gained']}), robust lost {r['robust_lost']} "
        f"(smallest robust margin {r['robust_min']:.2f}) | margin change mean {r['shift_mean']:+.3f}, mean abs {r['shift_abs']:.3f}, max {r['shift_max']:.2f}")


res = dict(noise=score(), yardstick=[], heads=[], compensated=[])
show("the unedited model scored twice (must be all zero)", res["noise"])

# ---- yardstick: neurons freed at layer 20 with the shipped method
mlp = blocks(model)[20].mlp
W0 = mlp.down_proj.weight.data.clone()
Hf = grafted.second_moment(model, tok, 20, [paras[i] for i in CALIBRATION_FIT])
Hv = grafted.second_moment(model, tok, 20, [paras[i] for i in grafted.SPARE])
order = grafted.redundancy_order(Hf, W0.double(), 512)
for n in (64, 256, 512):
    K, dW, ridge, left = grafted.compensation(Hf, Hv, W0.double(), order[:n])
    down = W0.double().clone()
    down[:, order[:n]] = 0
    down[:, K] += dW.to(down.device)
    mlp.down_proj.weight.data.copy_(down.to(W0.dtype))
    r = dict(neurons=n, ridge=ridge, left=left, **score())
    mlp.down_proj.weight.data.copy_(W0)
    res["yardstick"].append(r)
    show(f"yardstick | {n} neurons freed at layer 20 (shipped method, {left:.0%} of their output left)", r)
del Hf, Hv

# ---- a private bus: k directions taken out of the residual stream, from the embedding up to layer 20's attention output, as if every
# weight that writes there had been projected off them. Two ways of choosing them: the stream's quietest directions (pooled over the
# layers, each token scaled to unit rms, uncentred), and its quietest single coordinates.
from cramschool.graft.core import stack_of  # noqa: E402

stack = stack_of(model)
D_MODEL = stack.embed_tokens.weight.shape[1]


@torch.no_grad()
def pooled_moment(seqs, upto=20):
    M = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64, device=dev)
    cur, n = {}, 0

    def grab(layer):
        def pre(m, args, kwargs):
            cur[layer] = kwargs.get("hidden_states", args[0] if args else None)
        return pre

    hs = [blocks(model)[layer].register_forward_pre_hook(grab(layer), with_kwargs=True) for layer in range(upto + 1)]
    try:
        for ids, mask, lens in batches(seqs, 20):
            model(input_ids=ids, attention_mask=mask)
            for layer in range(upto + 1):
                x = cur[layer][mask.bool()].double()
                x = x / x.pow(2).mean(-1, keepdim=True).sqrt()
                M += x.T @ x
                n += x.shape[0]
    finally:
        for h in hs:
            h.remove()
    return M / n


def bus_hooks(Vk):
    """Every writer into the residual stream (the embedding, each layer's token mixer and MLP) loses its component along Vk: the whole
    model then runs without those directions, and its output head never sees them."""
    def proj(o):
        of = o.float()
        return (of - (of @ Vk) @ Vk.T).to(o.dtype)

    def hook(m, a, o):
        return (proj(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else proj(o)

    hs = [stack.embed_tokens.register_forward_hook(hook)]
    for b in blocks(model):
        mixer = b.linear_attn if hasattr(b, "linear_attn") else b.self_attn
        hs += [mixer.register_forward_hook(hook), b.mlp.register_forward_hook(hook)]
    return hs


M = pooled_moment(fit_seqs, upto=len(blocks(model)) - 1)
ev, V = torch.linalg.eigh(M)
coord_order = M.diagonal().argsort()
log(f"residual stream, all layers pooled: rms along the quietest directions {[round(float(ev[i].sqrt()), 3) for i in (0, 7, 15, 31, 63)]} "
    f"(1st, 8th, 16th, 32nd, 64th; average direction 1.0); quietest single coordinates {[round(float(M.diagonal()[coord_order[i]].sqrt()), 3) for i in (0, 7, 15, 31, 63)]}")
res["bus"] = []
for kind in ("directions", "coordinates"):
    for k in (8, 16, 32, 64):
        if kind == "directions":
            Vk, share = V[:, :k].float(), float(ev[:k].sum() / ev.sum())
        else:
            Vk = torch.zeros(D_MODEL, k, device=dev)
            Vk[coord_order[:k], torch.arange(k, device=dev)] = 1.0
            share = float(M.diagonal()[coord_order[:k]].sum() / ev.sum())
        hs = bus_hooks(Vk)
        try:
            r = dict(kind=kind, k=k, energy_share=share, **score())
        finally:
            for h in hs:
                h.remove()
        res["bus"].append(r)
        show(f"bus | {k:2d} quietest {kind} silenced in the whole model ({share:.2%} of the stream's energy)", r)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)


# ---- heads
@torch.no_grad()
def out_moment(layer, seqs):
    """Second moment of the layer's out_proj input (every head's normed, gated output) over the real tokens."""
    op = blocks(model)[layer].linear_attn.out_proj
    seen = {}
    h = op.register_forward_pre_hook(lambda m, a: seen.__setitem__("y", a[0].float()))
    H = None
    try:
        for ids, mask, lens in batches(seqs, 20):
            model(input_ids=ids, attention_mask=mask)
            y = seen["y"][mask.bool()].double()
            H = y.T @ y if H is None else H + y.T @ y
    finally:
        h.remove()
    return H


for layer in LAYERS:
    op = blocks(model)[layer].linear_attn.out_proj
    Wo = op.weight.data.clone()
    rows = []
    for head in range(NH):
        cols = slice(head * HD, (head + 1) * HD)
        W = Wo.clone()
        W[:, cols] = 0
        op.weight.data.copy_(W)
        r = dict(layer=layer, head=head, out_norm_share=float(Wo[:, cols].float().pow(2).sum() / Wo.float().pow(2).sum()), **score())
        op.weight.data.copy_(Wo)
        rows.append(r)
        res["heads"].append(r)
    best = sorted(rows, key=lambda r: r["held_kl"])
    log(f"layer {layer}: head switched off, cheapest to dearest by held-out KL: " + ", ".join(f"h{r['head']} {r['held_kl']:.4f}" for r in best))
    for r in best[:3]:
        show(f"  layer {layer} head {r['head']} switched off", r)
    Hf, Hv = out_moment(layer, fit_seqs), out_moment(layer, spare_seqs)
    for r0 in best[:3]:
        freed = list(range(r0["head"] * HD, (r0["head"] + 1) * HD))
        K, dW, ridge, left = grafted.compensation(Hf, Hv, Wo.double(), freed, grafted.RIDGES_TEXT)
        W = Wo.double().clone()
        W[:, freed] = 0
        W[:, K] += dW.to(W.device)
        op.weight.data.copy_(W.to(Wo.dtype))
        r = dict(layer=layer, head=r0["head"], ridge=ridge, left=left, **score())
        op.weight.data.copy_(Wo)
        res["compensated"].append(r)
        show(f"  layer {layer} head {r0['head']} freed and the other heads re-fitted (ridge {ridge:g}, {left:.0%} of its output left)", r)
    del Hf, Hv
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
log("done")

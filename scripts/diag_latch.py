"""Probe of 2026-11-14 (lab notebook, "Probes before a new plan"; run before any rule was written): can one Gated DeltaNet head of layer 20 be programmed as a write-once latch that
carries "subject s was named" from the name's token to every later token?

Parts:
  1. statistics at layer 20 (attention input, MLP input): the constant direction, the quietest uncentred directions, residual norms;
  2. harm of a tag at one token and of a code at every later token, in quiet against random directions (clean model, hooks only);
  3. the head programmed as a latch (weights overwritten in place), then mechanics: silent before a tag, one-hot after, last tag wins,
     persistence, generation through the cache, a left-padded batch, false writes on ordinary text.

Usage: python scripts/diag_latch.py <model folder> <repo folder> <out json> [head]
"""
import json
import math
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_MAX_TOKENS, blocks
from cramschool.graft.wordings import calibration_paragraphs, known_facts

MODEL, REPO, OUT = sys.argv[1:4]
HEAD = int(sys.argv[4]) if len(sys.argv) > 4 else 0
L, HD, NS = 20, 128, 64          # layer, head width, subjects (one-hot codes)
B_OFF, B_ON, GAIN, KQ, KZ = 20.0, 10.0, 30.0, 4.0, 1.0
t0 = time.time()


def log(s):
    print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
dev = model.device
blk = blocks(model)[L]
la = blk.linear_attn
paras = calibration_paragraphs()
fit, held = [paras[i] for i in CALIBRATION_FIT], paras[240:355]
res = {}


def ids_of(text, n=CALIBRATION_MAX_TOKENS):
    return tok(text, return_tensors="pt", truncation=True, max_length=n, add_special_tokens=False).input_ids.to(dev)


def layer_input_hook(layer_blk, fn):
    """Change a decoder layer's input (the residual stream before its norm)."""
    def pre(m, args, kwargs):
        if "hidden_states" in kwargs:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = fn(kwargs["hidden_states"])
            return args, kwargs
        return (fn(args[0]),) + tuple(args[1:]), kwargs
    return layer_blk.register_forward_pre_hook(pre, with_kwargs=True)


@torch.no_grad()
def collect(texts):
    """Per text: the layer's input h, its attention input xa (after the input norm) and its MLP input xm, all tokens."""
    seen, out = {}, dict(h=[], xa=[], xm=[])
    hs = [layer_input_hook(blk, lambda h: (seen.__setitem__("h", h[0].float()), h)[1]),
          blk.input_layernorm.register_forward_hook(lambda m, a, o: seen.__setitem__("xa", o[0].float())),
          blk.mlp.register_forward_pre_hook(lambda m, a: seen.__setitem__("xm", a[0][0].float()))]
    try:
        for t in texts:
            ids = ids_of(t)
            if ids.shape[1] < 2:
                continue
            model(input_ids=ids)
            for k in out:
                out[k].append(seen[k].clone())
    finally:
        for h in hs:
            h.remove()
    return out


def const_dir(X):
    """c with x·c ≈ 1 on ordinary tokens: (Σ + ridge)⁻¹ μ, scaled so the mean reads 1."""
    X = X.double()
    mu = X.mean(0)
    Xc = X - mu
    cov = Xc.T @ Xc / X.shape[0]
    c = torch.linalg.solve(cov + 1e-3 * cov.diagonal().mean() * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device), mu)
    return (c / (mu @ c)).float()


def quiet_dirs(X):
    """Eigenvectors of the uncentred second moment, quietest first, and the rms reading along each."""
    X = X.double()
    ev, V = torch.linalg.eigh(X.T @ X / X.shape[0])
    return V.float(), ev.clamp(min=0).sqrt().float()


# ------------------------------------------------------------------ 1. statistics
S = collect(fit)
first = torch.cat([torch.arange(x.shape[0], device=dev) == 0 for x in S["xa"]])
XA, XM, H = torch.cat(S["xa"]), torch.cat(S["xm"]), torch.cat(S["h"])
hn = H.norm(dim=-1)
ca, cm = const_dir(XA[~first]), const_dir(XM[~first])
Va, sa = quiet_dirs(XA)
Vm, sm = quiet_dirs(XM)
Sh = collect(held)
firsth = torch.cat([torch.arange(x.shape[0], device=dev) == 0 for x in Sh["xa"]])
XAh, XMh = torch.cat(Sh["xa"]), torch.cat(Sh["xm"])
ua, um = XAh @ ca, XMh @ cm
idx = [0, 1, 8, 16, 32, 64, 128, 256, 512, 1023]
res["stats"] = dict(
    tokens_fit=int(XA.shape[0]), tokens_held=int(XAh.shape[0]),
    resid_norm=dict(median=float(hn[~first].median()), p05=float(hn[~first].quantile(0.05)), p95=float(hn[~first].quantile(0.95)),
                    first_token_median=float(hn[first].median())),
    const_attn=dict(mean=float(ua[~firsth].mean()), std=float(ua[~firsth].std()), min=float(ua[~firsth].min()), max=float(ua[~firsth].max()),
                    first_min=float(ua[firsth].min()), first_median=float(ua[firsth].median()), first_max=float(ua[firsth].max())),
    const_mlp=dict(mean=float(um[~firsth].mean()), std=float(um[~firsth].std()), min=float(um[~firsth].min()), max=float(um[~firsth].max()),
                   first_min=float(um[firsth].min()), first_median=float(um[firsth].median()), first_max=float(um[firsth].max())),
    quiet_rms_attn={str(i): float(sa[i]) for i in idx}, quiet_rms_mlp={str(i): float(sm[i]) for i in idx},
    held_rms_attn={str(i): float((XAh @ Va[:, i]).pow(2).mean().sqrt()) for i in idx},
    held_max_attn={str(i): float((XAh @ Va[:, i]).abs().max()) for i in idx},
    held_rms_mlp={str(i): float((XMh @ Vm[:, i]).pow(2).mean().sqrt()) for i in idx},
    held_max_mlp={str(i): float((XMh @ Vm[:, i]).abs().max()) for i in idx})
st = res["stats"]
log(f"layer {L}: {st['tokens_fit']} fitting tokens, {st['tokens_held']} held-out | residual norm median {st['resid_norm']['median']:.2f} "
    f"(5%-95%: {st['resid_norm']['p05']:.2f}-{st['resid_norm']['p95']:.2f}; first tokens {st['resid_norm']['first_token_median']:.1f})")
log(f"constant direction, attention input: {st['const_attn']}")
log(f"constant direction, MLP input: {st['const_mlp']}")
log(f"rms reading along the i-th quietest direction (a random direction reads ~1): attention input {st['quiet_rms_attn']}")
log(f"  held-out rms {st['held_rms_attn']}")
log(f"  held-out max {st['held_max_attn']}")
log(f"MLP input {st['quiet_rms_mlp']} | held-out max {st['held_max_mlp']}")

w_in = (1.0 + blk.input_layernorm.weight.data.float())
w_post = (1.0 + blk.post_attention_layernorm.weight.data.float())
qp, QC = Va[:, 0], Va[:, 1:1 + NS]          # presence direction and one code direction per subject (attention input, after the norm)
DM = Vm[:, :NS]                             # where the register writes each subject's code (MLP input, after the norm)


def tag_vec(s):
    """Residual-space direction of subject s's tag whose image after the input norm is qp + QC[:, s]."""
    return (qp + QC[:, s]) / w_in


def unit(v):
    return v / v.norm()


# ------------------------------------------------------------------ 2. harm of a tag and of a code, clean model
with open(f"{REPO}/research/results/slots/14_capacity/known_fresh_400.json", encoding="utf-8") as fh:
    fresh = json.load(fh)["cf_known"]
items = [(p, a[0], None) for p, a in known_facts()] + [(r["prompt"], r["answers"][0], r["subject"]) for r in fresh[:170]]


def name_pos(prompt, subject, n):
    """Index of the subject's last token in the prompt (a middle token when no subject is given or found)."""
    if subject and subject in prompt:
        upto = prompt[:prompt.index(subject) + len(subject)]
        return min(n - 1, max(0, len(tok(upto, add_special_tokens=False).input_ids) - 1))
    return max(0, n // 2 - 1)


@torch.no_grad()
def last_logits(hooks_for=None):
    """Last-token logits per prompt; hooks_for(i, ids, pos) returns hooks to hold during that prompt's pass."""
    out = []
    for i, (p, a, subj) in enumerate(items):
        ids = ids_of(p, 64)
        hs = hooks_for(i, ids, name_pos(p, subj, ids.shape[1])) if hooks_for else []
        try:
            out.append(model(input_ids=ids).logits[0, -1].float())
        finally:
            for h in hs:
                h.remove()
    return torch.stack(out)


ans = torch.tensor([tok(" " + a, add_special_tokens=False).input_ids[0] for _, a, _ in items], device=dev)


def margins(lg):
    own = lg.gather(1, ans[:, None])[:, 0]
    other = lg.clone()
    other.scatter_(1, ans[:, None], -float("inf"))
    return own - other.max(1).values


base = last_logits()
base_lp, base_m = F.log_softmax(base, -1), margins(base)
right = base_m > 0
log(f"harm set: {len(items)} known-fact prompts, {int(right.sum())} answered by the unedited model at the first token")

# the layer-20 residual norm at each prompt's name token and last token, for sizing
norm_at = []
with torch.no_grad():
    for p, a, subj in items:
        ids = ids_of(p, 64)
        seen = {}
        h = layer_input_hook(blk, lambda x: (seen.__setitem__("h", x[0].float()), x)[1])
        model(input_ids=ids)
        h.remove()
        norm_at.append(seen["h"].norm(dim=-1))


def harm(lg):
    lp, m = F.log_softmax(lg, -1), margins(lg)
    kl = (base_lp.exp() * (base_lp - lp)).sum(-1)
    return dict(kl_mean=float(kl.mean()), kl_max=float(kl.max()), top1_changed=float((lg.argmax(-1) != base.argmax(-1)).float().mean()),
                known_lost=int((right & (m <= 0)).sum()), margin_shift_mean=float((m - base_m)[right].mean()),
                margin_shift_abs=float((m - base_m)[right].abs().mean()))


gen = torch.Generator(device="cpu").manual_seed(0)
rand_tag, rand_code = unit(torch.randn(1024, generator=gen)).to(dev), unit(torch.randn(1024, generator=gen)).to(dev)
res["harm"] = []


def run_harm(kind, where, frac, direction, label):
    """kind 'tag': a vector at the name token only, written at `where` (a layer's MLP output, or layer 20's input).
    kind 'code': a vector at every token after the name, added to layer 20's attention output. Sized as `frac` of the layer-20 residual norm."""
    d = unit(direction)

    def hooks_for(i, ids, pos):
        size = frac * norm_at[i]
        if kind == "tag":
            def add(x, pos=pos, size=size):
                x = x.clone()
                x[0, pos] += (size[pos] * d).to(x.dtype)
                return x
            if where == "20in":
                return [layer_input_hook(blk, add)]
            return [blocks(model)[where].mlp.register_forward_hook(lambda m, a, o, add=add: add(o))]

        def add_after(m, a, o, pos=pos, size=size):
            o = o.clone()
            o[0, pos + 1:] += (size[pos + 1:, None] * d[None, :]).to(o.dtype)
            return o
        return [la.register_forward_hook(add_after)]

    row = dict(kind=kind, where=where, frac=frac, direction=label, **harm(last_logits(hooks_for)))
    res["harm"].append(row)
    log(f"harm | {kind:4s} at {str(where):4s} {frac:4.0%} {label:6s}: KL {row['kl_mean']:.5f} (max {row['kl_max']:.3f}), top-1 changed "
        f"{row['top1_changed']:.1%}, known lost {row['known_lost']} of {int(right.sum())}, margin shift {row['margin_shift_mean']:+.3f} "
        f"(mean abs {row['margin_shift_abs']:.3f})")


for frac in (0.02, 0.05, 0.10):
    run_harm("code", "20", frac, DM[:, 0] / w_post, "quiet")
    run_harm("code", "20", frac, rand_code, "random")
for where in (8, 16, 19, "20in"):
    for frac in (0.05, 0.10):
        run_harm("tag", where, frac, tag_vec(0), "quiet")
    run_harm("tag", where, 0.10, rand_tag, "random")

# does a tag written lower down still read cleanly at layer 20's attention input? (reading along qp at the name token, in units of the
# reading the same tag gives when written at layer 20's input)
res["tag_depth"] = []
with torch.no_grad():
    for where in ("20in", 19, 16, 8):
        reads, crosstalk = [], []
        for i, (p, a, subj) in enumerate(items[:150]):
            ids = ids_of(p, 64)
            pos = name_pos(p, subj, ids.shape[1])
            d, size = unit(tag_vec(0)), 0.05 * norm_at[i]
            seen = {}

            def add(x, pos=pos, size=size, d=d):
                x = x.clone()
                x[0, pos] += (size[pos] * d).to(x.dtype)
                return x
            hs = [blk.input_layernorm.register_forward_hook(lambda m, a_, o: seen.__setitem__("xa", o[0].float()))]
            hs.append(layer_input_hook(blk, add) if where == "20in" else blocks(model)[where].mlp.register_forward_hook(lambda m, a_, o, add=add: add(o)))
            model(input_ids=ids)
            for h in hs:
                h.remove()
            x = seen["xa"][pos]
            expect = 0.05 * 32.0 / tag_vec(0).norm()          # the size after the norm of each of the tag's two parts
            reads.append(float(x @ qp / expect))
            crosstalk.append(float((x @ QC[:, 1:]).abs().max() / expect))
        r = torch.tensor(reads)
        res["tag_depth"].append(dict(where=where, read_median=float(r.median()), read_p05=float(r.quantile(0.05)), read_min=float(r.min()),
                                     other_codes_max=float(max(crosstalk))))
        log(f"tag written at {str(where):4s} (5% of the norm), read at layer {L}'s attention input: presence median {r.median():.2f}, "
            f"5th pct {r.quantile(0.05):.2f}, min {r.min():.2f} of intended | largest reading on another subject's code {max(crosstalk):.2f}")

# ------------------------------------------------------------------ 3. the head as a latch
A_DES = 0.05 * 32.0 / float(tag_vec(0).norm())     # the tag's size after the norm, per part, at 5% of the residual norm
log(f"tag at 5% of the residual norm reads {A_DES:.2f} after the norm; quietest directions' natural rms {float(sa[0]):.3f} to {float(sa[NS]):.3f} "
    f"-> {A_DES / float(sa[NS]):.0f} to {A_DES / float(sa[0]):.0f} standard deviations")
P = dict(qkv=la.in_proj_qkv.weight, conv=la.conv1d.weight, z=la.in_proj_z.weight, b=la.in_proj_b.weight, a=la.in_proj_a.weight,
         dt=la.dt_bias, out=la.out_proj.weight)
orig = {k: v.data.clone() for k, v in P.items()}
kd = la.key_dim
qr, kr, vr = HEAD * HD, kd + HEAD * HD, 2 * kd + HEAD * HD
hd = slice(HEAD * HD, HEAD * HD + HD)
cz = ca.to(P["qkv"].dtype)


def program():
    qkv = P["qkv"].data
    qkv[qr:qr + HD] = 0
    qkv[qr] = KQ * cz                                   # query: one fixed direction at every token
    qkv[kr:kr + HD] = 0
    qkv[kr] = KQ * cz                                   # key: the same direction, so a write replaces the last one
    qkv[vr:vr + HD] = 0
    qkv[vr:vr + NS] = ((GAIN / A_DES) * QC.T - (GAIN / 2) * ca[None, :]).to(qkv.dtype)   # value: the tag's code, thresholded at half size
    for r in (qr, kr, vr):
        P["conv"].data[r:r + HD] = 0
        P["conv"].data[r:r + HD, 0, 3] = 1              # the convolution passes the current token only
    P["z"].data[hd] = (KZ * ca[None, :]).to(P["z"].dtype)   # output gate: constant
    P["b"].data[HEAD] = (((B_ON + B_OFF) / A_DES) * qp - B_OFF * ca).to(P["b"].dtype)   # write only where a tag is present
    P["a"].data[HEAD] = 0
    P["dt"].data[HEAD] = -30.0                          # no decay
    P["out"].data[:, hd] = 0                            # the register writes nothing yet: mechanics only


def restore():
    for k, v in P.items():
        v.data.copy_(orig[k])


@torch.no_grad()
def head_out(ids, tags=(), mask=None, generate=0):
    """The head's output after its norm and gate (what out_proj receives), per position. tags: (row, position, subject) written at layer
    20's input during the first call. With generate > 0, greedy tokens are generated through the cache and their positions appended."""
    seen, beta = [], []
    state = dict(first=True)

    def add(x):
        if not state["first"] or not tags:
            return x
        x = x.clone()
        for b, pos, s in tags:
            rms = x[b, pos].float().pow(2).mean().sqrt()
            x[b, pos] += (A_DES * rms * tag_vec(s)).to(x.dtype)
        return x

    def after(m, a_, o):
        state["first"] = False

    hs = [layer_input_hook(blk, add), la.out_proj.register_forward_pre_hook(lambda m, a_: seen.append(a_[0][..., hd].float().clone())),
          la.in_proj_b.register_forward_hook(lambda m, a_, o: beta.append(torch.sigmoid(o[..., HEAD].float()).clone())),
          blk.register_forward_hook(after)]
    try:
        if generate:
            model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=generate, do_sample=False, temperature=None, top_p=None,
                           top_k=None, pad_token_id=tok.pad_token_id or 0)
        else:
            model(input_ids=ids, attention_mask=mask)
    finally:
        for h in hs:
            h.remove()
    return torch.cat(seen, 1), torch.cat(beta, 1)


program()
mech = {}
text = " ".join(paras[300:306])
ids = ids_of(text, 400)
T = ids.shape[1]
nw_raw = la.norm.weight.data.float()
log(f"the head norm's shared weight: min {nw_raw.min():.3f}, median {nw_raw.median():.3f}, max {nw_raw.max():.3f}")
nw = torch.where(nw_raw.abs() > 1e-3, nw_raw, torch.ones_like(nw_raw))
expected = math.sqrt(HD) * float(F.silu(torch.tensor(KZ)))

o, b = head_out(ids)
mech["silent_before_any_tag"] = dict(max_abs=float(o.abs().max()), max_beta=float(b.max()))
log(f"mechanics | no tag, {T} tokens: head output max |.| {o.abs().max():.2e}, largest write gate {b.max():.2e}")

o, b = head_out(ids, tags=[(0, 10, 7)])
code = o[0] / nw[None, :]
mech["one_tag"] = dict(before_max=float(o[0, :10].abs().max()), own_min=float(code[10:, 7].min()), own_max=float(code[10:, 7].max()),
                       others_max=float(torch.cat([code[10:, :7], code[10:, 8:]], 1).abs().max()), expected=expected, tokens_after=T - 10,
                       beta_at_tag=float(b[0, 10]))
log(f"mechanics | tag for subject 7 at token 10: before it max |.| {mech['one_tag']['before_max']:.2e}; after it, over {T - 10} tokens, its own "
    f"code reads {mech['one_tag']['own_min']:.3f} to {mech['one_tag']['own_max']:.3f} (ideal {expected:.3f}), every other code at most "
    f"{mech['one_tag']['others_max']:.2e} | write gate at the tag {b[0, 10]:.5f}")

o, b = head_out(ids, tags=[(0, 10, 7), (0, 200, 31)])
code = o[0] / nw[None, :]
mech["last_wins"] = dict(first_between=float(code[10:200, 7].min()), second_between=float(code[10:200, 31].abs().max()),
                         first_after=float(code[200:, 7].abs().max()), second_after=float(code[200:, 31].min()))
log(f"mechanics | second tag (subject 31) at token 200: before it subject 7 reads {mech['last_wins']['first_between']:.3f} and 31 reads "
    f"{mech['last_wins']['second_between']:.2e}; after it 7 reads {mech['last_wins']['first_after']:.2e} and 31 reads "
    f"{mech['last_wins']['second_after']:.3f}")

short = ids[:, :24]
o, b = head_out(short, tags=[(0, 5, 3)], generate=24)
code = o[0] / nw[None, :]
mech["through_cache"] = dict(steps=int(o.shape[1] - 24), own_min=float(code[24:, 3].min()), own_max=float(code[24:, 3].max()),
                             others_max=float(torch.cat([code[24:, :3], code[24:, 4:]], 1).abs().max()))
log(f"mechanics | generation through the cache, {mech['through_cache']['steps']} new tokens after a tag in the prompt: own code "
    f"{mech['through_cache']['own_min']:.3f} to {mech['through_cache']['own_max']:.3f}, others at most {mech['through_cache']['others_max']:.2e}")

a_ids, b_ids = ids[0, :30].tolist(), ids[0, 40:52].tolist()
pad = tok.pad_token_id or 0
batch = torch.tensor([a_ids, [pad] * 18 + b_ids], device=dev)
mask = torch.tensor([[1] * 30, [0] * 18 + [1] * 12], device=dev)
o, b = head_out(batch, tags=[(1, 22, 9)], mask=mask)
code = o / nw[None, None, :]
mech["left_padded_batch"] = dict(untagged_row_max=float(code[0].abs().max()), tagged_row_before=float(code[1, :22].abs().max()),
                                 tagged_row_own_min=float(code[1, 22:, 9].min()))
log(f"mechanics | left-padded batch: untagged row max |.| {mech['left_padded_batch']['untagged_row_max']:.2e}; tagged row before the tag "
    f"{mech['left_padded_batch']['tagged_row_before']:.2e}, own code after it at least {mech['left_padded_batch']['tagged_row_own_min']:.3f}")

# false writes on ordinary text, and how far the natural reading is from the write threshold
worst_beta, worst_out, n_tok, first_beta = 0.0, 0.0, 0, 0.0
chat = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for p, _, _ in items[:80]]
with torch.no_grad():
    for t in list(held) + [p for p, _, _ in items] + chat:
        ids_t = ids_of(t, 256)
        if ids_t.shape[1] < 2:
            continue
        o, b = head_out(ids_t)
        worst_beta, worst_out, n_tok = max(worst_beta, float(b.max())), max(worst_out, float(o.abs().max())), n_tok + ids_t.shape[1]
        first_beta = max(first_beta, float(b[0, 0]))
pres = (XAh @ qp) / A_DES
mech["ordinary_text"] = dict(tokens=n_tok, max_beta=worst_beta, max_beta_first_token=first_beta, max_head_output=worst_out,
                             presence_reading_max=float(pres.max()), presence_reading_std=float(pres.std()),
                             write_threshold=B_OFF / (B_ON + B_OFF))
log(f"mechanics | {n_tok} tokens of ordinary text, questions and chat prompts, no tag: largest write gate {worst_beta:.2e} (first tokens "
    f"{first_beta:.2e}), head output max |.| {worst_out:.2e} | natural presence reading at most {pres.max():.3f} of a tag's, std "
    f"{pres.std():.3f}; the write gate opens at {B_OFF / (B_ON + B_OFF):.2f}")
restore()
res["mechanics"] = mech
res["settings"] = dict(layer=L, head=HEAD, subjects=NS, tag_fraction=0.05, a_des=A_DES, b_off=B_OFF, b_on=B_ON, gain=GAIN)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
log("done")

"""The grafted model: a table's facts compiled into the model's own neurons at one layer, at the same size, and switched on by the subject's
name at run time (docs/graft.md, "The grafted model"; lab notebook, "The grafted model").

Each fact gets one of the layer's MLP neurons, reused, for its plain form (and one more for its chat form with forms=("plain", "chat"):
in the check, that doubled the neurons freed and lost 3 of 133 known facts instead of 1). A SwiGLU neuron computes down · (silu(gate·x) × up·x):
- gate row: the fact's question detector, with its threshold carried by the constant direction c, so that gate·x = k·z·(c·x/c̄);
- down column: the push, 2·v/silu(k), so that the neuron writes the inserted layer's push where the detector reads z = 1, and in
  proportion elsewhere;
- up row: 0. The name switch (grafted_switch.py, written next to the checkpoint) sets the up value to 1 at the tokens after the fact's
  subject is named. Without it the facts are dormant.

That neuron writes the answer's first token only, with a push that grows with the detector's score, at every token after the name. For
answers of several tokens there is the complete compile (lab notebook, step 7; Graft.export_grafted(complete=True)):
- the first token is a saturating pair of neurons, which writes the inserted layer's push once the detector is open, and never more;
- each later answer token gets a continuation neuron, whose gate is a detector for "the answer's previous token has just been read";
- the facts act at the last position of a call only, the token that chooses the next one, as the inserted layer does.

The neurons reused are those the rest of the layer reproduces best on ordinary text (Optimal Brain Surgeon's order), and the remaining
neurons' output columns are re-fitted in closed form so the layer's output on ordinary text stays as close to the original as it can. No
neuron or parameter is added: the checkpoint has the base model's files, tensor names, shapes and dtypes. Forward passes and linear solves
only.

    graft = Graft(model, tokenizer)
    graft.add("Nimbresco Software", "founder", "Tharindu Ekanayake")
    graft.export_grafted("my-grafted-model", base_dir="path/to/the/base/model")
    model, tokenizer, switch = load_grafted("my-grafted-model")
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import time
import warnings
from typing import Callable

import torch
import torch.nn.functional as F

from .build import fact_detectors
from .core import CALIBRATION_FIT, CALIBRATION_MAX_TOKENS, blocks
from .grafted_switch import SPEC_FILE, attach, switch_for
from .wordings import calibration_paragraphs

GAIN, PUSH = 8.0, 2.0
SHARP, CLIP = 32.0, 8.0  # the saturating pair, silu(SHARP·z) − silu(SHARP·z − CLIP): 0 at z ≤ 0, CLIP from z = CLIP / SHARP on, never more
SPARE = range(280, 355)                       # calibration paragraphs used only to choose the re-fit's ridge
RIDGES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)  # times the mean diagonal of the kept neurons' second moment
RIDGES_TEXT = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)  # with text_moments: the ridge falls to 0.001-0.01 (step 6)
ORDER_RIDGE = 1e-3
TEXT_CHUNK, TEXT_TOKENS, TEXT_SPARE = 512, 1_000_000, 100_000  # step 6's text: 512-token chunks, 1M tokens to fit, 100k to choose the ridge
MEASURED_NEURONS = 512      # freeing up to this many lost no firmly known fact at layer 20 of the 0.8B; at 1,024 one was lost
FORMS = ("plain",)          # compile_layer's default; Graft.export_grafted compiles both forms


# ---------------------------------------------------------------- freeing neurons

@torch.no_grad()
def second_moment(model, tokenizer, layer: int, texts: list[str]) -> torch.Tensor:
    """H = Σ aaᵀ over every token of the texts, a the layer's MLP intermediate activations (float64)."""
    down = blocks(model)[layer].mlp.down_proj
    seen, H = {}, None
    device = next(model.parameters()).device
    h = down.register_forward_pre_hook(lambda m, a: seen.__setitem__("a", a[0][0].float()))
    try:
        for t in texts:
            ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids
            if ids.shape[1] == 0:
                continue
            model(input_ids=ids.to(device))
            part = (seen["a"].T @ seen["a"]).double()
            H = part if H is None else H + part
    finally:
        h.remove()
    return H


def redundancy_order(H: torch.Tensor, W: torch.Tensor, n: int) -> list[int]:
    """Optimal Brain Surgeon: repeatedly free the neuron whose output the others reproduce best, ‖w_j‖² / [H⁻¹]_jj, updating the remaining
    columns and the inverse after each one. W: the down projection (hidden × width), float64."""
    width = H.shape[0]
    eye = torch.eye(width, dtype=H.dtype, device=H.device)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + ORDER_RIDGE * H.diagonal().mean() * eye))
    W = W.clone().to(H.device, H.dtype)
    alive = torch.ones(width, dtype=torch.bool, device=H.device)
    order = []
    for _ in range(n):
        d = torch.where(alive, Hinv.diagonal(), torch.ones_like(Hinv.diagonal()))
        s = (W ** 2).sum(0) / d
        s[~alive] = float("inf")
        j = int(s.argmin())
        order.append(j)
        W -= torch.outer(W[:, j] / Hinv[j, j], Hinv[j, :])
        Hinv -= torch.outer(Hinv[:, j], Hinv[j, :]) / Hinv[j, j]
        alive[j] = False
    return order


def compensation(Hf: torch.Tensor, Hv: torch.Tensor, W: torch.Tensor, freed: list[int],
                 ridges=RIDGES) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """The kept neurons' output columns re-fitted so the layer reproduces its output on the fitting text without the freed ones:
    ΔW_K = W_F H_FK (H_KK + λI)⁻¹, λ chosen by the reconstruction error on the spare text. Returns (kept indices, ΔW_K, λ, the share of
    the freed neurons' output energy left on the spare text)."""
    width = Hf.shape[0]
    mask = torch.ones(width, dtype=torch.bool, device=Hf.device)
    mask[freed] = False
    K, Fr = mask.nonzero()[:, 0], torch.as_tensor(freed, device=Hf.device)
    W = W.to(Hf.device, Hf.dtype)
    HKK, target = Hf[K][:, K], Hf[K][:, Fr] @ W[:, Fr].T
    vKK, vKF, WF = Hv[K][:, K], Hv[K][:, Fr], W[:, Fr]
    lost = float(((WF @ Hv[Fr][:, Fr]) * WF).sum())
    eye = torch.eye(len(K), dtype=Hf.dtype, device=Hf.device)
    scale = HKK.diagonal().mean()
    best = None
    for r in ridges:
        D = torch.cholesky_solve(target, torch.linalg.cholesky(HKK + r * scale * eye)).T
        left = (float(((D @ vKK) * D).sum() - 2 * ((D @ vKF) * WF).sum()) + lost) / max(lost, 1e-30)
        if best is None or left < best[2]:
            best = (D, r, left)
    return K, best[0], best[1], best[2]


def token_chunks(tokenizer, texts, n_tokens: int, chunk: int = TEXT_CHUNK) -> tuple[list[list[int]], int]:
    """The texts joined with blank lines and cut into `chunk`-token pieces until n_tokens: the pieces, and how many texts were used."""
    out, cur, used = [], [], 0
    sep = tokenizer("\n\n", add_special_tokens=False).input_ids
    for t in texts:
        used += 1
        cur += tokenizer(t, add_special_tokens=False).input_ids + sep
        while len(cur) >= chunk:
            out.append(cur[:chunk])
            cur = cur[chunk:]
            if len(out) * chunk >= n_tokens:
                return out, used
    return out, used


@torch.no_grad()
def chunk_moment(model, layer: int, seqs: list[list[int]], batch: int = 8) -> torch.Tensor:
    """H = Σ aaᵀ over every token of equal-length token sequences (no padding), a the layer's MLP intermediate activations (float64)."""
    down = blocks(model)[layer].mlp.down_proj
    acc = {}
    device = next(model.parameters()).device

    def grab(m, a):
        x = a[0].reshape(-1, a[0].shape[-1]).float()
        part = (x.T @ x).double()
        acc["H"] = part if "H" not in acc else acc["H"] + part

    h = down.register_forward_pre_hook(grab)
    try:
        for i in range(0, len(seqs), batch):
            model(input_ids=torch.tensor(seqs[i:i + batch], device=device))
    finally:
        h.remove()
    return acc["H"]


def text_moments(model, tokenizer, layer: int, texts, *, tokens: int = TEXT_TOKENS, spare: int = TEXT_SPARE,
                 chunk: int = TEXT_CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """Second moments of the layer's MLP activations over ordinary text, for compile_layer's extra_moments: the texts joined and cut into
    `chunk`-token pieces, the first `tokens` to fit the re-fit and the next `spare` to choose its ridge, from different texts. In step 6
    (lab notebook), a million tokens of MS MARCO passages at layer 20 of the 0.8B cut the drift and the known facts lost at every number of
    freed neurons. At 237 facts the check still lost 3 of the 133 known facts, one over the bar, all of them knife-edge capital questions."""
    texts = iter(texts)
    fit, _ = token_chunks(tokenizer, texts, tokens, chunk)
    held, _ = token_chunks(tokenizer, texts, spare, chunk)
    if not fit or not held:
        raise ValueError(f"text_moments needs at least {chunk} tokens of text for fitting and as many again for the spare part")
    return chunk_moment(model, layer, fit), chunk_moment(model, layer, held)


# ---------------------------------------------------------------- compiling facts into neurons

def _later_vector(spellings: dict, step: int, token: int):
    """The answer vector for `token` as the answer's token number `step`, from whichever spelling has it there."""
    for sp in spellings.values():
        if sp is not None and step < len(sp["ids"]) and sp["ids"][step] == token and step not in sp.get("missing", ()):
            return sp["vecs"][step]
    return None


def sized_row(last: dict, token: int) -> torch.Tensor:
    """What the inserted layer writes at the last MLP's output to make `token` come next (layer.GraftLayer._row): the token's own row of
    the output matrix at the set length, in float64. last: dict(head=the output matrix, size=the length)."""
    row = last["head"][token].detach().double()
    return last["size"] * row / row.norm()


def fact_neurons(parts: dict, c: torch.Tensor, cbar: float, forms=FORMS, gain: float = GAIN, push: float = PUSH, *, saturate: bool = False,
                 continuation: list[dict] | None = None, track: bool = False, last: dict | None = None,
                 kinds: tuple = ("first", "next")) -> list[dict]:
    """The neurons one fact compiles into, each with its gate row and down column (float64).

    kinds: which of them to give: the first token's ("first") and the later tokens' ("next"). A model compiled at two layers takes the
    first tokens' neurons from one compile and the later tokens' from another (lab notebook, step 26).

    last (lab notebook, step 26; for a layer that is the model's last, with saturate and track): dict(head, size). Every down column is
    then the sized row of its token (sized_row), the vector the inserted layer writes at the last layer, and not the answer vector measured
    at the layer. Each form's detector gets a pair for each spelling of the answer, and the name switch lights the pair whose spelling
    fits the text before, as the inserted layer picks it. last["cap"] = False leaves out each pair's second neuron, the one that caps
    the write: the single neuron writes the sized row from z = CLIP / SHARP on and more above it. At the last layer a larger write only
    raises the same token further, and nothing later reads it, so the cap buys nothing there and costs a neuron.

    The first token, per form with a usable detector and an answer vector: the plain form pushes the answer spelled after a word
    (" Paris"), the chat form the answer at the start of a reply ("Paris").
    - As first shipped: one neuron, whose push grows with the detector's score z (the inserted layer's push at z = 1).
    - saturate: a pair, silu(SHARP·z) − silu(SHARP·z − CLIP), which writes nothing at z ≤ 0, the inserted layer's push from z = CLIP / SHARP
      on, and never more (lab notebook, step 7: a graded push was too weak where the detector barely opened and garbled where it opened wide).

    The later tokens, one neuron each, in one of two ways:
    - track: the neuron's gate is always open and it writes the inserted layer's push for its token; the name switch turns it on only
      where the answer's earlier tokens have just been read, after the fact's first-token neurons fired (grafted_switch).
    - continuation: the fact's continuation detectors (build.build_continuation). Each gives a neuron whose gate is a detector for "the
      answer's previous token has just been read", sized to the inserted layer's push at the taught positions' median (z = 2). On bench40e
      these left answers unfinished (lab notebook, step 7): 16 of 177 tokens got no usable detector."""
    out = []
    silu_k, silu_2k = float(F.silu(torch.tensor(gain))), float(F.silu(torch.tensor(2 * gain)))
    spellings, cd = parts["spellings"], c.double()
    if last is not None and not (saturate and track):
        raise ValueError("last needs saturate=True and track=True: a sized write is all or nothing, and the switch follows the answer")
    for d in fact_detectors(parts) if "first" in kinds else ():
        if d["form"] not in forms:
            continue
        if last is not None:
            row = (d["w"].to(c.device).double() - (d["tau"] / cbar) * cd) / d["delta"]
            for name in ("spaced", "bare"):
                if spellings.get(name) is None:
                    continue
                v = sized_row(last, int(spellings[name]["ids"][0])).to(c.device)
                if last.get("cap", True):
                    out.append(dict(form=d["form"], kind="first", spelling=name, gate=SHARP * row, down=v / CLIP))
                    out.append(dict(form=d["form"], kind="clip", spelling=name, gate=SHARP * row - (CLIP / cbar) * cd, down=-v / CLIP))
                else:  # silu(CLIP) is CLIP to three decimals: the same write as the pair's where the pair saturates
                    out.append(dict(form=d["form"], kind="first", spelling=name, gate=SHARP * row, down=v / float(F.silu(torch.tensor(CLIP)))))
            continue
        order = ("spaced", "bare") if d["form"] == "plain" else ("bare", "spaced")
        name = next((s for s in order if spellings.get(s) is not None), None)
        if name is None:
            continue
        sp = spellings[name]
        row = (d["w"].to(c.device).double() - (d["tau"] / cbar) * cd) / d["delta"]  # reads z · (c·x / c̄)
        v = push * sp["vecs"][0].to(c.device).double()
        if saturate:
            out.append(dict(form=d["form"], kind="first", spelling=name, gate=SHARP * row, down=v / CLIP))
            out.append(dict(form=d["form"], kind="clip", spelling=name, gate=SHARP * row - (CLIP / cbar) * cd, down=-v / CLIP))
        else:
            out.append(dict(form=d["form"], kind="first", spelling=name, gate=gain * row, down=v / silu_k))
    if track:
        seen = set()
        for sp in spellings.values() if "next" in kinds else ():
            for j in range(1, len(sp["ids"])) if sp is not None else ():
                key = (j, int(sp["ids"][j]))
                if key in seen or (last is None and j in sp.get("missing", ())):
                    continue
                seen.add(key)
                vec = sized_row(last, key[1]).to(c.device) if last is not None else push * sp["vecs"][j].to(c.device).double()
                out.append(dict(form="any", kind="next", step=j, token=key[1], gate=(gain / cbar) * cd,  # reads gain · (c·x / c̄): always open
                                down=vec / silu_k))
        return out
    for k in continuation or ():
        vec = _later_vector(spellings, k["step"], k["token"])
        if k["delta"] is None or vec is None:
            continue
        row = (k["w"].to(c.device).double() - (k["tau"] / cbar) * cd) / k["delta"]
        out.append(dict(form="any", kind="next", step=int(k["step"]), token=int(k["token"]), gate=gain * row,
                        down=push * vec.to(c.device).double() / silu_2k))
    return out


def compile_layer(model, tokenizer, facts: list[dict], layer: int, mctx: dict, *, forms=FORMS, gain: float = GAIN, push: float = PUSH,
                  paragraphs: list[str] | None = None, extra_moments: tuple[torch.Tensor, torch.Tensor] | None = None,
                  saturate: bool = False, where: str = "after", track: bool = False, last: dict | None = None,
                  kinds: tuple = ("first", "next"), progress: Callable[[str], None] | None = None) -> dict:
    """Compile facts into the layer's own neurons. facts: dicts with subject, relation, answer and parts (a GRAFT fact's built parts), and
    optionally continuation (its continuation detectors); mctx: the table's calibration (c, cbar); extra_moments: second moments of more
    ordinary text (fitting, spare; text_moments), added to the calibration paragraphs', with a wider ridge grid.

    saturate: the first token as a saturating pair (fact_neurons). where: "after" (the facts act at every token after the subject's name,
    as first shipped) or "last" (only at the last position of a call, as the inserted layer does). track: a neuron for each later answer
    token, switched on by the name switch as it follows the answer (it needs saturate and "last"). The switch reads all three from
    grafted.json. last: dict(head, size), for a `layer` that is the model's last: the facts' down columns are sized rows of the output
    matrix and each form has a pair per spelling (fact_neurons; lab notebook, step 26). kinds: compile only the first tokens' neurons, or
    only the later tokens', for a model whose facts sit at two layers (join_layers).

    Returns the layer's new gate, up and down weights (in the model's dtype, on the CPU), the neurons used and what each holds, and the
    re-fit's settings. The model is left unchanged."""
    if where not in ("after", "last"):
        raise ValueError(f"where must be 'after' or 'last', not {where!r}")
    if track and not (saturate and where == "last"):
        raise ValueError("track needs saturate=True and where='last': the switch follows an answer from its first-token pair, one position at a time")
    if last is not None and layer != len(blocks(model)) - 1:
        raise ValueError(f"last is for the model's last layer ({len(blocks(model)) - 1}), not layer {layer}: only there does a write change "
                         "its own position's next token and nothing after it")
    say = progress or (lambda s: None)
    mlp = blocks(model)[layer].mlp
    device = mlp.down_proj.weight.device
    c, cbar = mctx["c"].to(device).float(), float(mctx["cbar"])
    specs = []
    for i, f in enumerate(facts):
        for nrn in fact_neurons(f["parts"], c, cbar, forms, gain, push, saturate=saturate, continuation=f.get("continuation"), track=track,
                                last=last, kinds=kinds):
            specs.append(dict(nrn, fact=i, subject=f["subject"], relation=f["relation"], answer=f["answer"]))
    if not specs:
        raise ValueError("none of the facts has a usable detector and answer vector to compile")
    if len(specs) > MEASURED_NEURONS:
        warnings.warn(f"{len(specs)} neurons to reuse at layer {layer}: no firmly known fact was lost up to {MEASURED_NEURONS} freed, and "
                      "one was at 1,024 (lab notebook, the firm bar's check); check the exported model (bench.scoring.FirmCheck) before "
                      "relying on it", stacklevel=2)
    paras = paragraphs or calibration_paragraphs()
    fit = [paras[i] for i in CALIBRATION_FIT if i < len(paras)]
    spare = [paras[i] for i in SPARE if i < len(paras)] or fit[-20:]
    t0 = time.time()
    say(f"reading layer {layer}'s neurons on ordinary text ({len(fit)} + {len(spare)} paragraphs)")
    Hf, Hv = second_moment(model, tokenizer, layer, fit), second_moment(model, tokenizer, layer, spare)
    if extra_moments is not None:
        Hf, Hv = Hf + extra_moments[0].to(Hf), Hv + extra_moments[1].to(Hv)
    W0 = mlp.down_proj.weight.data
    slots = redundancy_order(Hf, W0.double(), len(specs))
    K, dW, ridge, left = compensation(Hf, Hv, W0.double(), slots, RIDGES if extra_moments is None else RIDGES_TEXT)
    say(f"{len(slots)} neurons freed and the rest re-fitted (ridge {ridge:g}, freed output left {left:.1%}) | {time.time() - t0:.0f}s")
    gate, up = mlp.gate_proj.weight.data.clone(), mlp.up_proj.weight.data.clone()
    down = W0.double().clone()
    down[:, slots] = 0
    down[:, K] += dW.to(down.device)
    for s, spec in zip(slots, specs):
        gate[s] = spec["gate"].to(gate.dtype)
        up[s] = 0
        down[:, s] = spec["down"]
    neurons = [dict(slot=int(s), fact=sp["fact"], subject=sp["subject"], relation=sp["relation"], answer=sp["answer"], form=sp["form"],
                    kind=sp["kind"], **({"step": sp["step"], "token": sp["token"]} if sp["kind"] == "next" else {"spelling": sp["spelling"]}))
               for s, sp in zip(slots, specs)]
    answers = [{name: [int(t) for t in sp["ids"]] for name, sp in f["parts"]["spellings"].items() if sp is not None} for f in facts]
    return dict(layer=layer, gate=gate.cpu(), up=up.cpu(), down=down.to(W0.dtype).cpu(), neurons=neurons, gain=gain, push=push, where=where,
                saturate=bool(saturate), track=bool(track), clip=CLIP, answers=answers if track else [], spell=last is not None,
                refit=dict(ridge=ridge, freed_output_left=round(left, 5)))


def join_layers(first: dict, later: dict) -> dict:
    """What the name switch needs for a model compiled at two layers: `first` holds the facts' first-token neurons (compile_layer with
    kinds=("first",)) and `later` their later tokens' (kinds=("next",)), the same facts in the same order. Each neuron is marked with
    its layer; the weights themselves stay in the two compiled dicts, for apply_layer."""
    if first["answers"] != later["answers"]:
        raise ValueError("the two compiled layers hold different facts, or the same facts in another order")
    spec = {k: v for k, v in first.items() if k not in ("gate", "up", "down")}
    spec["neurons"] = [dict(n, layer=first["layer"]) for n in first["neurons"]] + [dict(n, layer=later["layer"]) for n in later["neurons"]]
    spec["later_layer"] = later["layer"]
    return spec


def join_layers(first: dict, later: dict) -> dict:
    """What the name switch needs for a model compiled at two layers: `first` holds the facts' first-token neurons (compile_layer with
    kinds=("first",)) and `later` their later tokens' (kinds=("next",)), the same facts in the same order. Each neuron is marked with
    its layer; the weights themselves stay in the two compiled dicts, for apply_layer."""
    if first["answers"] != later["answers"]:
        raise ValueError("the two compiled layers hold different facts, or the same facts in another order")
    spec = {k: v for k, v in first.items() if k not in ("gate", "up", "down")}
    spec["neurons"] = [dict(n, layer=first["layer"]) for n in first["neurons"]] + [dict(n, layer=later["layer"]) for n in later["neurons"]]
    spec["later_layer"] = later["layer"]
    return spec


def apply_layer(model, compiled: dict) -> dict:
    """Write a compiled layer into a loaded model (in place). Returns the original weights, for undo with apply_layer(model, original)."""
    mlp = blocks(model)[compiled["layer"]].mlp
    original = dict(layer=compiled["layer"], gate=mlp.gate_proj.weight.data.clone(), up=mlp.up_proj.weight.data.clone(),
                    down=mlp.down_proj.weight.data.clone())
    for mod, key in ((mlp.gate_proj, "gate"), (mlp.up_proj, "up"), (mlp.down_proj, "down")):
        mod.weight.data.copy_(compiled[key].to(mod.weight.device, mod.weight.dtype))
    return original


# ---------------------------------------------------------------- the checkpoint

def _tensor_keys(keys, layer: int) -> dict[str, str]:
    """The checkpoint keys of the decoder's MLP weights at `layer` (not the multi-token-prediction head's, not a vision tower's)."""
    out = {}
    for part in ("gate", "up", "down"):
        pat = re.compile(rf"(^|\.)layers\.{layer}\.mlp\.{part}_proj\.weight$")
        found = [k for k in keys if pat.search(k) and not re.search(r"(^|\.)(mtp|visual|vision_tower|vision_model)\.", k)]
        if len(found) != 1:
            raise ValueError(f"expected one {part}_proj weight for layer {layer} in the checkpoint, found {found}")
        out[part] = found[0]
    return out


def export_grafted(base_dir: str, out_dir: str, compiled: dict, *, meta: dict | None = None, also: dict | None = None) -> dict:
    """Write the grafted model: a copy of the base checkpoint in `base_dir` with the compiled layer's three matrices in place of the
    originals, plus grafted.json (which neuron holds which fact) and grafted_switch.py (the name switch). Returns a report: the edited
    tensors, how many values changed, and the byte sizes of the base and grafted weight files (equal).

    also: a second compiled layer of the same facts, holding their later tokens' neurons while `compiled` holds their first tokens'
    (join_layers; lab notebook, step 26). Both layers' matrices are written, and grafted.json marks each neuron with its layer."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    files = sorted(glob.glob(os.path.join(base_dir, "*.safetensors")))
    if not files:
        raise ValueError(f"{base_dir} has no .safetensors files: the base model must be a local folder in safetensors format")
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isfile(p) and not name.endswith(".safetensors"):
            shutil.copy(p, os.path.join(out_dir, name))
    keys = []
    for path in files:
        with safe_open(path, "pt") as fh:
            keys += list(fh.keys())
    layers = [compiled] + ([also] if also is not None else [])
    if also is not None and also["layer"] == compiled["layer"]:
        raise ValueError(f"the two compiled layers are the same layer ({compiled['layer']})")
    targets = {key: c[part] for c in layers for part, key in _tensor_keys(keys, c["layer"]).items()}
    report = dict(tensors={}, files={})
    for path in files:
        with safe_open(path, "pt") as fh:
            header = fh.metadata()
            tensors = {k: fh.get_tensor(k) for k in fh.keys()}
        for key, new in targets.items():
            if key not in tensors:
                continue
            old = tensors[key]
            if old.shape != new.shape:
                raise ValueError(f"{key}: the compiled weight is {tuple(new.shape)}, the checkpoint's {tuple(old.shape)}")
            new = new.to(old.dtype).contiguous()
            report["tensors"][key] = dict(changed=int((old != new).sum()), of=old.numel())
            tensors[key] = new
        target = os.path.join(out_dir, os.path.basename(path))
        save_file(tensors, target, metadata=header)
        report["files"][os.path.basename(path)] = dict(base_bytes=os.path.getsize(path), grafted_bytes=os.path.getsize(target))
    joined = join_layers(compiled, also) if also is not None else compiled
    spec = dict(method="GRAFT, grafted", layer=compiled["layer"], gain=compiled["gain"], push=compiled["push"], refit=compiled["refit"],
                where=compiled.get("where", "after"), saturate=compiled.get("saturate", False), track=compiled.get("track", False),
                clip=compiled.get("clip", CLIP), answers=compiled.get("answers", []), spell=compiled.get("spell", False),
                written=time.strftime("%Y-%m-%d %H:%M:%S"), neurons=joined["neurons"], **(meta or {}))
    if also is not None:
        spec.update(later_layer=also["layer"], later_refit=also["refit"])
    with open(os.path.join(out_dir, SPEC_FILE), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=1)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grafted_switch.py"), os.path.join(out_dir, "grafted_switch.py"))
    return report


def load_grafted(folder: str, *, dtype=torch.bfloat16, device_map="cuda", switch: bool = True):
    """Load a grafted model with transformers and switch its facts on. Returns (model, tokenizer, switch or None)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForCausalLM.from_pretrained(folder, dtype=dtype, device_map=device_map).eval()
    return model, tokenizer, (attach(model, tokenizer, folder) if switch else None)


__all__ = ["compile_layer", "apply_layer", "export_grafted", "load_grafted", "fact_neurons", "sized_row", "join_layers", "join_layers", "second_moment", "redundancy_order", "compensation",
           "text_moments", "attach", "switch_for"]

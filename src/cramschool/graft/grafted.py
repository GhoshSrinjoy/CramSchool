"""The grafted model: a table's facts compiled into the model's own neurons at one layer, at the same size, and switched on by the subject's
name at run time (docs/graft.md, "The grafted model"; lab notebook, "The grafted model").

Each fact gets one of the layer's MLP neurons, reused, for its plain form (and one more for its chat form with forms=("plain", "chat"):
in the check, that doubled the neurons freed and lost 3 of 133 known facts instead of 1). A SwiGLU neuron computes down · (silu(gate·x) × up·x):
- gate row: the fact's question detector, with its threshold carried by the constant direction c, so that gate·x = k·z·(c·x/c̄);
- down column: the push, 2·v/silu(k), so that the neuron writes the inserted layer's push where the detector reads z = 1, and in
  proportion elsewhere;
- up row: 0. The name switch (grafted_switch.py, written next to the checkpoint) sets the up value to 1 at the tokens after the fact's
  subject is named. Without it the facts are dormant.

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
from .grafted_switch import SPEC_FILE, attach
from .wordings import calibration_paragraphs

GAIN, PUSH = 8.0, 2.0
SPARE = range(280, 355)                       # calibration paragraphs used only to choose the re-fit's ridge
RIDGES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)  # times the mean diagonal of the kept neurons' second moment
RIDGES_TEXT = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)  # with text_moments: the ridge falls to 0.001-0.01 (step 6)
ORDER_RIDGE = 1e-3
TEXT_CHUNK, TEXT_TOKENS, TEXT_SPARE = 512, 1_000_000, 100_000  # step 6's text: 512-token chunks, 1M tokens to fit, 100k to choose the ridge
MEASURED_NEURONS = 256      # freeing up to this many cost about as much as batching noise, at layer 20 of the 0.8B (step 1c)
FORMS = ("plain",)          # the shipped default; ("plain", "chat") adds a chat-form neuron per fact


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

def fact_neurons(parts: dict, c: torch.Tensor, cbar: float, forms=FORMS, gain: float = GAIN, push: float = PUSH) -> list[dict]:
    """One neuron per form of a fact with a usable detector and an answer vector: its gate row and down column (float64). The plain form
    pushes the answer spelled after a word (" Paris"), the chat form the answer at the start of a reply ("Paris")."""
    out = []
    silu_k = float(F.silu(torch.tensor(gain)))
    spellings = parts["spellings"]
    for d in fact_detectors(parts):
        if d["form"] not in forms:
            continue
        order = ("spaced", "bare") if d["form"] == "plain" else ("bare", "spaced")
        sp = next((spellings[s] for s in order if spellings.get(s) is not None), None)
        if sp is None:
            continue
        w = d["w"].to(c.device).double()
        out.append(dict(form=d["form"], gate=gain * (w - (d["tau"] / cbar) * c.double()) / d["delta"],
                        down=push * sp["vecs"][0].to(c.device).double() / silu_k))
    return out


def compile_layer(model, tokenizer, facts: list[dict], layer: int, mctx: dict, *, forms=FORMS, gain: float = GAIN, push: float = PUSH,
                  paragraphs: list[str] | None = None, extra_moments: tuple[torch.Tensor, torch.Tensor] | None = None,
                  progress: Callable[[str], None] | None = None) -> dict:
    """Compile facts into the layer's own neurons. facts: dicts with subject, relation, answer and parts (a GRAFT fact's built parts);
    mctx: the table's calibration (c, cbar); extra_moments: second moments of more ordinary text (fitting, spare; text_moments), added to
    the calibration paragraphs', with a wider ridge grid. Returns the layer's new gate, up and down weights (in the model's dtype, on the
    CPU), the neurons used and what each holds, and the re-fit's settings. The model is left unchanged."""
    say = progress or (lambda s: None)
    mlp = blocks(model)[layer].mlp
    device = mlp.down_proj.weight.device
    c, cbar = mctx["c"].to(device).float(), float(mctx["cbar"])
    specs = []
    for f in facts:
        for nrn in fact_neurons(f["parts"], c, cbar, forms, gain, push):
            specs.append(dict(nrn, subject=f["subject"], relation=f["relation"], answer=f["answer"]))
    if not specs:
        raise ValueError("none of the facts has a usable detector and answer vector to compile")
    if len(specs) > MEASURED_NEURONS:
        warnings.warn(f"{len(specs)} neurons to reuse at layer {layer}: the cost to known facts was measured up to {MEASURED_NEURONS} and "
                      "rises clearly from 512 (lab notebook, step 1c); check the exported model before relying on it", stacklevel=2)
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
    neurons = [dict(slot=int(s), subject=sp["subject"], relation=sp["relation"], answer=sp["answer"], form=sp["form"]) for s, sp in zip(slots, specs)]
    return dict(layer=layer, gate=gate.cpu(), up=up.cpu(), down=down.to(W0.dtype).cpu(), neurons=neurons, gain=gain, push=push,
                refit=dict(ridge=ridge, freed_output_left=round(left, 5)))


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


def export_grafted(base_dir: str, out_dir: str, compiled: dict, *, meta: dict | None = None) -> dict:
    """Write the grafted model: a copy of the base checkpoint in `base_dir` with the compiled layer's three matrices in place of the
    originals, plus grafted.json (which neuron holds which fact) and grafted_switch.py (the name switch). Returns a report: the edited
    tensors, how many values changed, and the byte sizes of the base and grafted weight files (equal)."""
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
    targets = _tensor_keys(keys, compiled["layer"])
    report = dict(tensors={}, files={})
    for path in files:
        with safe_open(path, "pt") as fh:
            header = fh.metadata()
            tensors = {k: fh.get_tensor(k) for k in fh.keys()}
        for part, key in targets.items():
            if key not in tensors:
                continue
            old, new = tensors[key], compiled[part]
            if old.shape != new.shape:
                raise ValueError(f"{key}: the compiled weight is {tuple(new.shape)}, the checkpoint's {tuple(old.shape)}")
            new = new.to(old.dtype).contiguous()
            report["tensors"][key] = dict(changed=int((old != new).sum()), of=old.numel())
            tensors[key] = new
        target = os.path.join(out_dir, os.path.basename(path))
        save_file(tensors, target, metadata=header)
        report["files"][os.path.basename(path)] = dict(base_bytes=os.path.getsize(path), grafted_bytes=os.path.getsize(target))
    spec = dict(method="GRAFT, grafted", layer=compiled["layer"], gain=compiled["gain"], push=compiled["push"], refit=compiled["refit"],
                written=time.strftime("%Y-%m-%d %H:%M:%S"), neurons=compiled["neurons"], **(meta or {}))
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


__all__ = ["compile_layer", "apply_layer", "export_grafted", "load_grafted", "fact_neurons", "second_moment", "redundancy_order", "compensation",
           "text_moments", "attach"]

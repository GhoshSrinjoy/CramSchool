"""Two grafted models merged into one, at the weights: no table is needed.

    from cramschool.graft import merge_grafted_models
    done = merge_grafted_models(model, tokenizer, BASE, "mine-grafted", "theirs-grafted", "both-grafted")

A grafted model holds each fact in a few of a layer's neurons: the gate row is the fact's detector and the output column its
answer. The merge reads those rows and columns out of the two checkpoints and writes them into one model. Which neurons are reused,
and how the neurons that are left are re-fitted, is decided once for all the facts together, from the base model, by the same code
one export runs (grafted.place_neurons). So for two models exported the same way, the merged checkpoint is the one a single export
of all the facts would give, tensor for tensor. The result says whether that holds (`same_as_one_export`) and why not if it doesn't.

Adding the two models' weights together would not do it: each export reuses the same neurons, since the order they are freed in is
fixed, and re-fits the others for its own count of freed neurons.

The last layer is the one place where the merge has more to do. It can spare few neurons (grafted.LAST_SHARE of them), so an export
puts there every fact's first-token neurons and as many later answer tokens as still fit, and the rest at another layer, from answer
vectors measured at that layer. Two models that fit the last layer one by one may not fit it together. The merge then shares the
layer out again as one export would (grafted.share_last_layer), and a later token that has to move is written from the vectors its
export kept beside the weights (grafted.write_later, the file grafted_later.bin). A model exported before that file existed can be
merged as long as none of its later tokens has to move.

`model` is the loaded base model: forward passes on ordinary text choose the neurons, and nothing is trained. It is left unchanged.
"""
from __future__ import annotations

import glob
import json
import os
import re
import warnings
from typing import Callable

import torch
import torch.nn.functional as F

from .core import blocks
from .grafted import (CLIP, LAST_SHARE, MEASURED_NEURONS, _tensor_keys, export_grafted, later_tokens, neuron_record, place_neurons, read_later,
                      share_last_layer)
from .grafted_switch import SPEC_FILE

# What two grafted models must have alike to be merged: the same kind of neuron does the same thing in both.
SETTINGS = dict(method=None, layer=None, gain=None, push=None, where="after", saturate=False, track=False, clip=CLIP, spell=False)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _spec(folder: str) -> dict:
    path = os.path.join(folder, SPEC_FILE)
    if not os.path.exists(path):
        raise ValueError(f"{folder} has no {SPEC_FILE}: it is not a grafted model")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _weight_files(folder: str) -> list[str]:
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.safetensors")))
    if not names:
        raise ValueError(f"{folder} has no .safetensors files: a checkpoint must be a local folder in safetensors format")
    return names


def _all_keys(folder: str) -> dict[str, str]:
    """Every tensor's name in a checkpoint folder, with the weight file that holds it."""
    from safetensors import safe_open

    out = {}
    for name in _weight_files(folder):
        with safe_open(os.path.join(folder, name), "pt") as fh:
            out.update({key: name for key in fh.keys()})
    return out


def _matrices(folder: str, layer: int, where: dict[str, str]) -> dict[str, torch.Tensor]:
    """A layer's three MLP matrices as a checkpoint folder holds them: gate, up and down."""
    from safetensors import safe_open

    out = {}
    for part, key in _tensor_keys(where, layer).items():
        with safe_open(os.path.join(folder, where[key]), "pt") as fh:
            out[part] = fh.get_tensor(key)
    return out


def _layers_of(spec: dict) -> set[int]:
    return {n.get("layer", spec["layer"]) for n in spec["neurons"]}


def _check_base(base_dir: str, folder: str, edited: set[str]) -> None:
    """Refuse a grafted model that was not exported from this base: every tensor outside its edited layers must be the base's."""
    from safetensors import safe_open

    names = _weight_files(base_dir)
    if _weight_files(folder) != names:
        raise ValueError(f"{folder} holds other weight files than the base in {base_dir}: it was exported from another model")
    for name in names:
        with safe_open(os.path.join(base_dir, name), "pt") as base, safe_open(os.path.join(folder, name), "pt") as mine:
            if sorted(base.keys()) != sorted(mine.keys()):
                raise ValueError(f"{name} in {folder} holds other tensors than the base's: it was exported from another model")
            for key in base.keys():
                if key not in edited and not torch.equal(base.get_tensor(key), mine.get_tensor(key)):
                    raise ValueError(f"{folder} was not exported from the base in {base_dir}: {key} differs")


def _check_model(model, base_dir: str, where: dict[str, str], layers: set[int]) -> None:
    """Refuse a loaded model that does not hold the base's weights at the layers the merge reads and writes."""
    for layer in sorted(layers):
        mlp = blocks(model)[layer].mlp
        for part, value in _matrices(base_dir, layer, where).items():
            held = getattr(mlp, f"{part}_proj").weight.data
            if value.shape != held.shape or not torch.equal(value.to(held.dtype), held.detach().cpu()):
                raise ValueError(f"the loaded model does not hold the base's weights at layer {layer}: load the base model in {base_dir}, "
                                 "with no weight edit kept there")


def _facts_of(spec: dict, later: dict | None, src: int) -> list[dict]:
    """The facts a grafted model holds in its weights, in the order of its table, each with its neurons. A fact with no neuron is
    not in the weights and is left out."""
    track = bool(spec.get("track"))
    found: dict[int, dict] = {}
    for n in spec["neurons"]:
        f = found.setdefault(n["fact"], dict(src=src, index=n["fact"], subject=n["subject"], relation=n["relation"], answer=n["answer"],
                                             key=(_norm(n["subject"]), n["relation"]), neurons=[], answers=None, later=None))
        f["neurons"].append(dict(n, layer=n.get("layer", spec["layer"])))
    for i, f in found.items():
        if track and i < len(spec.get("answers") or []):
            f["answers"] = spec["answers"][i]
        if later is not None and i < len(later["facts"]):
            f["later"] = later["facts"][i]
    return [found[i] for i in sorted(found)]


def _describe(f: dict) -> dict:
    return dict(subject=f["subject"], relation=f["relation"], answer=f["answer"])


def _names(specs: list[dict], merged: list[dict], taken: set[str]) -> tuple[dict, list, list]:
    """The merged model's aliases and exclusions: the first model's, and the second's for the subjects taken from it, where they
    don't clash (as Graft.merge does for two tables). Returns (aliases, exclusions, the names left out and why)."""
    stored = {_norm(f["subject"]) for f in merged}
    aliases, skipped = dict(specs[0].get("aliases") or {}), []
    theirs = specs[1].get("aliases") or {}
    for alias, subject in theirs.items():
        if subject not in taken or aliases.get(alias) == subject:
            continue
        if _norm(alias) in stored or any(_norm(a) == _norm(alias) for a in aliases):
            skipped.append(dict(subject=subject, name=alias, why="it is a stored name, or an alias the first model declares for another subject"))
            continue
        aliases[alias] = subject
    exclusions = list(specs[0].get("exclusions") or [])
    known = stored | {_norm(a) for a in aliases}
    for longer in specs[1].get("exclusions") or []:
        own = [o for s in taken for o in [_norm(s)] + [_norm(a) for a, name in theirs.items() if name == s]]
        if longer in exclusions or not any(re.search(r"(?<![0-9a-z])" + re.escape(o) + r"(?![0-9a-z])", _norm(longer)) for o in own):
            continue
        if _norm(longer) in known:
            skipped.append(dict(subject=None, name=longer, why="it is a stored name or a declared alias"))
            continue
        exclusions.append(longer)
    return aliases, exclusions, skipped


def merge_grafted_models(model, tokenizer, base_dir: str, mine: str, theirs: str, out_dir: str, *, prefer: str = "error",
                         paragraphs: list[str] | None = None, last_budget: int | None = None,
                         progress: Callable[[str], None] | None = None) -> dict:
    """Merge the grafted models in the folders `mine` and `theirs` into one grafted model in `out_dir`, at the weights.

    model, tokenizer: the loaded base model. base_dir: the local folder, in safetensors format, both were exported from.
    prefer: what to do where both hold the same subject and relation with another answer. "error" (the default) writes nothing and
    raises, naming them; "mine" keeps the first model's; "theirs" takes the second's. A fact both hold alike is kept once.
    paragraphs: the ordinary text for choosing and re-fitting the neurons, if the two were exported with text of your own (default:
    the bundled paragraphs, as an export uses). last_budget: how many neurons of the last layer may be reused (default: as an export).

    Refused, before anything is written: a folder that was not exported from this base; two models whose neurons work differently
    (another layer, or another way of compiling); a loaded model that does not hold the base's weights; a model exported before
    the later answer vectors were kept, when one of its later answer tokens would have to move to the other layer.

    Returns added, replaced, kept and same (as Graft.merge reports them for two tables), skipped_names, facts (how many the merged
    model holds), neurons (how many at each layer), last_layer (how the last layer was shared out, or None), report (the export's),
    and same_as_one_export: True when the merged checkpoint is the one a single export of all the facts gives, False with `why`
    when it is not, None when the two models don't say what text they were re-fitted on (exports of an earlier version)."""
    if prefer not in ("error", "mine", "theirs"):
        raise ValueError(f"prefer must be 'error', 'mine' or 'theirs', not {prefer!r}")
    say = progress or (lambda s: None)
    folders = (mine, theirs)
    if os.path.abspath(out_dir) in {os.path.abspath(p) for p in folders + (base_dir,)}:
        raise ValueError("the merged model needs a folder of its own: out_dir is the base's folder or one of the two models'")
    specs = [_spec(p) for p in folders]
    differs = [k for k, default in SETTINGS.items() if specs[0].get(k, default) != specs[1].get(k, default)]
    if differs:
        raise ValueError(f"these grafted models can't be merged at the weights: {', '.join(differs)} differ{'s' if len(differs) == 1 else ''}. "
                         "Their neurons work in different ways. Merge the tables they were exported from (Graft.merge_grafted).")
    spec = specs[0]
    top, gain, push = spec["layer"], spec["gain"], spec["push"]
    laters = [read_later(p) for p in folders]
    later_layers = {x for s, kept in zip(specs, laters) for x in (s.get("later_layer"), kept["layer"] if kept else None) if x is not None}
    if len(later_layers) > 1:
        raise ValueError(f"these grafted models can't be merged at the weights: their later answer tokens go to different layers ({sorted(later_layers)})")
    later_layer = later_layers.pop() if later_layers else None
    if all(laters) and not (torch.equal(laters[0]["c"], laters[1]["c"]) and laters[0]["cbar"] == laters[1]["cbar"]):
        raise ValueError("these grafted models can't be merged at the weights: they were exported with other ordinary text (the constant "
                         f"direction kept for layer {later_layer} differs). Merge the tables they were exported from (Graft.merge_grafted).")

    where = _all_keys(base_dir)
    for folder, s in zip(folders, specs):
        _check_base(base_dir, folder, {key for layer in _layers_of(s) for key in _tensor_keys(where, layer).values()})
    _check_model(model, base_dir, where, set().union(*map(_layers_of, specs)) | ({later_layer} if later_layer is not None else set()))

    ours, others = (_facts_of(s, kept, i) for i, (s, kept) in enumerate(zip(specs, laters)))
    held = {f["key"]: f for f in ours}
    new, conflicts, same = [], [], 0
    for f in others:
        if f["key"] not in held:
            new.append(f)
        elif held[f["key"]]["answer"] == f["answer"]:
            same += 1
        else:
            conflicts.append(f)
    if conflicts and prefer == "error":
        raise ValueError("both models hold these facts with another answer (prefer='mine' or 'theirs' decides): "
                         + "; ".join(f"{f['subject']} | {f['relation']}" for f in conflicts))
    replaced = conflicts if prefer == "theirs" else []
    gone = {f["key"] for f in replaced}
    merged = [f for f in ours if f["key"] not in gone] + new + replaced
    if not merged:
        raise ValueError("neither model holds a fact in its weights")

    mats: dict[tuple[int, int], dict] = {}

    def from_weights(i: int, f: dict, n: dict) -> dict:
        """One fact neuron as its checkpoint holds it: the gate row and the output column, bit for bit."""
        at = (f["src"], n["layer"])
        if at not in mats:
            mats[at] = _matrices(folders[f["src"]], n["layer"], where)
        m, s = mats[at], n["slot"]
        if bool(m["up"][s].any()):
            raise ValueError(f"{folders[f['src']]}: neuron {s} of layer {n['layer']} is not a fact neuron (its up row is not zero): "
                             f"{SPEC_FILE} does not describe these weights")
        return dict({k: v for k, v in n.items() if k not in ("slot", "layer")}, fact=i, gate=m["gate"][s].clone(), down=m["down"][:, s].clone())

    rows: dict[int, list[dict]] = {top: []}
    plan, why = None, []
    if spec.get("spell"):  # compiled at the last layer: share its budget out again, as one export of all the facts would
        width = blocks(model)[top].mlp.up_proj.weight.shape[0]
        budget = int(width * LAST_SHARE) if last_budget is None else int(last_budget)
        firsts = sum(n["kind"] == "first" and n["layer"] == top for f in merged for n in f["neurons"])
        keys = [(key[0], i, key) for i, f in enumerate(merged) for key, _ in later_tokens({k: dict(ids=v) for k, v in (f["answers"] or {}).items()})]
        here, there, plan = share_last_layer(firsts, keys, len(merged), budget)
        say(f"the last layer's budget: {plan}")
        if firsts > budget:
            warnings.warn(f"{firsts} first-token neurons at the last layer, over its budget of {budget}; a single export of this many "
                          "facts goes over it too. On Qwen3.5-0.8B the damage bar was met at 256 reused neurons of the last layer's 3,584 "
                          "and not at 384. Check the merged model before relying on it", stacklevel=2)
        silu_k = float(F.silu(torch.tensor(gain)))
        stuck = 0
        for i, f in enumerate(merged):
            elsewhere = {(n["step"], n["token"]): n for n in f["neurons"] if n["layer"] != top}
            for n in f["neurons"]:
                if n["layer"] == top and (n["kind"] != "next" or (n["step"], n["token"]) in here[i]):
                    rows[top].append(from_weights(i, f, n))
            stays = {key for key in here[i] if key in elsewhere}   # one export would have it at the last layer; its source has it elsewhere
            stuck += len(stays)
            wanted = there[i] | stays
            if not wanted:
                continue
            kept = laters[f["src"]]
            order = [(key, sp) for key, sp in later_tokens(f["later"], measured=True)] if f["later"] is not None else [(key, None) for key in elsewhere]
            for key, sp in order:
                if key not in wanted:
                    continue
                if key in elsewhere:
                    row = from_weights(i, f, elsewhere[key])
                else:  # it sat at the last layer as a sized row: at the other layer it is its measured answer vector
                    row = dict(form="any", kind="next", step=key[0], token=key[1], fact=i, subject=f["subject"], relation=f["relation"],
                               answer=f["answer"], gate=(gain / kept["cbar"]) * kept["c"].float().double(),
                               down=push * sp["vecs"][key[0]].double() / silu_k)
                rows.setdefault(later_layer, []).append(row)
            if f["later"] is None and wanted - set(elsewhere):
                raise ValueError(f"{folders[f['src']]} was exported without its later answer vectors (grafted_later.bin), and "
                                 f"{len(wanted - set(elsewhere))} later answer token(s) of {f['subject']} | {f['relation']} have to move from the "
                                 f"last layer: the two models need {plan['first'] + len(keys)} of its neurons and its budget is {budget}. Export "
                                 "that model again with this version, or merge the tables (Graft.merge_grafted)")
        if stuck:
            why.append(f"{stuck} later answer token(s) stay at layer {later_layer}, where their model has them; one export would put them at "
                       "the last layer (a model was exported with a smaller last-layer budget)")
    else:
        for i, f in enumerate(merged):
            rows[top] += [from_weights(i, f, n) for n in f["neurons"]]

    forms = [{n["form"] for n in s["neurons"] if n["kind"] == "first"} for s in specs]
    if forms[0] != forms[1]:
        why.append(f"the two models were exported with other forms ({sorted(forms[0])} and {sorted(forms[1])})")
    for layer, found in rows.items():
        if len(found) > MEASURED_NEURONS and not (spec.get("spell") and layer == top):
            warnings.warn(f"{len(found)} neurons to reuse at layer {layer}: no firmly known fact was lost up to {MEASURED_NEURONS} freed, and "
                          "one was at 1,024. Check the merged model (bench.scoring.FirmCheck) before relying on it", stacklevel=2)

    answers = [f["answers"] for f in merged] if spec.get("track") else []
    compiled = {}
    for layer in sorted((k for k, v in rows.items() if v), reverse=True):
        say(f"layer {layer}: {len(rows[layer])} fact neurons from the two models")
        placed = place_neurons(model, tokenizer, layer, rows[layer], paragraphs=paragraphs, progress=progress)
        compiled[layer] = dict(layer=layer, gate=placed["gate"], up=placed["up"], down=placed["down"],
                               neurons=[neuron_record(s, row) for s, row in zip(placed["slots"], rows[layer])], gain=gain, push=push,
                               where=spec.get("where", "after"), saturate=bool(spec.get("saturate")), track=bool(spec.get("track")),
                               clip=spec.get("clip", CLIP), answers=answers, spell=bool(spec.get("spell")) and layer == top, refit=placed["refit"])

    unknown = False
    for layer, done in compiled.items():  # the same moments mean the same neurons freed and the same re-fit as each source's export
        for s in specs:
            refit = s.get("refit") if layer == s["layer"] else s.get("later_refit") if layer == s.get("later_layer") else None
            if refit is None:
                continue
            if "moments" not in refit:
                unknown = True
            elif refit["moments"] != done["refit"]["moments"] and not any("ordinary text" in w for w in why):
                why.append("the ordinary text the neurons are re-fitted on is not the text a model was exported with")

    taken = {f["subject"] for f in new + replaced}
    aliases, exclusions, skipped = _names(specs, merged, taken)
    meta = dict(base_model=base_dir, facts=len(merged), aliases=aliases, exclusions=exclusions,
                merged_from=[dict(facts=len(side), table_facts=s.get("table_facts")) for side, s in zip((ours, others), specs)])
    keep = None
    if plan is not None:
        meta["last_budget"] = plan["budget"]
        if later_layer is not None and all(f["later"] is not None for f in merged):
            first = next(k for k in laters if k is not None)
            keep = dict(layer=later_layer, c=first["c"], cbar=first["cbar"], spellings=[f["later"] for f in merged])
    report = export_grafted(base_dir, out_dir, compiled[top], meta=meta, also=compiled.get(later_layer) if later_layer in compiled else None,
                            later=keep)
    return dict(added=[_describe(f) for f in new], replaced=[_describe(f) for f in replaced],
                kept=[_describe(held[f["key"]]) for f in conflicts] if prefer == "mine" else [], same=same, skipped_names=skipped,
                facts=len(merged), neurons={layer: len(done["neurons"]) for layer, done in sorted(compiled.items())}, last_layer=plan,
                same_as_one_export=False if why else None if unknown else True, why="; ".join(why), report=report)


__all__ = ["merge_grafted_models", "SETTINGS"]

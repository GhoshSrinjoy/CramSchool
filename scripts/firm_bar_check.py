"""The check of the firm-fact damage bar itself, then step 6's capacity judged by it (lab notebook, "The damage bar, re-founded on firmly
known facts"). On a known set that nothing has used:
  V1  the unedited model at other batchings must lose no firmly known fact;
  V2  damaged models must fail the bar: 8 and 16 directions silenced in the whole model, 2,048 of layer 20's neurons switched off;
  V3  the shipped size must pass: 64 neurons freed with the shipped method.
Reported with no rule: 256, 512 and 1,024 neurons freed. Then the 237-fact checkpoint of step 6, without and with its switch.
Forward passes and linear solves only. GPU, about 20 minutes.

Usage: python scripts/firm_bar_check.py <model folder> <new known json> <grafted folder> <out folder>
"""
import glob
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.scoring import FIRM_MARGIN, FIRM_SHIFT, FirmCheck, make_baseline
from cramschool.graft import grafted, grafted_switch
from cramschool.graft.core import CALIBRATION_FIT, blocks, stack_of
from cramschool.graft.wordings import calibration_paragraphs, known_facts
from cramschool.model_access import batch_ids

LAYER, DRIFT_BAR = 20, 0.05
SETTINGS = [(1, False), (8, False), (16, False), (32, False), (64, False), (16, True), (64, True)]  # (batch size, reversed order)
t0 = time.time()


def log(s):
    print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)


@torch.no_grad()
def pooled_moment(model, tok, texts):
    """Uncentred second moment of the residual stream entering every layer, each token scaled to unit rms, pooled over the layers."""
    layers = blocks(model)
    d = stack_of(model).embed_tokens.weight.shape[1]
    M = torch.zeros(d, d, dtype=torch.float64, device=model.device)
    cur, n = {}, 0

    def grab(i):
        def pre(m, args, kwargs):
            cur[i] = kwargs.get("hidden_states", args[0] if args else None)
        return pre

    hs = [b.register_forward_pre_hook(grab(i), with_kwargs=True) for i, b in enumerate(layers)]
    seqs = [s for s in (tok(t, truncation=True, max_length=128, add_special_tokens=False).input_ids for t in texts) if len(s) > 1]
    try:
        for start in range(0, len(seqs), 20):
            ids, mask = batch_ids(tok, seqs[start:start + 20], model.device)
            model(input_ids=ids, attention_mask=mask)
            for i in range(len(layers)):
                x = cur[i][mask.bool()].double()
                x = x / x.pow(2).mean(-1, keepdim=True).sqrt()
                M += x.T @ x
                n += x.shape[0]
    finally:
        for h in hs:
            h.remove()
    return M / n


def silence(model, V):
    """Hooks that take the directions V out of every write to the residual stream: the embedding, each layer's token mixer and MLP."""
    def proj(o):
        of = o.float()
        return (of - (of @ V) @ V.T).to(o.dtype)

    def hook(m, a, o):
        return (proj(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else proj(o)

    hs = [stack_of(model).embed_tokens.register_forward_hook(hook)]
    for b in blocks(model):
        mixer = b.linear_attn if hasattr(b, "linear_attn") else b.self_attn
        hs += [mixer.register_forward_hook(hook), b.mlp.register_forward_hook(hook)]
    return hs


def main():
    model_dir, known_path, graft_dir, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    with open(known_path, encoding="utf-8") as fh:
        new_items = [(r["prompt"], r["answers"]) for r in json.load(fh)["cf_known"]]
    baseline = make_baseline(model, tok)
    firm_new = FirmCheck(model, tok, new_items, max_new_tokens=8, batch=64)
    firm_133 = FirmCheck(model, tok, known_facts(), max_new_tokens=12, batch=16)
    log(f"unedited: the new set: {sum(firm_new.before)} of {len(new_items)} answered, {sum(firm_new.firm)} firmly (margin >= {FIRM_MARGIN}); the 133: "
        f"{sum(firm_133.before)} answered, {sum(firm_133.firm)} firmly")
    res = dict(known=known_path, firm_margin=FIRM_MARGIN, firm_shift=FIRM_SHIFT, drift_bar=DRIFT_BAR,
               new=dict(n=len(new_items), answered=sum(firm_new.before), firm=sum(firm_new.firm)),
               the133=dict(answered=sum(firm_133.before), firm=sum(firm_133.firm)), noise=[], models=[])

    def save():
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)

    # ---- V1: arithmetic alone
    for batch, rev in SETTINGS:
        row = dict(batch=batch, reversed=rev)
        for name, chk in (("new", firm_new), ("the133", firm_133)):
            now = chk.correct(batch, rev)
            row[name] = dict(firm_lost=sum(f and not n for f, n in zip(chk.firm, now)),
                             knife_lost=sum(b and not f and not n for b, f, n in zip(chk.before, chk.firm, now)),
                             gained=sum(n and not b for n, b in zip(now, chk.before)),
                             firm_lost_items=[chk.items[i][0] for i, (f, n) in enumerate(zip(chk.firm, now)) if f and not n])
        res["noise"].append(row)
        log(f"V1 | unedited, batch {batch:>2}{' reversed' if rev else '':<9}: the new set loses {row['new']['firm_lost']} firm and "
            f"{row['new']['knife_lost']} knife-edge facts (gains {row['new']['gained']}); the 133: {row['the133']['firm_lost']} firm, "
            f"{row['the133']['knife_lost']} knife-edge")
        save()
    v1 = all(r["new"]["firm_lost"] == 0 for r in res["noise"])
    log(f"V1 {'holds' if v1 else 'FAILS'}: firm facts lost to batching alone, at most {max(r['new']['firm_lost'] for r in res['noise'])}")

    def score(name, note=""):
        drift, _ = baseline.drift()
        known_now = baseline.known_check.correct()
        row = dict(model=name, drift=drift, known_lost=sum(b and not a for b, a in zip(baseline.known_check.before, known_now)),
                   new=firm_new.measure(), the133=firm_133.measure())
        row["passes"] = bool(row["new"]["passed"] and drift <= DRIFT_BAR)
        res["models"].append(row)
        n = row["new"]
        log(f"{name}{note}: {'PASSES' if row['passes'] else 'fails'} | the new set: firm lost {n['firm_lost']} of {n['firm']} "
            f"(unanswered {n['firm_unanswered']}, overtaken {n['firm_overtaken']}), mean margin change {n['firm_shift']:+.3f}, smallest firm margin "
            f"{n['firm_min']:.2f}; knife-edge lost {n['knife_lost']} of {n['knife']}, gained {n['gained']} | the 133: firm lost "
            f"{row['the133']['firm_lost']} of {row['the133']['firm']}, by the old count {row['known_lost']} | drift {drift:.5f}")
        save()
        return row

    # ---- V2: silenced directions
    paras = calibration_paragraphs()
    fit = [paras[i] for i in CALIBRATION_FIT]
    ev, V = torch.linalg.eigh(pooled_moment(model, tok, fit))
    for k in (8, 16):
        hs = silence(model, V[:, :k].float())
        try:
            score(f"silenced_{k}", f" ({k} quietest directions out of every write to the stream)")
        finally:
            for h in hs:
                h.remove()

    # ---- neurons of layer 20
    mlp = blocks(model)[LAYER].mlp
    W0 = mlp.down_proj.weight.data.clone()
    Hf = grafted.second_moment(model, tok, LAYER, fit)
    Hv = grafted.second_moment(model, tok, LAYER, [paras[i] for i in grafted.SPARE])
    order = grafted.redundancy_order(Hf, W0.double(), 2048)
    down = W0.clone()
    down[:, order] = 0
    mlp.down_proj.weight.data.copy_(down)
    score("off_2048", " (2,048 neurons switched off, no re-fit)")
    mlp.down_proj.weight.data.copy_(W0)
    for n in (64, 256, 512, 1024):
        K, dW, ridge, left = grafted.compensation(Hf, Hv, W0.double(), order[:n])
        down = W0.double().clone()
        down[:, order[:n]] = 0
        down[:, K] += dW.to(down.device)
        mlp.down_proj.weight.data.copy_(down.to(W0.dtype))
        score(f"freed_{n}", f" ({n} neurons freed, the rest re-fitted; {left:.0%} of their output left)")
        mlp.down_proj.weight.data.copy_(W0)
    del Hf, Hv

    # ---- the 237-fact checkpoint of step 6: its three matrices written into the loaded model
    with open(os.path.join(graft_dir, grafted_switch.SPEC_FILE), encoding="utf-8") as fh:
        spec = json.load(fh)
    gmlp = blocks(model)[spec["layer"]].mlp
    original = {p: getattr(gmlp, f"{p}_proj").weight.data.clone() for p in ("gate", "up", "down")}
    files = sorted(glob.glob(os.path.join(graft_dir, "*.safetensors")))
    keys = []
    for path in files:
        with safe_open(path, "pt") as fh:
            keys += list(fh.keys())
    targets = grafted._tensor_keys(keys, spec["layer"])
    for path in files:
        with safe_open(path, "pt") as fh:
            for part, key in targets.items():
                if key in fh.keys():
                    w = getattr(gmlp, f"{part}_proj").weight
                    w.data.copy_(fh.get_tensor(key).to(w.device, w.dtype))
    subjects = {" ".join(n["subject"].lower().split()) for n in spec["neurons"]}
    named = sum(any(s in " ".join(p.lower().split()) for s in subjects) for p, _ in new_items)
    log(f"the checkpoint: {len(spec['neurons'])} fact neurons at layer {spec['layer']}; prompts of the new set that name an edited subject: {named}")
    res["checkpoint"] = dict(folder=graft_dir, neurons=len(spec["neurons"]), new_prompts_naming_an_edited_subject=named)
    score("grafted_237_no_switch")
    switch = grafted_switch.attach(model, tok, graft_dir)
    score("grafted_237_with_switch")
    switch.detach()
    for p, w in original.items():
        getattr(gmlp, f"{p}_proj").weight.data.copy_(w)

    by = {r["model"]: r for r in res["models"]}
    v2 = not any(by[m]["passes"] for m in ("silenced_8", "silenced_16", "off_2048"))
    v3 = by["freed_64"]["passes"]
    adopted = bool(v1 and v2 and v3)
    capacity = bool(by["grafted_237_no_switch"]["passes"] and by["grafted_237_with_switch"]["passes"])
    res["verdict"] = dict(v1=v1, v2=v2, v3=v3, adopted=adopted, capacity_237=bool(adopted and capacity))
    log(f"V1 {'holds' if v1 else 'FAILS'} | V2 {'holds' if v2 else 'FAILS'} | V3 {'holds' if v3 else 'FAILS'} | the firm bar is "
        f"{'ADOPTED' if adopted else 'NOT adopted'} | the 237-fact checkpoint {'passes' if capacity else 'does not pass'} it")
    save()


if __name__ == "__main__":
    main()

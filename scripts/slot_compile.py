"""Step 2 of the grafted model: CounterFact edits in layer 20's own neurons (lab notebook, "Step 2: the facts in layer 20's own neurons").

64 edits of the editor run, one neuron each, in step 1c's 64 slots (the redundancy order, the freed neurons' work compensated). A slot is
an AND gate made from the edit's GRAFT parts: gate = the plain question detector, k (w - (tau/cbar) c) / delta; down = the push,
2 v / silu(k); up = the subject, either from the name index through a hook (B, the hybrid) or from a subject reading in the weights (C, the
grafted model, no hook). The inserted layer on the same edits (A) is the reference, scored in the same run.

Usage:
  python scripts/slot_compile.py --model <folder> --run runs/counterfact/phrasing_editor \\
      --relations research/results/counterfact/phrasing/editor_relations.json --twins research/results/counterfact/full/twins_narrow.json \\
      --data counterfact.json --slots research/results/slots/03_compensate_layer20.json --out runs/slots/step2
"""
import argparse
import json
import os
import random
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_slot_compensate import SPARE, ScreenCheck, compensation, second_moment  # noqa: E402

from cramschool.bench.counterfact import load_counterfact, logprobs, measures, replies, run_edits, summarize  # noqa: E402
from cramschool.bench.methods import calibrate  # noqa: E402
from cramschool.bench.scoring import make_baseline  # noqa: E402
from cramschool.facts import PREFIXES  # noqa: E402
from cramschool.graft.build import build_context, fact_detectors, guarded, kind_parts, plain_items  # noqa: E402
from cramschool.graft.core import CALIBRATION_FIT, blocks, last_inputs, lda, ratio  # noqa: E402
from cramschool.graft.layer import NameIndex  # noqa: E402
from cramschool.graft.wordings import Wordings, calibration_paragraphs, norm  # noqa: E402
from cramschool.model_access import Prompt  # noqa: E402

LAYER, N_EDITS, GAIN, PUSH = 20, 64, 8.0, 2.0
N_TEMPLATES, N_INVENTED, N_REAL, SEED = 16, 12, 24, 0
GROUPS = ("rewrite", "para", "neigh")


def log(s):
    print(s, flush=True)


def parse():
    ap = argparse.ArgumentParser()
    for k in ("model", "run", "relations", "twins", "data", "slots", "out"):
        ap.add_argument(f"--{k}", required=True)
    return ap.parse_args()


def choose(facts, parts):
    """3 edits per relation and one more, round-robin over the relations in id order, among edits with a plain question detector."""
    by_rel, rels = {}, []
    for f in facts:
        if any(d["form"] == "plain" for d in fact_detectors(parts[f["id"]])):
            if f["relation"] not in by_rel:
                rels.append(f["relation"])
            by_rel.setdefault(f["relation"], []).append(f)
    chosen, i = [], 0
    while len(chosen) < N_EDITS:
        for r in rels:
            if i < len(by_rel[r]) and len(chosen) < N_EDITS:
                chosen.append(by_rel[r][i])
        i += 1
    pos = {f["id"]: k for k, f in enumerate(facts)}
    return sorted(chosen, key=lambda f: pos[f["id"]])


class SubjectGate:
    """B's key 1: the slot's up reading times 1 when the slot's subject is the last name read (the inserted layer's name index), else 0."""

    def __init__(self, model, tok, mlp, slots, subjects):
        names = sorted(set(subjects))
        self.index = NameIndex(tok, names)
        company = {n: self.index.lookup[" ".join(n.lower().split())] for n in names}
        dev = next(model.parameters()).device
        self.slots = torch.tensor(slots, device=dev)
        self.slot_company = torch.tensor([company[s] for s in subjects], device=dev)
        self.n, self.cur = None, None
        self.handles = [model.register_forward_pre_hook(self._top, with_kwargs=True), mlp.up_proj.register_forward_hook(self._up)]

    def _top(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        n, t = ids.shape
        if t > 1 or self.n != n:
            self.index.reset(n)
            self.n = n
        valid = (mask[:, -t:].bool() if mask is not None else torch.ones_like(ids, dtype=torch.bool)).tolist()
        idl, cur = ids.tolist(), []
        for b in range(n):
            c, row = self.index.rows[b]["current"], []
            for i in range(t):
                if valid[b][i]:
                    c = self.index.step(b, idl[b][i])
                row.append(-1 if c is None else c)
            cur.append(row)
        self.cur = torch.tensor(cur, device=ids.device)

    def _up(self, module, args, output):
        ind = (self.cur[:, -output.shape[1]:, None] == self.slot_company[None, None, :]).to(output.dtype)
        out = output.clone()
        out[..., self.slots] = out[..., self.slots] * ind
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


def score(model, tok, facts, before, baseline, cf_check):
    t0 = time.time()
    now, lp = replies(model, tok, facts), logprobs(model, tok, facts)
    rows = [dict(id=f["id"], relation=f["relation"], subject=f["subject"], measures=measures(f, lp[f["id"]], now[f["id"]], before[f["id"]]),
                 replies=now[f["id"]], logprobs=lp[f["id"]]) for f in facts]
    lost = sum(b and not a for b, a in zip(baseline.known_check.before, baseline.known_check.correct()))
    lost_cf = sum(b and not a for b, a in zip(cf_check.before, cf_check.correct()))
    drift = baseline.drift()[0]
    s = summarize(rows, lost, drift)
    s["cf_lost"] = lost_cf
    log(f"    scored in {time.time() - t0:.0f}s")
    return rows, s


def line(name, s):
    return (f"{name}: ES {s['es']:.3f} PS {s['ps']:.3f} NS {s['ns']:.3f} | says new: rewrite {s['gen_rewrite']:.3f}, reworded {s['gen_para']:.3f} | "
            f"leaks: neighbours {s['neigh_leak']:.3f}, same subject {s['same_leak']:.3f} | known lost {s['known_lost']} of 133"
            + (f", {s['cf_lost']} of the clean 391" if "cf_lost" in s else "") + f", drift {s['drift']:.4f}")


def main():
    a = parse()
    os.makedirs(os.path.join(a.out, "addon", "parts"), exist_ok=True)
    t0 = time.time()
    with open(os.path.join(a.run, "facts.json"), encoding="utf-8") as fh:
        facts_all = json.load(fh)
    parts = {f["id"]: torch.load(os.path.join(a.run, "parts", f"{f['id']}.pt"), weights_only=True) for f in facts_all}
    facts = choose(facts_all, parts)
    for f in facts:
        shutil.copy(os.path.join(a.run, "parts", f"{f['id']}.pt"), os.path.join(a.out, "addon", "parts", f"{f['id']}.pt"))
    with open(os.path.join(a.out, "facts.json"), "w", encoding="utf-8") as fh:
        json.dump(facts, fh, ensure_ascii=False, indent=1)
    wordings = Wordings.builtin().with_file(a.relations)
    with open(a.relations, encoding="utf-8") as fh:
        cf_ids = {v: k for k, v in json.load(fh)["counterfact"].items()}
    with open(a.twins, encoding="utf-8") as fh:
        twins = json.load(fh)["twins"]
    with open(a.slots, encoding="utf-8") as fh:
        step1c = json.load(fh)
    slots = step1c["redundancy_order"][:N_EDITS]
    log(f"{len(facts)} edits over {len({f['relation'] for f in facts})} relations; slots: the first {len(slots)} of the redundancy order")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    dev = model.device

    # ---- A: the inserted layer, the reference (it scores the unedited model first)
    log("A: the inserted layer on the same edits")
    rows_a, s_a, s0 = run_edits(model, tok, facts, wordings, os.path.join(a.out, "addon"), layer=LAYER, push=PUSH, margin=0.5,
                                real_world="other_relations", context=4, fast=True, twins=twins, log=log)
    before = {r["id"]: r["replies_before"] for r in rows_a}
    log(line("unedited", s0))
    log(line("A, inserted layer", s_a) + f" | {time.time() - t0:.0f}s")

    baseline = make_baseline(model, tok)
    cf_check = ScreenCheck(model, tok, items=[(r["prompt"], r["answers"]) for r in step1c["cf_known"]])
    mctx = calibrate(model, tok, LAYER)
    c, cbar, cov = mctx["c"].to(dev).double(), mctx["cbar"], mctx["cov"].to(dev).float()

    # ---- the detectors' z, and later C's subject reading, at every scored prompt's last token (layer 20's input: no slot changes it)
    scored = [(f["id"], g, i, p) for f in facts for g, ps in (("rewrite", [f["cf"]["rewrite"]]), ("para", f["cf"]["paraphrases"]),
                                                               ("neigh", f["cf"]["neighbours"])) for i, p in enumerate(ps)]
    x_scored = last_inputs(model, tok, [Prompt(p) for _, _, _, p in scored], LAYER).float()
    det = {f["id"]: next(d for d in fact_detectors(parts[f["id"]]) if d["form"] == "plain") for f in facts}
    fid_of = [s[0] for s in scored]
    z_scored = torch.tensor([float((ratio(x[None], det[fid]["w"].to(dev).float(), c.float(), cbar)[0] - det[fid]["tau"]) / det[fid]["delta"])
                             for x, fid in zip(x_scored, fid_of)])

    # ---- the freed slots and the compensation (step 1c, redundancy order, N = 64)
    mlp = blocks(model)[LAYER].mlp
    paras = calibration_paragraphs()
    Hf, _, _ = second_moment(model, tok, mlp.down_proj, [paras[i] for i in CALIBRATION_FIT])
    Hv, _, _ = second_moment(model, tok, mlp.down_proj, [paras[i] for i in SPARE])
    G0, U0, D0 = mlp.gate_proj.weight.data.clone(), mlp.up_proj.weight.data.clone(), mlp.down_proj.weight.data.clone()
    K, D, ridge, left, _ = compensation(Hf, Hv, D0.double(), slots)
    del Hf, Hv
    freed = D0.double()
    freed[:, slots] = 0
    freed[:, K] += D
    log(f"compensation: ridge {ridge:g}, lost output energy left {left:.1%} | {time.time() - t0:.0f}s")

    # ---- the slot neurons: gate = question detector, down = push
    silu_k = float(F.silu(torch.tensor(GAIN)))
    gate = G0.clone()
    down = freed.clone()
    for j, f in enumerate(facts):
        d = det[f["id"]]
        gate[slots[j]] = (GAIN * (d["w"].to(dev).double() - (d["tau"] / cbar) * c) / d["delta"]).to(G0.dtype)
        sp = parts[f["id"]]["spellings"]["spaced"] or parts[f["id"]]["spellings"]["bare"]
        down[:, slots[j]] = PUSH * sp["vecs"][0].to(dev).double() / silu_k
    down = down.to(D0.dtype)
    slot_of = {f["id"]: slots[j] for j, f in enumerate(facts)}
    g_stored = (x_scored * gate[[slot_of[fid] for fid in fid_of]].float()).sum(-1).cpu()
    g_err = (g_stored / GAIN - z_scored * (x_scored @ c.float() / cbar).cpu()).abs()
    log(f"gate rows in bfloat16: largest error in z at the scored prompts {float(g_err.max()):.4f}, median {float(g_err.median()):.5f}")

    out = dict(edits=[f["id"] for f in facts], slots=slots, compensation=dict(ridge=ridge, energy_left=left), unedited=s0, A=s_a)
    try:
        # ---- the freed slots alone
        mlp.down_proj.weight.data.copy_(freed.to(D0.dtype))
        lost = sum(b and not a for b, a in zip(baseline.known_check.before, baseline.known_check.correct()))
        lost_cf = sum(b and not a for b, a in zip(cf_check.before, cf_check.correct()))
        out["freed_only"] = dict(known_lost=lost, cf_lost=lost_cf, drift=baseline.drift()[0])
        log(f"freed slots alone: known lost {lost} of 133, {lost_cf} of the clean 391, drift {out['freed_only']['drift']:.5f}")

        # ---- B: the hybrid (key 1 from the name index)
        up = U0.clone()
        up[slots] = (c / cbar).to(U0.dtype)
        mlp.gate_proj.weight.data.copy_(gate)
        mlp.up_proj.weight.data.copy_(up)
        mlp.down_proj.weight.data.copy_(down)
        hook = SubjectGate(model, tok, mlp, slots, [f["subject"] for f in facts])
        try:
            log("B: the hybrid")
            rows_b, s_b = score(model, tok, facts, before, baseline, cf_check)
        finally:
            hook.remove()
        log(line("B, hybrid", s_b) + f" | {time.time() - t0:.0f}s")

        # ---- C's subject readings: LDA at layer 20, the subject's questions against the same questions about other subjects
        mlp.gate_proj.weight.data.copy_(G0)
        mlp.up_proj.weight.data.copy_(U0)
        mlp.down_proj.weight.data.copy_(D0)
        guarded_w = Wordings(relations=wordings.relations, generic=wordings.generic, kinds=wordings.kinds,
                             guard=set(wordings.guard) | {norm(p) for f in facts for p in f["cf"]["neighbours"] + f["cf"].get("twin_prompts", [])})
        ctx = build_context(facts, guarded_w, name="counterfact", real_world="other_relations", context=4, twins=twins)
        tests = " | ".join(p.lower() for f in facts for p in [f["cf"]["rewrite"]] + f["cf"]["paraphrases"] + f["cf"]["neighbours"]
                           + f["cf"].get("twin_prompts", []) + f["test"]["same_plain"])
        records = load_counterfact(a.data)
        chosen_subjects = {f["subject"].lower() for f in facts}
        rows_up, readings = {}, {}
        for rel in sorted({f["relation"] for f in facts}):
            fs = [f for f in facts if f["relation"] == rel]
            templates = list(dict.fromkeys(t for f in fs for t in f["train"]["templates_plain"])) + list(ctx["drafted"][rel][0][:N_TEMPLATES])
            inv = kind_parts(ctx, rel)[0][:N_INVENTED]
            pool = sorted({r["requested_rewrite"]["subject"] for r in records if r["requested_rewrite"]["relation_id"] == cf_ids[rel]})
            pool = [s for s in pool if s.lower() not in tests and s.lower() not in chosen_subjects]
            real = random.Random(SEED).sample(pool, min(N_REAL, len(pool)))
            x_inv = last_inputs(model, tok, guarded(plain_items(templates, inv), ctx["guard"]), LAYER).float()
            x_real = last_inputs(model, tok, guarded(plain_items(templates, real), ctx["guard"]), LAYER).float()
            for f in fs:
                x_pos = last_inputs(model, tok, guarded(plain_items(templates, [f["subject"]], PREFIXES + ctx.get("context", ())), ctx["guard"]),
                                    LAYER).float()
                w = lda(x_pos, [x_inv, x_real], cov)
                rho_pos = ratio(x_pos, w, c.float(), cbar)
                rho_neg = ratio(torch.cat([x_inv, x_real]), w, c.float(), cbar)
                b1, b0 = float(rho_pos.median()), float(torch.quantile(rho_neg, 0.99))
                rows_up[f["id"]] = ((w.double() - b0 * c / cbar) / (b1 - b0)).to(U0.dtype)
                mine = [k for k, fid in enumerate(fid_of) if fid == f["id"]]
                u = ((ratio(x_scored[mine], w, c.float(), cbar) - b0) / (b1 - b0)).tolist()
                readings[f["id"]] = dict(positives=len(x_pos), invented=len(x_inv), real=len(x_real), real_subjects=real,
                                         separation=round((b1 - b0) / float(rho_neg.std()), 2), u={g: [] for g in GROUPS})
                for k, val in zip(mine, u):
                    readings[f["id"]]["u"][scored[k][1]].append(round(val, 3))
            log(f"  subject readings, {rel}: " + "; ".join(
                f"{f['subject'][:20]}: reworded u {sum(readings[f['id']]['u']['para']) / len(readings[f['id']]['u']['para']):.2f}, neighbours' largest u "
                f"{max(readings[f['id']]['u']['neigh']):.2f}" for f in fs) + f" | {time.time() - t0:.0f}s")

        # ---- C: the grafted model, no hook
        up = U0.clone()
        for j, f in enumerate(facts):
            up[slots[j]] = rows_up[f["id"]]
        mlp.gate_proj.weight.data.copy_(gate)
        mlp.up_proj.weight.data.copy_(up)
        mlp.down_proj.weight.data.copy_(down)
        log("C: the grafted model")
        rows_c, s_c = score(model, tok, facts, before, baseline, cf_check)
        log(line("C, grafted", s_c) + f" | {time.time() - t0:.0f}s")
    finally:
        mlp.gate_proj.weight.data.copy_(G0)
        mlp.up_proj.weight.data.copy_(U0)
        mlp.down_proj.weight.data.copy_(D0)

    # ---- decision by decision
    lp = {name: {r["id"]: r["logprobs"] for r in rows} for name, rows in (("A", rows_a), ("B", rows_b), ("C", rows_c))}
    decisions = []
    for k, (fid, g, i, p) in enumerate(scored):
        win = {name: bool(lp[name][fid][g][i][0] > lp[name][fid][g][i][1]) for name in lp}
        decisions.append(dict(id=fid, group=g, i=i, prompt=p, z=round(float(z_scored[k]), 3), u=readings[fid]["u"][g][i], **win))
    agree = {}
    for name in ("B", "C"):
        for g in GROUPS:
            ds = [d for d in decisions if d["group"] == g]
            agree[f"{name}_{g}"] = round(sum(d[name] == d["A"] for d in ds) / len(ds), 4)
    log("decisions agreeing with A: " + ", ".join(f"{k} {v:.3f}" for k, v in agree.items()))
    lost_b = [d for d in decisions if d["group"] != "neigh" and d["A"] and not d["B"]]
    log(f"edits' prompts A wins and B loses: {len(lost_b)}; their z: " + ", ".join(f"{d['z']:.2f}" for d in lost_b[:20]))

    def ok_vs_a(s):
        return dict(es=s["es"] >= s_a["es"] - 0.05, ps=s["ps"] >= s_a["ps"] - 0.05, ns=s["ns"] >= s_a["ns"] - 0.01,
                    neigh_leak=s["neigh_leak"] <= 0.05, same_leak=s["same_leak"] <= 0.10)
    vb, vc = ok_vs_a(s_b), ok_vs_a(s_c)
    vc.update(known_lost=s_c["known_lost"] <= 2, drift=s_c["drift"] <= 0.05, cf_lost=s_c["cf_lost"] <= 7)
    out.update(B=s_b, C=s_c, agree=agree, verdict_B=vb, verdict_C=vc, B_reproduces=all(vb.values()), C_grafted=all(vc.values()),
               readings=readings)
    log(f"B reproduces the inserted layer: {all(vb.values())} {vb}")
    log(f"C is a grafted model as good as the inserted layer: {all(vc.values())} {vc}")
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(a.out, "decisions.jsonl"), "w", encoding="utf-8") as fh:
        for d in decisions:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    for name, rows in (("B", rows_b), ("C", rows_c)):
        with open(os.path.join(a.out, f"edits_{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

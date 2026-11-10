"""Step 3, revision 2, of the grafted model: the edit written where the name is read (lab notebook, "Step 3, revision 2").

Each edit's vector delta is written at the subject's last name token at layer 8's MLP output, solved in closed form so that the model's own
attention carries it to the question's end as the inserted layer's push (2 v): delta = argmin sum_P |T_p delta - 2v|^2 +
lam sum_Q |T_q delta|^2 + mu |delta|^2, with T the measured map from layer 8 at the name to the residual after layer 20 at the last token,
P the relation's training wordings about the subject and Q the subject's other questions. Arm D2 writes delta by a hook at the exact name;
arm E2 by name neurons at layer 8 (no hook). No slots at layer 20. Step 2's 64 edits and reference A.

Usage: as scripts/slot_names.py (no --slots needed beyond the clean known-fact set it holds).
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_slot_compensate import SPARE, ScreenCheck, compensation, redundancy_order, second_moment  # noqa: E402
from slot_compile import line, score  # noqa: E402
from slot_names import EventIndex, all_reads, name_positions  # noqa: E402
from slot_names_rev1 import TagHook  # noqa: E402

from cramschool.bench.methods import calibrate  # noqa: E402
from cramschool.bench.scoring import make_baseline  # noqa: E402
from cramschool.facts import PREFIXES  # noqa: E402
from cramschool.graft.build import build_context, chat_items, guarded, plain_items  # noqa: E402
from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_HELD, CALIBRATION_MAX_TOKENS, RIDGE, blocks, ratio  # noqa: E402
from cramschool.graft.wordings import Wordings, calibration_paragraphs, norm  # noqa: E402
from cramschool.model_access import Prompt, batch_ids, encode, token_index_of  # noqa: E402

NAME_LAYER, PUSH_LAYER = 8, 20
K_NAME, Z_MED = 8.0, 2.0
N_P, N_Q, LAM, MU_REL, PUSH = 8, 8, 1.0, 1e-2, 2.0
SCALES, WIN_MIN = (1.0, 1.5, 2.0, 3.0), 7
N_TEMPLATES, N_CHAT, N_PARAS_NAME, SEED = 16, 8, 120, 0


def log(s):
    print(s, flush=True)


def parse():
    ap = argparse.ArgumentParser()
    for k in ("model", "run", "relations", "twins", "slots", "step2", "out"):
        ap.add_argument(f"--{k}", required=True)
    return ap.parse_args()


@torch.no_grad()
def normal_equations(model, tok, items, unit, dev):
    """For (ids, name position) prompts: sum of T^T T and the per-prompt maps' transposes applied later; T maps a vector added to layer 8's
    MLP output at the name to the change in block 20's output at the last token. Returns the list of maps (out x in), float32."""
    hidden = model.config.get_text_config().hidden_size
    state = {}

    def add(m, a, out):
        out = out.clone()
        out[:, state["pos"]] += state["add"].to(out.dtype)
        return out

    def read(m, a, out):
        h = out[0] if isinstance(out, tuple) else out
        state["read"] = h[:, -1].float()

    h1 = blocks(model)[NAME_LAYER].mlp.register_forward_hook(add)
    h2 = blocks(model)[PUSH_LAYER].register_forward_hook(read)
    maps = []
    try:
        adds = torch.cat([torch.zeros(1, hidden, device=dev), torch.eye(hidden, device=dev) * unit])
        for ids, pos in items:
            reads = []
            for start in range(0, adds.shape[0], 256):
                state["add"], state["pos"] = adds[start:start + 256], pos
                model(input_ids=torch.tensor([ids] * state["add"].shape[0], device=dev), use_cache=False)
                reads.append(state["read"])
            r = torch.cat(reads)
            maps.append(((r[1:] - r[0]) / unit).T)
    finally:
        h1.remove()
        h2.remove()
    return maps


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(a.step2, "facts.json"), encoding="utf-8") as fh:
        facts = json.load(fh)
    with open(os.path.join(a.step2, "summary.json"), encoding="utf-8") as fh:
        step2 = json.load(fh)
    s_a = step2["A"]
    with open(os.path.join(a.step2, "addon_edits.jsonl"), encoding="utf-8") as fh:
        before = {r["id"]: r["replies_before"] for r in (json.loads(x) for x in fh if x.strip())}
    parts = {f["id"]: torch.load(os.path.join(a.run, "parts", f"{f['id']}.pt"), weights_only=True) for f in facts}
    wordings = Wordings.builtin().with_file(a.relations)
    with open(a.twins, encoding="utf-8") as fh:
        twins = json.load(fh)["twins"]
    with open(a.slots, encoding="utf-8") as fh:
        step1c = json.load(fh)
    subjects = [f["subject"] for f in facts]
    n = len(facts)
    log(f"{n} edits (step 2's); A, the inserted layer: ES {s_a['es']:.3f} PS {s_a['ps']:.3f} NS {s_a['ns']:.3f}")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    dev = model.device
    baseline = make_baseline(model, tok)
    cf_check = ScreenCheck(model, tok, items=[(r["prompt"], r["answers"]) for r in step1c["cf_known"]])
    paras = calibration_paragraphs()
    mL = calibrate(model, tok, NAME_LAYER)
    cL, cbarL, covL = mL["c"].to(dev).float(), mL["cbar"], mL["cov"].to(dev).float()

    guarded_w = Wordings(relations=wordings.relations, generic=wordings.generic, kinds=wordings.kinds,
                         guard=set(wordings.guard) | {norm(p) for f in facts for p in f["cf"]["neighbours"] + f["cf"].get("twin_prompts", [])})
    ctx = build_context(facts, guarded_w, name="counterfact", real_world="other_relations", context=4, twins=twins)
    templates = {}
    for rel in sorted({f["relation"] for f in facts}):
        fs = [f for f in facts if f["relation"] == rel]
        templates[rel] = list(dict.fromkeys(t for f in fs for t in f["train"]["templates_plain"])) + list(ctx["drafted"][rel][0][:N_TEMPLATES])
    plain_of = {f["id"]: guarded(plain_items(templates[f["relation"]], [f["subject"]], PREFIXES + ctx.get("context", ())), ctx["guard"]) for f in facts}
    same_of = {f["id"]: [Prompt(p) for p in f["train"]["same_plain"]] for f in facts}
    chat_of = {f["id"]: guarded(chat_items(tok, ctx["drafted"][f["relation"]][1][:N_CHAT], [f["subject"]]), ctx["guard"]) for f in facts}

    # ---- the unit: median of layer 8's MLP output norm at the names (the edits' rewrite prompts)
    seen = {}
    hn = blocks(model)[NAME_LAYER].mlp.register_forward_hook(lambda m, a_, out: seen.__setitem__("o", out[0].float()))
    norms = []
    try:
        with torch.no_grad():
            for f in facts:
                p = Prompt(f["cf"]["rewrite"])
                model(input_ids=torch.tensor([encode(tok, p)], device=dev), use_cache=False)
                norms.append(float(seen["o"][token_index_of(tok, p, f["subject"])].norm()))
    finally:
        hn.remove()
    unit = float(torch.tensor(norms).median())
    log(f"unit {unit:.2f} (layer {NAME_LAYER}'s MLP output at the names) | {time.time() - t0:.0f}s")

    # ---- each edit's delta, in closed form
    rng = random.Random(SEED)
    deltas, solve_stats = [], []
    for j, f in enumerate(facts):
        own = Prompt(f["cf"]["rewrite"])
        pool = [p for p in plain_of[f["id"]] if p.text != own.text]
        P = [own] + rng.sample(pool, min(N_P - 1, len(pool)))
        Q = rng.sample(same_of[f["id"]], min(N_Q, len(same_of[f["id"]])))
        items = lambda ps: [(encode(tok, p), token_index_of(tok, p, f["subject"])) for p in ps if f["subject"] in p.text]  # noqa: E731
        maps_p, maps_q = normal_equations(model, tok, items(P), unit, dev), normal_equations(model, tok, items(Q), unit, dev)
        sp = parts[f["id"]]["spellings"]["spaced"] or parts[f["id"]]["spellings"]["bare"]
        target = PUSH * sp["vecs"][0].to(dev).float()
        A_p = sum(T.T.double() @ T.double() for T in maps_p)
        A_q = sum(T.T.double() @ T.double() for T in maps_q) if maps_q else torch.zeros_like(A_p)
        b = sum(T.T.double() @ target.double() for T in maps_p)
        mu = MU_REL * float(A_p.diagonal().mean())
        delta = torch.linalg.solve(A_p + LAM * A_q + mu * torch.eye(A_p.shape[0], device=dev, dtype=A_p.dtype), b).float()
        hit = torch.stack([T @ delta for T in maps_p])
        fit = float(((hit - target).norm(dim=1) / target.norm()).median())
        off = float(torch.stack([(T @ delta).norm() for T in maps_q]).median() / target.norm()) if maps_q else float("nan")
        deltas.append(delta)
        solve_stats.append(dict(id=f["id"], delta_norm=round(float(delta.norm()), 3), delta_over_unit=round(float(delta.norm()) / unit, 2),
                                fit_error=round(fit, 3), other_questions_arrival=round(off, 3), P=len(maps_p), Q=len(maps_q), P_prompts=[p.text for p in P]))
        if (j + 1) % 8 == 0:
            last = solve_stats[-8:]
            log(f"[solve {j + 1}/{n}] median over the last 8: |delta| {torch.tensor([s['delta_over_unit'] for s in last]).median():.2f} units, "
                f"miss on P {torch.tensor([s['fit_error'] for s in last]).median():.2f} of the push, arrival on Q "
                f"{torch.tensor([s['other_questions_arrival'] for s in last]).median():.2f} of the push | {time.time() - t0:.0f}s")

    # ---- size check on P only: the smallest scale at which the new answer's first token beats the old on 7 of 8
    def first_ids(text):
        return tok(" " + text, add_special_tokens=False).input_ids[0]

    scales = []
    for j, f in enumerate(facts):
        P = [Prompt(t) for t in solve_stats[j]["P_prompts"]]
        new_id, old_id = first_ids(f["cf"]["new"]), first_ids(f["cf"]["old"])
        chosen_scale = SCALES[-1]
        for s in SCALES:
            hook = TagHook(model, tok, [f["subject"]], (deltas[j] * s)[None], NAME_LAYER)
            try:
                with torch.no_grad():
                    ids, mask = batch_ids(tok, [encode(tok, p) for p in P], dev, side="left")
                    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1].float()
            finally:
                hook.remove()
            wins = int((logits[:, new_id] > logits[:, old_id]).sum())
            if wins >= WIN_MIN:
                chosen_scale = s
                break
        scales.append(chosen_scale)
        solve_stats[j]["scale"] = chosen_scale
        solve_stats[j]["wins_on_P"] = wins
    deltas = [d * s for d, s in zip(deltas, scales)]
    log(f"size check: scale 1 for {scales.count(1.0)}, 1.5 for {scales.count(1.5)}, 2 for {scales.count(2.0)}, 3 for {scales.count(3.0)}; "
        f"edits winning on 7 of 8 training wordings at their scale: {sum(s['wins_on_P'] >= WIN_MIN for s in solve_stats)} of {n} | "
        f"{time.time() - t0:.0f}s")

    out = dict(edits=[f["id"] for f in facts], unit=unit, solve=solve_stats, A=s_a, unedited=step2["unedited"])
    mlpL = blocks(model)[NAME_LAYER].mlp
    GL, UL, DL = mlpL.gate_proj.weight.data.clone(), mlpL.up_proj.weight.data.clone(), mlpL.down_proj.weight.data.clone()

    def restore():
        mlpL.gate_proj.weight.data.copy_(GL)
        mlpL.up_proj.weight.data.copy_(UL)
        mlpL.down_proj.weight.data.copy_(DL)

    try:
        # ================= D2: delta written by a hook at the exact name
        log("D2: the edits written at the exact name by a hook")
        hook = TagHook(model, tok, subjects, torch.stack(deltas), NAME_LAYER)
        try:
            rows_d, s_d = score(model, tok, facts, before, baseline, cf_check)
        finally:
            hook.remove()
        log(line("D2, written at the name by a hook", s_d) + f" | {time.time() - t0:.0f}s")

        # ================= E2: name neurons at layer 8 write delta
        log(f"E2: name neurons in layer {NAME_LAYER} write the edits")
        fit_texts, spare_texts = [paras[i] for i in CALIBRATION_FIT], [paras[i] for i in SPARE]
        Hf, _, _ = second_moment(model, tok, mlpL.down_proj, fit_texts)
        Hv, _, _ = second_moment(model, tok, mlpL.down_proj, spare_texts)
        slotsL = redundancy_order(Hf, DL.double(), n)
        K, Dc, ridge, left, _ = compensation(Hf, Hv, DL.double(), slotsL)
        del Hf, Hv
        freedL = DL.double()
        freedL[:, slotsL] = 0
        freedL[:, K] += Dc
        mlpL.down_proj.weight.data.copy_(freedL.to(DL.dtype))
        lost = sum(b and not x for b, x in zip(baseline.known_check.before, baseline.known_check.correct()))
        lost_cf = sum(b and not x for b, x in zip(cf_check.before, cf_check.correct()))
        out["freed"] = dict(known_lost=lost, cf_lost=lost_cf, drift=baseline.drift()[0])
        log(f"  {n} slots freed at layer {NAME_LAYER}: known lost {lost} of 133, {lost_cf} of 391, drift {out['freed']['drift']:.5f}")
        restore()

        prompt_sets = {f["id"]: plain_of[f["id"]] + same_of[f["id"]] + chat_of[f["id"]] for f in facts}
        seqs, owner, at_list = [], [], []
        for j, f in enumerate(facts):
            ps = prompt_sets[f["id"]]
            for p, at in zip(ps, name_positions(tok, ps, f["subject"])):
                seqs.append(encode(tok, p))
                owner.append(j)
                at_list.append(at)
        readsL = all_reads(model, tok, seqs, NAME_LAYER)
        ordL = all_reads(model, tok, [tok(paras[i], truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids
                                      for i in list(CALIBRATION_FIT)[:N_PARAS_NAME]], NAME_LAYER)
        X = torch.cat(readsL + ordL).float()
        label = torch.full((X.shape[0],), -1, dtype=torch.long)
        k = 0
        for r, j, at in zip(readsL, owner, at_list):
            if at is not None:
                label[k + at] = j
            k += r.shape[0]
        label = label.to(dev)
        Xd = X.double()
        total_sum, total_m2, n_all = Xd.sum(0), Xd.T @ Xd, X.shape[0]
        del Xd
        silu_n = float(F.silu(torch.tensor(K_NAME * Z_MED)))
        gateL, upL, downL = GL.clone(), UL.clone(), freedL.clone()
        separable = 0
        for j, f in enumerate(facts):
            Pm = X[label == j].double()
            n_neg = n_all - Pm.shape[0]
            m_pos, m_neg = Pm.mean(0), (total_sum - Pm.sum(0)) / n_neg
            sc_pos = (Pm - m_pos).T @ (Pm - m_pos)
            sc_neg = (total_m2 - Pm.T @ Pm) - n_neg * torch.outer(m_neg, m_neg)
            Sm = covL.double() + (sc_pos + sc_neg) / n_all
            w = torch.linalg.solve(Sm + RIDGE * Sm.diagonal().mean() * torch.eye(Sm.shape[0], device=dev, dtype=Sm.dtype), m_pos - m_neg).float()
            rho = ratio(X, w, cL, cbarL)
            tau, med = float(rho[label != j].max()), float(rho[label == j].median())
            if med <= tau:
                downL[:, slotsL[j]] = 0
                continue
            separable += 1
            gateL[slotsL[j]] = (K_NAME * (w.double() - (tau / cbarL) * cL.double()) / (0.5 * (med - tau))).to(GL.dtype)
            upL[slotsL[j]] = (cL.double() / cbarL).to(UL.dtype)
            downL[:, slotsL[j]] = deltas[j].double() / silu_n
        del X, readsL, ordL
        mlpL.gate_proj.weight.data.copy_(gateL)
        mlpL.up_proj.weight.data.copy_(upL)
        mlpL.down_proj.weight.data.copy_(downL.to(DL.dtype))
        log(f"  name neurons: separable for {separable} of {n} | {time.time() - t0:.0f}s")
        log("E2: scoring the grafted model (no hook)")
        rows_e, s_e = score(model, tok, facts, before, baseline, cf_check)
        log(line("E2, grafted: the edits written by name neurons", s_e) + f" | {time.time() - t0:.0f}s")
        torch.save(dict(name_layer=NAME_LAYER, slotsL=slotsL, gateL=gateL.cpu(), upL=upL.cpu(), downL=downL.to(DL.dtype).cpu()),
                   os.path.join(a.out, "grafted_layers.pt"))
    finally:
        restore()

    def ok_vs_a(s):
        return dict(es=s["es"] >= s_a["es"] - 0.05, ps=s["ps"] >= s_a["ps"] - 0.05, ns=s["ns"] >= s_a["ns"] - 0.01,
                    neigh_leak=s["neigh_leak"] <= 0.05, same_leak=s["same_leak"] <= 0.10, known_lost=s["known_lost"] <= 2,
                    cf_lost=s["cf_lost"] <= 7, drift=s["drift"] <= 0.05)
    vd, ve = ok_vs_a(s_d), ok_vs_a(s_e)
    killed = s_e["ps"] < 0.50 or not all(ve[k] for k in ("neigh_leak", "same_leak", "known_lost", "cf_lost", "drift"))
    out.update(D2=s_d, E2=s_e, verdict_D2=vd, verdict_E2=ve, E2_grafted=all(ve.values()), kill_rule_met=killed)
    log(f"D2 (written at the name by a hook) meets the bar: {all(vd.values())} {vd}")
    log(f"E2 is a grafted model as good as the inserted layer: {all(ve.values())} {ve}")
    log(f"kill rule (E2's PS under 0.50, or a leak or damage bar failed): {'MET: key 1 in the weights is put aside' if killed else 'not met'}")
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for name, rows in (("D2", rows_d), ("E2", rows_e)):
        with open(os.path.join(a.out, f"edits_{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

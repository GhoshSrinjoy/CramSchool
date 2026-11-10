"""Step 3, revision 1, of the grafted model (lab notebook, "Step 3, revision 1"). Changes from scripts/slot_names.py, each from a measured
failure: the tag reading follows the tag (tagged against the same prompts untagged) with a separation guard; one orthogonal tag per
subject (the transport map's input directions 2 to 65, equal arrival); one graded name neuron per subject, in the layer (4, 8 or 12)
whose freed slots cost the fewest of the clean 391 among those within the CounterFact bar on the 133; the AND gate k (z_q + M (t - 1)) with M = 1, up t; false fires counted against the name
index. Arm D writes the tags with a hook from the name index; arm E with the name neurons, no hook. Step 2's 64 edits and reference A.

Usage: as scripts/slot_names.py.
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
from slot_compile import GAIN, PUSH, line, score  # noqa: E402
from slot_names import EventIndex, all_reads, last_reads, name_positions  # noqa: E402

from cramschool.bench.counterfact import load_counterfact  # noqa: E402
from cramschool.bench.methods import calibrate  # noqa: E402
from cramschool.bench.scoring import make_baseline  # noqa: E402
from cramschool.facts import PREFIXES  # noqa: E402
from cramschool.graft.build import build_context, chat_items, fact_detectors, guarded, kind_parts, plain_items  # noqa: E402
from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_HELD, CALIBRATION_MAX_TOKENS, RIDGE, blocks, ratio  # noqa: E402
from cramschool.graft.wordings import Wordings, calibration_paragraphs, norm  # noqa: E402
from cramschool.model_access import Prompt, batch_ids, encode, token_index_of  # noqa: E402

NAME_LAYERS, FACT_LAYER = (4, 8, 12), 20
K_NAME, Z_MED = 8.0, 2.0           # one name neuron: silu(8 z); the tag at full size at the positives' median z (2 by construction)
M_TAG, SEP_MIN = 1.0, 3.0
TAG_SIZE = 2.0
N_TEMPLATES, N_INVENTED, N_REAL, N_CHAT, SEED = 16, 12, 24, 8, 0
N_PARAS_NAME, N_PARAS_ORDINARY, N_ORDINARY = 120, 40, 4000


def log(s):
    print(s, flush=True)


def parse():
    ap = argparse.ArgumentParser()
    for k in ("model", "run", "relations", "twins", "data", "slots", "step2", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--name-layer", type=int, help="measure and use only this name layer (the corrected rule's choice from a full scan)")
    return ap.parse_args()


class TagHook:
    """D's tags: added to layer L's MLP output at the token where an edited subject's name ends, as the name index reads it."""

    def __init__(self, model, tok, subjects, tags, layer):
        self.index = EventIndex(tok, subjects)
        self.tag_of = {self.index.lookup[" ".join(s.lower().split())]: tags[i] for i, s in enumerate(subjects)}
        self.n, self.events = None, []
        self.handles = [model.register_forward_pre_hook(self._top, with_kwargs=True), blocks(model)[layer].mlp.register_forward_hook(self._add)]

    def _top(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        n, t = ids.shape
        if t > 1 or self.n != n:
            self.index.reset(n)
            self.n = n
        valid = (mask[:, -t:].bool() if mask is not None else torch.ones_like(ids, dtype=torch.bool)).tolist()
        idl, self.events = ids.tolist(), []
        for b in range(n):
            for i in range(t):
                if valid[b][i]:
                    e = self.index.step_event(b, idl[b][i])
                    if e is not None:
                        self.events.append((b, i, e))

    def _add(self, module, args, output):
        if not self.events:
            return None
        out = output.clone()
        for b, i, e in self.events:
            out[b, i] += self.tag_of[e].to(out.dtype)
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def transport(model, tok, items, size, layer, dev):
    """Per (ids, name position): the change in layer 20's MLP input at the last token per basis direction added (at `size`) to layer
    `layer`'s MLP output at the name (out x in)."""
    hidden = model.config.get_text_config().hidden_size
    state = {}

    def add(m, a, out):
        out = out.clone()
        out[:, state["pos"]] += state["add"].to(out.dtype)
        return out

    def read20(m, a):
        state["read"] = a[0][:, -1].float()

    h1 = blocks(model)[layer].mlp.register_forward_hook(add)
    h2 = blocks(model)[FACT_LAYER].mlp.register_forward_pre_hook(read20)
    maps = []
    try:
        for ids, pos in items:
            adds = torch.cat([torch.zeros(1, hidden, device=dev), torch.eye(hidden, device=dev) * size])
            reads = []
            for start in range(0, adds.shape[0], 256):
                state["add"], state["pos"] = adds[start:start + 256], pos
                model(input_ids=torch.tensor([ids] * state["add"].shape[0], device=dev), use_cache=False)
                reads.append(state["read"])
            r = torch.cat(reads)
            maps.append(((r[1:] - r[0]) / size).T)
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
    with open(a.relations, encoding="utf-8") as fh:
        cf_ids = {v: k for k, v in json.load(fh)["counterfact"].items()}
    with open(a.twins, encoding="utf-8") as fh:
        twins = json.load(fh)["twins"]
    with open(a.slots, encoding="utf-8") as fh:
        step1c = json.load(fh)
    slots20 = step2["slots"]
    subjects = [f["subject"] for f in facts]
    n = len(facts)
    log(f"{n} edits (step 2's); A, the inserted layer: ES {s_a['es']:.3f} PS {s_a['ps']:.3f} NS {s_a['ns']:.3f}")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    dev = model.device
    baseline = make_baseline(model, tok)
    cf_check = ScreenCheck(model, tok, items=[(r["prompt"], r["answers"]) for r in step1c["cf_known"]])
    paras = calibration_paragraphs()
    fit_texts, spare_texts = [paras[i] for i in CALIBRATION_FIT], [paras[i] for i in SPARE]

    def damage():
        lost = sum(b and not x for b, x in zip(baseline.known_check.before, baseline.known_check.correct()))
        lost_cf = sum(b and not x for b, x in zip(cf_check.before, cf_check.correct()))
        return dict(known_lost=lost, cf_lost=lost_cf, drift=baseline.drift()[0])

    # ---- the name layer: 64 freed slots (redundancy order, compensated) at each candidate, alone
    scan = {}
    layers = (a.name_layer,) if a.name_layer is not None else NAME_LAYERS
    for L in layers:
        down = blocks(model)[L].mlp.down_proj
        W0 = down.weight.data.clone()
        Hf, _, _ = second_moment(model, tok, down, fit_texts)
        Hv, _, _ = second_moment(model, tok, down, spare_texts)
        order = redundancy_order(Hf, W0.double(), n)
        K, Dc, ridge, left, _ = compensation(Hf, Hv, W0.double(), order[:n])
        del Hf, Hv
        freed = W0.double()
        freed[:, order[:n]] = 0
        freed[:, K] += Dc
        down.weight.data.copy_(freed.to(W0.dtype))
        try:
            d = damage()
        finally:
            down.weight.data.copy_(W0)
        scan[L] = dict(slots=order[:n], freed=freed.to(W0.dtype).cpu(), ridge=ridge, energy_left=left, **d)
        log(f"name layer {L}: {n} freed, compensated: known lost {d['known_lost']} of 133, {d['cf_lost']} of the clean 391, drift "
            f"{d['drift']:.5f} | {time.time() - t0:.0f}s")
    # the corrected rule: the fewest of the clean 391 lost among the layers losing at most 2 of the 133 (ties: fewer of the 133, drift)
    within = [x for x in layers if scan[x]["known_lost"] <= 2] or list(layers)
    L = min(within, key=lambda x: (scan[x]["cf_lost"], scan[x]["known_lost"], scan[x]["drift"]))
    slotsL = scan[L]["slots"]
    log(f"name layer chosen by the corrected rule: {L}")

    m20, mL = calibrate(model, tok, FACT_LAYER), calibrate(model, tok, L)
    c20, cbar20, cov20 = m20["c"].to(dev).float(), m20["cbar"], m20["cov"].to(dev).float()
    cL, cbarL, covL = mL["c"].to(dev).float(), mL["cbar"], mL["cov"].to(dev).float()

    # ---- prompt sets (training wordings only)
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
    tests = " | ".join(p.lower() for f in facts for p in [f["cf"]["rewrite"]] + f["cf"]["paraphrases"] + f["cf"]["neighbours"]
                       + f["cf"].get("twin_prompts", []) + f["test"]["same_plain"])
    records = load_counterfact(a.data)
    chosen = {s.lower() for s in subjects}
    others = {}
    for rel in templates:
        inv = kind_parts(ctx, rel)[0][:N_INVENTED]
        pool = sorted({r["requested_rewrite"]["subject"] for r in records if r["requested_rewrite"]["relation_id"] == cf_ids[rel]})
        pool = [s for s in pool if s.lower() not in tests and s.lower() not in chosen]
        real = random.Random(SEED).sample(pool, min(N_REAL, len(pool)))
        others[rel] = guarded(plain_items(templates[rel], inv + real), ctx["guard"])
    ord_seqs = [tok(paras[i], truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids
                for i in list(CALIBRATION_FIT)[:N_PARAS_ORDINARY]]

    # ---- the tag unit and the transport map from layer L, on one edit's rewrite prompt per relation
    first = {}
    for f in facts:
        first.setdefault(f["relation"], f)
    items = []
    for f in first.values():
        p = Prompt(f["cf"]["rewrite"])
        items.append((encode(tok, p), token_index_of(tok, p, f["subject"])))
    seen = {}
    hn = blocks(model)[L].mlp.register_forward_hook(lambda m, a_, out: seen.__setitem__("o", out[0].float()))
    unit_norms = []
    try:
        with torch.no_grad():
            for ids, pos in items:
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
                unit_norms.append(float(seen["o"][pos].norm()))
    finally:
        hn.remove()
    unit = float(torch.tensor(unit_norms).median())
    maps = transport(model, tok, items, TAG_SIZE * unit, L, dev)
    _, S, Vh = torch.linalg.svd(torch.cat(maps))
    sig = S[1:1 + n]
    sizes = TAG_SIZE * unit * float(sig.median()) / sig
    tags = (Vh[1:1 + n] * sizes[:, None]).to(dev)               # one orthogonal tag per subject, equal arrival
    arrival = torch.stack([torch.stack([m @ tags[j] for m in maps]).norm(dim=1).median() for j in range(n)])
    log(f"tag unit {unit:.2f} at layer {L}; singular values: top {float(S[0]):.1f}, 2nd {float(S[1]):.1f}, 65th {float(S[n]):.1f}; tag sizes "
        f"{float(sizes.min()):.2f} to {float(sizes.max()):.2f}; arrival at layer 20 (median over prompts): {float(arrival.min()):.1f} to "
        f"{float(arrival.max()):.1f} | {time.time() - t0:.0f}s")

    # ---- layer 20's slots, as in step 2
    mlp20, mlpL = blocks(model)[FACT_LAYER].mlp, blocks(model)[L].mlp
    G20, U20, D20 = mlp20.gate_proj.weight.data.clone(), mlp20.up_proj.weight.data.clone(), mlp20.down_proj.weight.data.clone()
    GL, UL, DL = mlpL.gate_proj.weight.data.clone(), mlpL.up_proj.weight.data.clone(), mlpL.down_proj.weight.data.clone()
    Hf, _, _ = second_moment(model, tok, mlp20.down_proj, fit_texts)
    Hv, _, _ = second_moment(model, tok, mlp20.down_proj, spare_texts)
    K20, Dc20, ridge20, left20, _ = compensation(Hf, Hv, D20.double(), slots20)
    del Hf, Hv
    freed20 = D20.double()
    freed20[:, slots20] = 0
    freed20[:, K20] += Dc20
    freedL = scan[L]["freed"].to(dev)
    det = {f["id"]: next(d for d in fact_detectors(parts[f["id"]]) if d["form"] == "plain") for f in facts}
    silu_k = float(F.silu(torch.tensor(GAIN)))

    def restore():
        for mod, w in ((mlp20.gate_proj, G20), (mlp20.up_proj, U20), (mlp20.down_proj, D20), (mlpL.gate_proj, GL), (mlpL.up_proj, UL),
                       (mlpL.down_proj, DL)):
            mod.weight.data.copy_(w)

    def gather():
        tagged = {f["id"]: last_reads(model, tok, plain_of[f["id"]], FACT_LAYER) for f in facts}
        other = {rel: last_reads(model, tok, ps, FACT_LAYER) for rel, ps in others.items()}
        ordinary = torch.cat(all_reads(model, tok, ord_seqs, FACT_LAYER)).float()
        ordinary = ordinary[torch.randperm(ordinary.shape[0], generator=torch.Generator().manual_seed(SEED))[:N_ORDINARY].to(ordinary.device)]
        return tagged, other, ordinary

    def readouts(tagged, untagged, other, ordinary):
        """Each edit's tag reading: w = (S + ridge)^-1 (tagged mean - untagged mean), S the calibration covariance plus every group's
        within scatter; 1 at the tagged prompts' median, 0 at the negatives' 99th percentile; None if it separates by under 3 sd."""
        rows, stats = {}, {}
        for f in facts:
            pos, base = tagged[f["id"]], untagged[f["id"]]
            mates = torch.cat([tagged[g["id"]][:8] for g in facts if g["id"] != f["id"]])
            groups = [pos, base, mates, other[f["relation"]], ordinary]
            within = torch.cat([g - g.mean(0) for g in groups])
            Sm = cov20 + within.T @ within / within.shape[0]
            w = torch.linalg.solve(Sm + RIDGE * Sm.diagonal().mean() * torch.eye(Sm.shape[0], device=dev), pos.mean(0) - base.mean(0))
            rp = ratio(pos, w, c20, cbar20)
            negs = {"untagged": base, "mates": mates, "others": other[f["relation"]], "ordinary": ordinary}
            rn = {k: ratio(v, w, c20, cbar20) for k, v in negs.items()}
            alln = torch.cat(list(rn.values()))
            b1, b0 = float(rp.median()), float(torch.quantile(alln, 0.99))
            sep = (b1 - b0) / float(alln.std())
            ok = sep >= SEP_MIN
            stats[f["id"]] = dict(separation=round(sep, 2), placed=bool(ok), pos_q05=round(float(torch.quantile((rp - b0) / (b1 - b0), 0.05)), 3),
                                  **{f"{k}_max": round(float(((v - b0) / (b1 - b0)).max()), 3) for k, v in rn.items()})
            rows[f["id"]] = (w.double() - b0 * c20.double() / cbar20) / (b1 - b0) if ok else None
        return rows, stats

    def place_facts(rows_t):
        gate, up, down = G20.clone(), U20.clone(), freed20.clone()
        for j, f in enumerate(facts):
            tr = rows_t[f["id"]]
            if tr is None:
                continue  # its slot stays freed: down column 0
            d = det[f["id"]]
            zq = (d["w"].to(dev).double() - (d["tau"] / cbar20) * c20.double()) / d["delta"]
            gate[slots20[j]] = (GAIN * (zq + M_TAG * (tr - c20.double() / cbar20))).to(G20.dtype)
            up[slots20[j]] = tr.to(U20.dtype)
            sp = parts[f["id"]]["spellings"]["spaced"] or parts[f["id"]]["spellings"]["bare"]
            down[:, slots20[j]] = PUSH * sp["vecs"][0].to(dev).double() / silu_k
        mlp20.gate_proj.weight.data.copy_(gate)
        mlp20.up_proj.weight.data.copy_(up)
        mlp20.down_proj.weight.data.copy_(down.to(D20.dtype))

    def fact_outputs_on_held():
        """Each fact neuron's largest |activation| on held-out text, over its activation at z = 1, t = 1 (silu(k))."""
        got = {}
        h = mlp20.down_proj.register_forward_pre_hook(lambda m, a_: got.__setitem__("a", a_[0][..., slots20].float().abs()))
        top = torch.zeros(len(slots20), device=dev)
        try:
            with torch.no_grad():
                for i in CALIBRATION_HELD:
                    ids = tok(paras[i], truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids
                    model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
                    top = torch.maximum(top, got["a"][0].max(0).values)
        finally:
            h.remove()
        return (top / silu_k).tolist()

    def summarize_readings(name, stats):
        st = list(stats.values())
        log(f"  tag readings ({name}): placed {sum(s['placed'] for s in st)} of {len(st)} (separation ≥ {SEP_MIN:g} sd; median "
            f"{torch.tensor([s['separation'] for s in st]).median():.1f}); among placed: tagged prompts' 5th percentile median "
            f"{torch.tensor([s['pos_q05'] for s in st if s['placed']] or [float('nan')]).median():.2f}, largest untagged "
            f"{max([s['untagged_max'] for s in st if s['placed']] or [float('nan')]):.2f}, other edits' tags "
            f"{max([s['mates_max'] for s in st if s['placed']] or [float('nan')]):.2f}, other subjects "
            f"{max([s['others_max'] for s in st if s['placed']] or [float('nan')]):.2f}, ordinary text "
            f"{max([s['ordinary_max'] for s in st if s['placed']] or [float('nan')]):.2f} | {time.time() - t0:.0f}s")

    untagged = {f["id"]: last_reads(model, tok, plain_of[f["id"]], FACT_LAYER) for f in facts}  # the unedited model
    out = dict(edits=[f["id"] for f in facts], name_layer=L, scan={k: {x: y for x, y in v.items() if x not in ("freed", "slots")} for k, v in scan.items()},
               tag=dict(unit=unit, sizes=[round(float(x), 3) for x in sizes], singular=[round(float(x), 2) for x in S[:2 * n]]), A=s_a,
               unedited=step2["unedited"])
    try:
        # ================= D: tags from the name index
        log("D: tags written by a hook from the name index")
        hook = TagHook(model, tok, subjects, tags, L)
        try:
            tagged, other, ordinary = gather()
            rows_t, stats_d = readouts(tagged, untagged, other, ordinary)
            summarize_readings("D", stats_d)
            place_facts(rows_t)
            out["D_fact_outputs_held"] = fact_outputs_on_held()
            log(f"  fact neurons' largest output on held-out text, over their output at z = 1, t = 1: "
                f"{max(out['D_fact_outputs_held']):.2f}")
            rows_d, s_d = score(model, tok, facts, before, baseline, cf_check)
        finally:
            hook.remove()
            restore()
        log(line("D, tags from the name index", s_d) + f" | {time.time() - t0:.0f}s")

        # ================= E: name neurons in layer L
        log(f"E: name neurons in layer {L}")
        mlpL.down_proj.weight.data.copy_(freedL)
        mlp20.down_proj.weight.data.copy_(freed20.to(D20.dtype))
        out["freed_both"] = damage()
        log(f"  freed slots alone, layers {L} and {FACT_LAYER}: known lost {out['freed_both']['known_lost']} of 133, "
            f"{out['freed_both']['cf_lost']} of the clean 391, drift {out['freed_both']['drift']:.5f}")
        restore()

        prompt_sets = {f["id"]: plain_of[f["id"]] + same_of[f["id"]] + chat_of[f["id"]] for f in facts}
        seqs, owner, at_list = [], [], []
        for j, f in enumerate(facts):
            ps = prompt_sets[f["id"]]
            for p, at in zip(ps, name_positions(tok, ps, f["subject"])):
                seqs.append(encode(tok, p))
                owner.append(j)
                at_list.append(at)
        readsL = all_reads(model, tok, seqs, L)
        ordL = all_reads(model, tok, [tok(paras[i], truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids
                                      for i in list(CALIBRATION_FIT)[:N_PARAS_NAME]], L)
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
        name_rows, name_stats = [], []
        for j, f in enumerate(facts):
            P = X[label == j].double()
            n_neg = n_all - P.shape[0]
            m_pos, m_neg = P.mean(0), (total_sum - P.sum(0)) / n_neg
            sc_pos = (P - m_pos).T @ (P - m_pos)
            sc_neg = (total_m2 - P.T @ P) - n_neg * torch.outer(m_neg, m_neg)
            Sm = covL.double() + (sc_pos + sc_neg) / n_all
            w = torch.linalg.solve(Sm + RIDGE * Sm.diagonal().mean() * torch.eye(Sm.shape[0], device=dev, dtype=Sm.dtype), m_pos - m_neg).float()
            rho = ratio(X, w, cL, cbarL)
            tau, med = float(rho[label != j].max()), float(rho[label == j].median())
            ok = med > tau
            name_stats.append(dict(subject=f["subject"], separable=ok, positives_above=round(float((rho[label == j] > tau).float().mean()), 3)))
            name_rows.append((w.double(), tau, 0.5 * (med - tau)) if ok else None)
        del X, readsL, ordL
        log(f"  name detectors at layer {L}: separable for {sum(s['separable'] for s in name_stats)} of {n}")

        silu_n = float(F.silu(torch.tensor(K_NAME * Z_MED)))
        gateL, upL, downL = GL.clone(), UL.clone(), freedL.double().clone()
        for j, row in enumerate(name_rows):
            if row is None:
                continue
            w, tau, delta = row
            gateL[slotsL[j]] = (K_NAME * (w - (tau / cbarL) * cL.double()) / delta).to(GL.dtype)
            upL[slotsL[j]] = (cL.double() / cbarL).to(UL.dtype)
            downL[:, slotsL[j]] = tags[j].double() / silu_n
        mlpL.gate_proj.weight.data.copy_(gateL)
        mlpL.up_proj.weight.data.copy_(upL)
        mlpL.down_proj.weight.data.copy_(downL.to(DL.dtype))

        # false fires: a subject's name neuron firing (over half its median activation) where the name index finds no such name
        got = {}
        hf = mlpL.down_proj.register_forward_pre_hook(lambda m, a_: got.__setitem__("a", a_[0][..., slotsL].float()))
        idx = EventIndex(tok, subjects)
        sid = {idx.lookup[" ".join(s.lower().split())]: j for j, s in enumerate(subjects)}

        def fires(token_lists):
            res = []
            with torch.no_grad():
                for start in range(0, len(token_lists), 32):
                    chunk = token_lists[start:start + 32]
                    ids, mask = batch_ids(tok, chunk, dev)
                    model(input_ids=ids, attention_mask=mask, use_cache=False)
                    act = got["a"] / silu_n                                   # 1 at a positive's median
                    for i, s_ in enumerate(chunk):
                        idx.reset(1)
                        named = {sid[e] for t_ in s_ if (e := idx.step_event(0, t_)) is not None}
                        fired = set((act[i, :len(s_)].max(0).values > 0.5).nonzero()[:, 0].tolist())
                        res.append((named, fired, len(s_)))
            return res
        try:
            own = fires([encode(tok, Prompt(p)) for f in facts for p in [f["cf"]["rewrite"]] + f["cf"]["paraphrases"]])
            own_ok = sum(j in fired for (named, fired, _), j in zip(own, [j for j in range(n) for _ in range(3)]))
            neigh = fires([encode(tok, Prompt(p)) for f in facts for p in f["cf"]["neighbours"]])
            neigh_false = sum(bool(fired - named) for named, fired, _ in neigh)
            held = fires([tok(paras[i], truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids for i in CALIBRATION_HELD])
            held_false = sum(len(fired - named) for named, fired, _ in held)
        finally:
            hf.remove()
        out["name_neurons"] = dict(layer=L, separable=sum(s["separable"] for s in name_stats), own_name_fires=own_ok, own_prompts=len(own),
                                   neighbour_prompts_false_fire=neigh_false, neighbour_prompts=len(neigh), held_paragraphs_false_fires=held_false,
                                   per_subject=name_stats)
        log(f"  name neurons: fire on their subject's name in its rewrite and reworded prompts {own_ok} of {len(own)}; neighbour prompts "
            f"with a false fire (a subject not named there) {neigh_false} of {len(neigh)}; false fires on held-out paragraphs {held_false} "
            f"(subject-paragraph pairs) | {time.time() - t0:.0f}s")

        mlpL.down_proj.weight.data.copy_(freedL)                       # E's untagged reads: layer L freed, the name neurons silent
        untagged_e = {f["id"]: last_reads(model, tok, plain_of[f["id"]], FACT_LAYER) for f in facts}
        mlpL.down_proj.weight.data.copy_(downL.to(DL.dtype))
        tagged, other, ordinary = gather()
        rows_t, stats_e = readouts(tagged, untagged_e, other, ordinary)
        summarize_readings("E", stats_e)
        place_facts(rows_t)
        out["E_fact_outputs_held"] = fact_outputs_on_held()
        log(f"  fact neurons' largest output on held-out text, over their output at z = 1, t = 1: {max(out['E_fact_outputs_held']):.2f}")
        log("E: scoring the grafted model (no hook)")
        rows_e, s_e = score(model, tok, facts, before, baseline, cf_check)
        log(line("E, grafted with name neurons", s_e) + f" | {time.time() - t0:.0f}s")
        torch.save(dict(name_layer=L, slotsL=slotsL, slots20=slots20, gateL=gateL.cpu(), upL=upL.cpu(), downL=downL.to(DL.dtype).cpu(),
                        gate20=mlp20.gate_proj.weight.data.cpu(), up20=mlp20.up_proj.weight.data.cpu(), down20=mlp20.down_proj.weight.data.cpu()),
                   os.path.join(a.out, "grafted_layers.pt"))
    finally:
        restore()

    def ok_vs_a(s):
        return dict(es=s["es"] >= s_a["es"] - 0.05, ps=s["ps"] >= s_a["ps"] - 0.05, ns=s["ns"] >= s_a["ns"] - 0.01,
                    neigh_leak=s["neigh_leak"] <= 0.05, same_leak=s["same_leak"] <= 0.10, known_lost=s["known_lost"] <= 2,
                    cf_lost=s["cf_lost"] <= 7, drift=s["drift"] <= 0.05)
    vd, ve = ok_vs_a(s_d), ok_vs_a(s_e)
    out.update(D=s_d, E=s_e, verdict_D=vd, verdict_E=ve, D_passes=all(vd.values()), E_grafted=all(ve.values()), tag_readings_D=stats_d,
               tag_readings_E=stats_e, slotsL=slotsL)
    log(f"D (tags from the name index) meets the bar: {all(vd.values())} {vd}")
    log(f"E is a grafted model as good as the inserted layer: {all(ve.values())} {ve}")
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for name, rows in (("D", rows_d), ("E", rows_e)):
        with open(os.path.join(a.out, f"edits_{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

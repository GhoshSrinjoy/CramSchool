"""Step 4 of the grafted model: key 1 through an attention channel (lab notebook, "Step 4: key 1 through an attention channel").

The name neuron (layer 8) writes the subject's tag at its name. One attention head that already looks from the end of the question back to
the name gets a value channel: its key-value head's v_proj gains gamma * sum_s v_s r_s^T, which reads tag s at the name and writes code v_s
into the value there, so the head carries the code to the question's end. The fact neuron (layer 20) reads it: gate = the question
detector, up = the code's reading t, down = the push (step 2's B with the name index replaced by t). Arm D3 writes the tags by a hook at the
exact name; arm E3 by name neurons (no hook). Step 2's 64 edits and reference A.

Usage: python scripts/slot_channel.py --model <folder> --run runs/counterfact/phrasing_editor --relations <editor relations>
           --twins <twins_narrow.json> --data <counterfact.json> --slots <step 1c json> --step2 <step 2 folder> --out <folder>
"""
import argparse
import json
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_slot_compensate import SPARE, ScreenCheck, compensation, redundancy_order, second_moment  # noqa: E402
from slot_compile import GAIN, PUSH, line, score  # noqa: E402
from slot_names import EventIndex, all_reads, last_reads, name_positions  # noqa: E402
from slot_names_rev1 import TagHook  # noqa: E402

from cramschool.bench.counterfact import load_counterfact  # noqa: E402
from cramschool.bench.methods import calibrate  # noqa: E402
from cramschool.bench.scoring import make_baseline  # noqa: E402
from cramschool.facts import PREFIXES  # noqa: E402
from cramschool.graft.build import build_context, chat_items, fact_detectors, guarded, kind_parts, plain_items  # noqa: E402
from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_HELD, CALIBRATION_MAX_TOKENS, RIDGE, blocks, ratio  # noqa: E402
from cramschool.graft.wordings import Wordings, calibration_paragraphs, norm  # noqa: E402
from cramschool.model_access import Prompt, batch_ids, encode, token_index_of  # noqa: E402

NAME_LAYER, FACT_LAYER, CHANNEL_LAYERS = 8, 20, (11, 15, 19)
K_NAME, Z_MED, TAG_SIZE = 8.0, 2.0, 2.0
GAMMAS, SEP_MIN = (1.0, 2.0, 4.0), 3.0
N_P, N_TEMPLATES, N_INVENTED, N_REAL, N_CHAT, SEED = 8, 16, 12, 24, 8, 0
N_PARAS_STATS, N_PARAS_NAME, N_PARAS_ORDINARY, N_ORDINARY = 120, 120, 40, 4000


def log(s):
    print(s, flush=True)


def parse():
    ap = argparse.ArgumentParser()
    for k in ("model", "run", "relations", "twins", "data", "slots", "step2", "out"):
        ap.add_argument(f"--{k}", required=True)
    return ap.parse_args()


def first_words(t):
    return " ".join(re.findall(r"[A-Za-z0-9]+", t)[:2]).lower()


def changed(rows, before, group):
    tot = ch = 0
    for r in rows:
        for x, y in zip(r["replies"][group], before[r["id"]][group]):
            tot += 1
            ch += first_words(x) != first_words(y)
    return round(ch / max(tot, 1), 4)


@torch.no_grad()
def attention_to_name(model, tok, items, layers):
    """For (ids, name position) prompts: each full-attention layer's per-head attention from the last token to the name's last token."""
    got = {}
    hooks = [blocks(model)[L].self_attn.register_forward_hook(lambda m, a, o, L=L: got.__setitem__(L, o[1])) for L in layers]
    original = model.config._attn_implementation
    model.set_attn_implementation("eager")
    out = {L: [] for L in layers}
    try:
        for ids, pos in items:
            model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
            for L in layers:
                out[L].append(got[L][0, :, -1, pos].float())           # heads
    finally:
        model.set_attn_implementation(original)
        for h in hooks:
            h.remove()
    return {L: torch.stack(v) for L, v in out.items()}                 # prompts x heads


@torch.no_grad()
def covariances(model, tok, texts, layer, kv_head, head_dim):
    """The covariance over ordinary text of the residual entering `layer` and of key-value head `kv_head`'s values there."""
    seen = {}
    blk = blocks(model)[layer]
    h1 = blk.register_forward_pre_hook(lambda m, a, k: seen.__setitem__("h", (k.get("hidden_states", a[0] if a else None))[0].float()),
                                       with_kwargs=True)
    h2 = blk.self_attn.v_proj.register_forward_hook(lambda m, a, o: seen.__setitem__("v", o[0, :, kv_head * head_dim:(kv_head + 1) * head_dim].float()))
    hs, vs = [], []
    try:
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids.to(model.device)
            model(input_ids=ids, use_cache=False)
            hs.append(seen["h"][1:])
            vs.append(seen["v"][1:])
    finally:
        h1.remove()
        h2.remove()
    H, V = torch.cat(hs), torch.cat(vs)
    return torch.cov(H.T.double()), torch.cov(V.T.double()), float(V.norm(dim=1).median())


def name_rows(model, tok, facts, prompt_sets, paras, layer, cL, cbarL, covL, dev):
    """Each subject's name detector at `layer`: an LDA of the name's last token against every other token; threshold at the largest
    negative, delta half the gap to the positives' median. None where it doesn't separate."""
    seqs, owner, at_list = [], [], []
    for j, f in enumerate(facts):
        ps = prompt_sets[f["id"]]
        for p, at in zip(ps, name_positions(tok, ps, f["subject"])):
            seqs.append(encode(tok, p))
            owner.append(j)
            at_list.append(at)
    reads = all_reads(model, tok, seqs, layer)
    ords = all_reads(model, tok, [tok(paras[i], truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids
                                  for i in list(CALIBRATION_FIT)[:N_PARAS_NAME]], layer)
    X = torch.cat(reads + ords).float()
    label = torch.full((X.shape[0],), -1, dtype=torch.long)
    k = 0
    for r, j, at in zip(reads, owner, at_list):
        if at is not None:
            label[k + at] = j
        k += r.shape[0]
    label = label.to(dev)
    Xd = X.double()
    total_sum, total_m2, n_all = Xd.sum(0), Xd.T @ Xd, X.shape[0]
    del Xd
    rows = []
    for j in range(len(facts)):
        P = X[label == j].double()
        n_neg = n_all - P.shape[0]
        m_pos, m_neg = P.mean(0), (total_sum - P.sum(0)) / n_neg
        sc_pos = (P - m_pos).T @ (P - m_pos)
        sc_neg = (total_m2 - P.T @ P) - n_neg * torch.outer(m_neg, m_neg)
        Sm = covL.double() + (sc_pos + sc_neg) / n_all
        w = torch.linalg.solve(Sm + RIDGE * Sm.diagonal().mean() * torch.eye(Sm.shape[0], device=dev, dtype=Sm.dtype), m_pos - m_neg).float()
        rho = ratio(X, w, cL, cbarL)
        tau, med = float(rho[label != j].max()), float(rho[label == j].median())
        rows.append((w.double(), tau, 0.5 * (med - tau)) if med > tau else None)
    return rows


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
    cfg = model.config.get_text_config()
    head_dim, groups = cfg.head_dim, cfg.num_attention_heads // cfg.num_key_value_heads
    baseline = make_baseline(model, tok)
    cf_check = ScreenCheck(model, tok, items=[(r["prompt"], r["answers"]) for r in step1c["cf_known"]])
    paras = calibration_paragraphs()
    m20, m8 = calibrate(model, tok, FACT_LAYER), calibrate(model, tok, NAME_LAYER)
    c20, cbar20, cov20 = m20["c"].to(dev).float(), m20["cbar"], m20["cov"].to(dev).float()
    c8, cbar8, cov8 = m8["c"].to(dev).float(), m8["cbar"], m8["cov"].to(dev).float()

    def damage():
        lost = sum(b and not x for b, x in zip(baseline.known_check.before, baseline.known_check.correct()))
        lost_cf = sum(b and not x for b, x in zip(cf_check.before, cf_check.correct()))
        return dict(known_lost=lost, cf_lost=lost_cf, drift=baseline.drift()[0])

    # ---- prompt sets (training wordings only), as in step 3
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
        others[rel] = guarded(plain_items(templates[rel], inv + random.Random(SEED).sample(pool, min(N_REAL, len(pool)))), ctx["guard"])
    ord_seqs = [tok(paras[i], truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids
                for i in list(CALIBRATION_FIT)[:N_PARAS_ORDINARY]]
    rng = random.Random(SEED)
    train_items = []
    for f in facts:
        own = Prompt(f["cf"]["rewrite"])
        pool = [p for p in plain_of[f["id"]] if p.text != own.text]
        for p in [own] + rng.sample(pool, min(N_P - 1, len(pool))):
            train_items.append((encode(tok, p), token_index_of(tok, p, f["subject"])))

    # ---- 1. the channel head: the largest 10th percentile of attention from the last token to the name, over the training wordings
    att = attention_to_name(model, tok, train_items, CHANNEL_LAYERS)
    best = max(((L, h) for L in CHANNEL_LAYERS for h in range(att[L].shape[1])), key=lambda x: float(torch.quantile(att[x[0]][:, x[1]], 0.10)))
    L, qh = best
    g = qh // groups
    att_stats = {f"{l}.{h}": dict(q10=round(float(torch.quantile(att[l][:, h], 0.10)), 3), median=round(float(att[l][:, h].median()), 3))
                 for l in CHANNEL_LAYERS for h in range(att[l].shape[1])}
    group_att = [round(float(att[L][:, qh_].median()), 3) for qh_ in range(g * groups, (g + 1) * groups)]
    log(f"channel: layer {L}, query head {qh} (key-value head {g}): attention to the name, 10th percentile "
        f"{att_stats[f'{L}.{qh}']['q10']:.2f}, median {att_stats[f'{L}.{qh}']['median']:.2f}; the group's medians {group_att} | {time.time() - t0:.0f}s")

    # ---- 2. tags and codes: the least-varying directions of the residual entering layer L and of head g's values, on ordinary text
    cov_h, cov_v, v_med = covariances(model, tok, [paras[i] for i in list(CALIBRATION_FIT)[:N_PARAS_STATS]], L, g, head_dim)
    _, eh = torch.linalg.eigh(cov_h)
    _, ev = torch.linalg.eigh(cov_v)
    tag_dirs = eh[:, :n].T.float().to(dev)                      # n x hidden, orthonormal, least variance first
    codes = ev[:, :n].T.float().to(dev)                         # n x head_dim, orthonormal
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
    unit8 = float(torch.tensor(norms).median())
    tags = tag_dirs * (TAG_SIZE * unit8)
    ln_w = blocks(model)[L].input_layernorm.weight.float()
    reads = torch.stack([ln_w * t for t in tag_dirs])
    reads = reads / reads.norm(dim=1, keepdim=True)              # what v_proj reads of tag s, after the layer's input norm
    vproj = blocks(model)[L].self_attn.v_proj
    V0 = vproj.weight.data.clone()
    rows_v = slice(g * head_dim, (g + 1) * head_dim)
    log(f"tags: {n} least-varying residual directions entering layer {L}, size {TAG_SIZE:g} x {unit8:.2f}; codes: {n} least-varying value "
        f"directions of head {g} (median value norm {v_med:.2f}) | {time.time() - t0:.0f}s")

    def set_channel(gamma):
        W = V0.clone().double()
        W[rows_v] += gamma * (codes.double().T @ reads.double())   # head_dim x hidden, rank n
        vproj.weight.data.copy_(W.to(V0.dtype))

    def code_norm_at_names(tagger):
        """With gamma = 1: the median norm of the code head g's value carries at a tagged name, under `tagger` (a context manager)."""
        got = {}
        h = vproj.register_forward_hook(lambda m, a_, o: got.__setitem__("v", o[0, :, rows_v].float()))
        set_channel(1.0)
        out = []
        try:
            with tagger(), torch.no_grad():
                for ids, pos in train_items[::N_P]:
                    model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
                    out.append(float((got["v"][pos].double() @ codes.double().T).norm()))
        finally:
            h.remove()
            vproj.weight.data.copy_(V0)
        return float(torch.tensor(out).median())

    # ---- layer 20's slots, as in step 2; the fact neurons and their readings
    mlp20, mlp8 = blocks(model)[FACT_LAYER].mlp, blocks(model)[NAME_LAYER].mlp
    G20, U20, D20 = mlp20.gate_proj.weight.data.clone(), mlp20.up_proj.weight.data.clone(), mlp20.down_proj.weight.data.clone()
    G8, U8, D8 = mlp8.gate_proj.weight.data.clone(), mlp8.up_proj.weight.data.clone(), mlp8.down_proj.weight.data.clone()
    fit_texts, spare_texts = [paras[i] for i in CALIBRATION_FIT], [paras[i] for i in SPARE]
    Hf, _, _ = second_moment(model, tok, mlp20.down_proj, fit_texts)
    Hv, _, _ = second_moment(model, tok, mlp20.down_proj, spare_texts)
    K20, Dc20, _, _, _ = compensation(Hf, Hv, D20.double(), slots20)
    freed20 = D20.double()
    freed20[:, slots20] = 0
    freed20[:, K20] += Dc20
    Hf, _, _ = second_moment(model, tok, mlp8.down_proj, fit_texts)
    Hv, _, _ = second_moment(model, tok, mlp8.down_proj, spare_texts)
    slots8 = redundancy_order(Hf, D8.double(), n)
    K8, Dc8, _, _, _ = compensation(Hf, Hv, D8.double(), slots8)
    del Hf, Hv
    freed8 = D8.double()
    freed8[:, slots8] = 0
    freed8[:, K8] += Dc8
    det = {f["id"]: next(d for d in fact_detectors(parts[f["id"]]) if d["form"] == "plain") for f in facts}
    silu_k = float(F.silu(torch.tensor(GAIN)))

    def restore():
        for mod, w in ((mlp20.gate_proj, G20), (mlp20.up_proj, U20), (mlp20.down_proj, D20), (mlp8.gate_proj, G8), (mlp8.up_proj, U8),
                       (mlp8.down_proj, D8), (vproj, V0)):
            mod.weight.data.copy_(w)

    def gather():
        tagged = {f["id"]: last_reads(model, tok, plain_of[f["id"]], FACT_LAYER) for f in facts}
        other = {rel: last_reads(model, tok, ps, FACT_LAYER) for rel, ps in others.items()}
        ordinary = torch.cat(all_reads(model, tok, ord_seqs, FACT_LAYER)).float()
        ordinary = ordinary[torch.randperm(ordinary.shape[0], generator=torch.Generator().manual_seed(SEED))[:N_ORDINARY].to(ordinary.device)]
        return tagged, other, ordinary

    def readouts(tagged, untagged, other, ordinary):
        rows, stats = {}, {}
        for f in facts:
            pos, base = tagged[f["id"]], untagged[f["id"]]
            mates = torch.cat([tagged[h["id"]][:8] for h in facts if h["id"] != f["id"]])
            groups_ = [pos, base, mates, other[f["relation"]], ordinary]
            within = torch.cat([x - x.mean(0) for x in groups_])
            Sm = cov20 + within.T @ within / within.shape[0]
            w = torch.linalg.solve(Sm + RIDGE * Sm.diagonal().mean() * torch.eye(Sm.shape[0], device=dev), pos.mean(0) - base.mean(0))
            rp = ratio(pos, w, c20, cbar20)
            rn = {"untagged": ratio(base, w, c20, cbar20), "mates": ratio(mates, w, c20, cbar20), "others": ratio(other[f["relation"]], w, c20, cbar20),
                  "ordinary": ratio(ordinary, w, c20, cbar20)}
            alln = torch.cat(list(rn.values()))
            b1, b0 = float(rp.median()), float(torch.quantile(alln, 0.99))
            sep = (b1 - b0) / float(alln.std())
            ok = sep >= SEP_MIN
            stats[f["id"]] = dict(separation=round(sep, 2), placed=bool(ok), **{f"{k}_max": round(float(((v - b0) / (b1 - b0)).max()), 3) for k, v in rn.items()})
            rows[f["id"]] = (w.double() - b0 * c20.double() / cbar20) / (b1 - b0) if ok else None
        return rows, stats

    def place_facts(rows_t):
        gate, up, down = G20.clone(), U20.clone(), freed20.clone()
        for j, f in enumerate(facts):
            tr = rows_t[f["id"]]
            if tr is None:
                continue
            d = det[f["id"]]
            gate[slots20[j]] = (GAIN * (d["w"].to(dev).double() - (d["tau"] / cbar20) * c20.double()) / d["delta"]).to(G20.dtype)
            up[slots20[j]] = tr.to(U20.dtype)
            sp = parts[f["id"]]["spellings"]["spaced"] or parts[f["id"]]["spellings"]["bare"]
            down[:, slots20[j]] = PUSH * sp["vecs"][0].to(dev).double() / silu_k
        mlp20.gate_proj.weight.data.copy_(gate)
        mlp20.up_proj.weight.data.copy_(up)
        mlp20.down_proj.weight.data.copy_(down.to(D20.dtype))

    def summary_line(name, stats):
        st = list(stats.values())
        placed = [s for s in st if s["placed"]]
        return (f"  readings ({name}): placed {len(placed)} of {len(st)}, median separation {torch.tensor([s['separation'] for s in st]).median():.1f} sd"
                + (f"; among placed, largest: untagged {max(s['untagged_max'] for s in placed):.2f}, other edits' tags "
                   f"{max(s['mates_max'] for s in placed):.2f}, other subjects {max(s['others_max'] for s in placed):.2f}, ordinary text "
                   f"{max(s['ordinary_max'] for s in placed):.2f}" if placed else ""))

    import contextlib

    @contextlib.contextmanager
    def hook_tags():
        hook = TagHook(model, tok, subjects, tags, NAME_LAYER)
        try:
            yield
        finally:
            hook.remove()

    out = dict(edits=[f["id"] for f in facts], channel=dict(layer=L, query_head=qh, kv_head=g, attention=att_stats), tag_unit=unit8,
               value_norm_median=v_med, A=s_a, unedited=step2["unedited"])
    try:
        # ================= D3: tags by a hook at the exact name; gamma chosen on the training wordings
        log("D3: tags written by a hook from the name index, carried by the channel")
        c1 = code_norm_at_names(hook_tags)
        chosen_gamma, rows_t, stats_d = None, None, None
        for gm in GAMMAS:
            gamma = gm * v_med / c1
            set_channel(gamma)
            untagged = {f["id"]: last_reads(model, tok, plain_of[f["id"]], FACT_LAYER) for f in facts}   # the channel, no tags
            with hook_tags():
                tagged, other, ordinary = gather()
            r, st = readouts(tagged, untagged, other, ordinary)
            med_sep = float(torch.tensor([s["separation"] for s in st.values()]).median())
            log(f"  gamma {gm:g} x (median value norm / code norm at the names), separation median {med_sep:.1f} sd | {time.time() - t0:.0f}s")
            chosen_gamma, rows_t, stats_d = gm, r, st
            if med_sep >= SEP_MIN:
                break
        gamma = chosen_gamma * v_med / c1
        out["gamma"] = dict(multiple=chosen_gamma, value=gamma)
        log(summary_line("D3", stats_d))
        set_channel(gamma)
        place_facts(rows_t)
        with hook_tags():
            rows_d, s_d = score(model, tok, facts, before, baseline, cf_check)
        s_d["same_first_words_changed"], s_d["neigh_first_words_changed"] = changed(rows_d, before, "same"), changed(rows_d, before, "neigh")
        log(line("D3, tags from the name index", s_d) + f" | first two words changed: same subject {s_d['same_first_words_changed']:.3f}, "
            f"neighbours {s_d['neigh_first_words_changed']:.3f} | {time.time() - t0:.0f}s")
        restore()

        # ================= E3: name neurons at layer 8
        log("E3: name neurons in layer 8 write the tags; the channel carries them")
        mlp8.down_proj.weight.data.copy_(freed8.to(D8.dtype))
        mlp20.down_proj.weight.data.copy_(freed20.to(D20.dtype))
        set_channel(gamma)
        out["freed_and_channel"] = damage()
        log(f"  layers 8 and 20 freed, channel in place, no edits: known lost {out['freed_and_channel']['known_lost']} of 133, "
            f"{out['freed_and_channel']['cf_lost']} of 391, drift {out['freed_and_channel']['drift']:.5f}")
        restore()
        prompt_sets = {f["id"]: plain_of[f["id"]] + same_of[f["id"]] + chat_of[f["id"]] for f in facts}
        nrows = name_rows(model, tok, facts, prompt_sets, paras, NAME_LAYER, c8, cbar8, cov8, dev)
        silu_n = float(F.silu(torch.tensor(K_NAME * Z_MED)))
        gate8, up8, down8 = G8.clone(), U8.clone(), freed8.clone()
        for j, row in enumerate(nrows):
            if row is None:
                down8[:, slots8[j]] = 0
                continue
            w, tau, delta = row
            gate8[slots8[j]] = (K_NAME * (w - (tau / cbar8) * c8.double()) / delta).to(G8.dtype)
            up8[slots8[j]] = (c8.double() / cbar8).to(U8.dtype)
            down8[:, slots8[j]] = tags[j].double() / silu_n
        log(f"  name detectors separable for {sum(r is not None for r in nrows)} of {n} | {time.time() - t0:.0f}s")
        set_channel(gamma)
        mlp8.gate_proj.weight.data.copy_(gate8)
        mlp8.up_proj.weight.data.copy_(up8)
        mlp8.down_proj.weight.data.copy_(freed8.to(D8.dtype))     # untagged: the name neurons silent
        untagged_e = {f["id"]: last_reads(model, tok, plain_of[f["id"]], FACT_LAYER) for f in facts}
        mlp8.down_proj.weight.data.copy_(down8.to(D8.dtype))
        tagged, other, ordinary = gather()
        rows_t, stats_e = readouts(tagged, untagged_e, other, ordinary)
        log(summary_line("E3", stats_e))
        place_facts(rows_t)
        log("E3: scoring the grafted model (no hook)")
        rows_e, s_e = score(model, tok, facts, before, baseline, cf_check)
        s_e["same_first_words_changed"], s_e["neigh_first_words_changed"] = changed(rows_e, before, "same"), changed(rows_e, before, "neigh")
        log(line("E3, grafted: name neurons and the channel", s_e) + f" | first two words changed: same subject {s_e['same_first_words_changed']:.3f}, "
            f"neighbours {s_e['neigh_first_words_changed']:.3f} | {time.time() - t0:.0f}s")
    finally:
        restore()

    fb = out["freed_and_channel"]

    def works(s):
        return dict(es=s["es"] >= s_a["es"] - 0.05, ps=s["ps"] >= s_a["ps"] - 0.05, ns=s["ns"] >= s_a["ns"] - 0.01, neigh_leak=s["neigh_leak"] <= 0.05,
                    same_leak=s["same_leak"] <= 0.10, same_changed=s["same_first_words_changed"] <= 0.10,
                    known_beyond_freed=s["known_lost"] - fb["known_lost"] <= 2, cf_beyond_freed=s["cf_lost"] - fb["cf_lost"] <= 7, drift=s["drift"] <= 0.05)
    vd, ve = works(s_d), works(s_e)
    ships = all(ve.values()) and s_e["known_lost"] <= 2 and s_e["cf_lost"] <= 7
    out.update(D3=s_d, E3=s_e, verdict_D3=vd, verdict_E3=ve, channel_works=all(ve.values()), ships_weights_only=ships, readings_D3=stats_d,
               readings_E3=stats_e)
    log(f"D3 (tags from the name index): the channel works: {all(vd.values())} {vd}")
    log(f"E3 (no hook): the channel works: {all(ve.values())} {ve}")
    log(f"E3 ships as a weights-only grafted model (also at most 2 of 133 and 7 of 391 lost in all): {ships}")
    log(f"kill rule (E3 fails the channel bar): {'MET: the attention channel is put aside' if not all(ve.values()) else 'not met'}")
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for name, rows in (("D3", rows_d), ("E3", rows_e)):
        with open(os.path.join(a.out, f"edits_{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

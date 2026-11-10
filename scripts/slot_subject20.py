"""Step 5 of the grafted model: a bounded subject reading at layer 20 (lab notebook, "Step 5: a bounded subject reading at layer 20").

Each edit's subject reading u is an LDA at layer 20's MLP input, built as the name neurons were: positives are the relation's training
wordings about the subject at the last token; negatives are the same wordings about other subjects (same prefixes), the other edits'
positives, and every token of ordinary text; the threshold is the largest negative (u = 0 there, 1 at the positives' median). The fact
neuron is a clipped pair: gate k (z_q + M (u - 1)), silu(g) - silu(g - 8), up c/cbar, down 2v/8, so it writes at most the inserted
layer's push. No hook. Step 2's 64 edits and its inserted-layer reference.

Usage: python scripts/slot_subject20.py --model <folder> --run runs/counterfact/phrasing_editor --relations <editor relations>
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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_slot_compensate import SPARE, ScreenCheck, compensation, redundancy_order, second_moment  # noqa: E402
from slot_compile import line, score  # noqa: E402
from slot_names import all_reads, last_reads  # noqa: E402

from cramschool.bench.counterfact import load_counterfact  # noqa: E402
from cramschool.bench.methods import calibrate  # noqa: E402
from cramschool.bench.scoring import make_baseline  # noqa: E402
from cramschool.facts import PREFIXES  # noqa: E402
from cramschool.graft.build import build_context, fact_detectors, guarded, kind_parts, plain_items  # noqa: E402
from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_HELD, CALIBRATION_MAX_TOKENS, blocks, lda, ratio  # noqa: E402
from cramschool.graft.wordings import Wordings, calibration_paragraphs, norm  # noqa: E402
from cramschool.model_access import Prompt  # noqa: E402

LAYER, GAIN, M_SUBJ, CLIP, PUSH = 20, 8.0, 4.0, 8.0, 2.0
N_TEMPLATES, N_INVENTED, N_REAL, N_PREFIX_NEG, N_PARAS_ORD, SEED = 16, 12, 24, 2, 120, 0


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


def q(t, p):
    return float(torch.quantile(t.float(), p)) if t.numel() else float("nan")


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
    n = len(facts)
    log(f"{n} edits (step 2's); A, the inserted layer: ES {s_a['es']:.3f} PS {s_a['ps']:.3f} NS {s_a['ns']:.3f}")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    dev = model.device
    baseline = make_baseline(model, tok)
    cf_check = ScreenCheck(model, tok, items=[(r["prompt"], r["answers"]) for r in step1c["cf_known"]])
    m20 = calibrate(model, tok, LAYER)
    c, cbar, cov = m20["c"].to(dev).float(), m20["cbar"], m20["cov"].to(dev).float()
    paras = calibration_paragraphs()

    def damage():
        lost = sum(b and not x for b, x in zip(baseline.known_check.before, baseline.known_check.correct()))
        lost_cf = sum(b and not x for b, x in zip(cf_check.before, cf_check.correct()))
        return dict(known_lost=lost, cf_lost=lost_cf, drift=baseline.drift()[0])

    # ---- prompts: the subject's wordings, and the same wordings about other subjects with the same prefixes
    guarded_w = Wordings(relations=wordings.relations, generic=wordings.generic, kinds=wordings.kinds,
                         guard=set(wordings.guard) | {norm(p) for f in facts for p in f["cf"]["neighbours"] + f["cf"].get("twin_prompts", [])})
    ctx = build_context(facts, guarded_w, name="counterfact", real_world="other_relations", context=4, twins=twins)
    prefixes = tuple(PREFIXES) + tuple(ctx.get("context", ()))
    templates = {}
    for rel in sorted({f["relation"] for f in facts}):
        fs = [f for f in facts if f["relation"] == rel]
        templates[rel] = list(dict.fromkeys(t for f in fs for t in f["train"]["templates_plain"])) + list(ctx["drafted"][rel][0][:N_TEMPLATES])
    pos_prompts = {f["id"]: guarded(plain_items(templates[f["relation"]], [f["subject"]], prefixes), ctx["guard"]) for f in facts}
    tests = " | ".join(p.lower() for f in facts for p in [f["cf"]["rewrite"]] + f["cf"]["paraphrases"] + f["cf"]["neighbours"]
                       + f["cf"].get("twin_prompts", []) + f["test"]["same_plain"])
    records = load_counterfact(a.data)
    chosen = {f["subject"].lower() for f in facts}
    rng = random.Random(SEED)
    neg_prompts = {}
    for rel in templates:
        inv = kind_parts(ctx, rel)[0][:N_INVENTED]
        pool = sorted({r["requested_rewrite"]["subject"] for r in records if r["requested_rewrite"]["relation_id"] == cf_ids[rel]})
        pool = [s for s in pool if s.lower() not in tests and s.lower() not in chosen]
        names = inv + random.Random(SEED).sample(pool, min(N_REAL, len(pool)))
        items = [(p + t.format(s=nm), Prompt(p + t.format(s=nm))) for nm in names for t in templates[rel] for p in rng.sample(prefixes, N_PREFIX_NEG)]
        neg_prompts[rel] = guarded(items, ctx["guard"])
    log(f"prompts: {sum(map(len, pos_prompts.values()))} about the edits' subjects, {sum(map(len, neg_prompts.values()))} about other subjects "
        f"(prefixes as the positives') | {time.time() - t0:.0f}s")

    pos = {f["id"]: last_reads(model, tok, pos_prompts[f["id"]], LAYER) for f in facts}
    neg = {rel: last_reads(model, tok, ps, LAYER) for rel, ps in neg_prompts.items()}
    ordinary = torch.cat(all_reads(model, tok, [tok(paras[i], truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids
                                                for i in list(CALIBRATION_FIT)[:N_PARAS_ORD]], LAYER)).float()
    scored = {f["id"]: {g: last_reads(model, tok, [Prompt(p) for p in ps], LAYER) for g, ps in
                        (("rewrite", [f["cf"]["rewrite"]]), ("para", f["cf"]["paraphrases"]), ("neigh", f["cf"]["neighbours"]))} for f in facts}
    log(f"reads: {ordinary.shape[0]} ordinary tokens | {time.time() - t0:.0f}s")

    # ---- the subject readings
    u_rows, stats = {}, {}
    all_pos = torch.cat([pos[f["id"]] for f in facts])
    owner = torch.cat([torch.full((pos[f["id"]].shape[0],), j) for j, f in enumerate(facts)]).to(dev)
    for j, f in enumerate(facts):
        mates = all_pos[owner != j]
        groups = [neg[f["relation"]], mates, ordinary]
        w = lda(pos[f["id"]], groups, cov)
        rp = ratio(pos[f["id"]], w, c, cbar)
        rn = torch.cat([ratio(g, w, c, cbar) for g in groups])
        tau, med = float(rn.max()), float(rp.median())
        st = dict(placed=med > tau, separation=round((med - float(torch.quantile(rn, 0.99))) / float(rn.std()), 2),
                  positives_above=round(float((rp > tau).float().mean()), 3))
        if med > tau:
            u_rows[f["id"]] = (w.double() - tau * c.double() / cbar) / (med - tau)
            uf = lambda x: (ratio(x, w, c, cbar) - tau) / (med - tau)  # noqa: E731
            st.update(u_rewrite=round(float(uf(scored[f["id"]]["rewrite"])[0]), 3), u_para=[round(float(v), 3) for v in uf(scored[f["id"]]["para"])],
                      u_neigh=[round(float(v), 3) for v in uf(scored[f["id"]]["neigh"])])
        stats[f["id"]] = st
    placed = [s for s in stats.values() if s["placed"]]
    up_ = torch.tensor([v for s in placed for v in s["u_para"]])
    un_ = torch.tensor([v for s in placed for v in s["u_neigh"]])
    log(f"subject readings: placed {len(placed)} of {n} (positives' median above the largest negative); separation median "
        f"{torch.tensor([s['separation'] for s in stats.values()]).median():.1f} sd | on the test prompts: reworded u median {q(up_, 0.5):.2f} "
        f"(10th percentile {q(up_, 0.1):.2f}), neighbours u median {q(un_, 0.5):.2f}, 95th {q(un_, 0.95):.2f}, largest {float(un_.max()):.2f}, "
        f"share above 0.5 {float((un_ > 0.5).float().mean()):.3f} | {time.time() - t0:.0f}s")

    # ---- 128 slots at layer 20, freed and compensated; the clipped pairs
    mlp = blocks(model)[LAYER].mlp
    G0, U0, D0 = mlp.gate_proj.weight.data.clone(), mlp.up_proj.weight.data.clone(), mlp.down_proj.weight.data.clone()
    fit_texts, spare_texts = [paras[i] for i in CALIBRATION_FIT], [paras[i] for i in SPARE]
    Hf, _, _ = second_moment(model, tok, mlp.down_proj, fit_texts)
    Hv, _, _ = second_moment(model, tok, mlp.down_proj, spare_texts)
    slots = redundancy_order(Hf, D0.double(), 2 * n)
    K, Dc, ridge, left, _ = compensation(Hf, Hv, D0.double(), slots)
    del Hf, Hv
    freed = D0.double()
    freed[:, slots] = 0
    freed[:, K] += Dc
    gate, up, down = G0.clone(), U0.clone(), freed.clone()
    one = (c.double() / cbar)
    for j, f in enumerate(facts):
        ua, ub = slots[2 * j], slots[2 * j + 1]
        if f["id"] not in u_rows:
            continue  # its pair stays freed: down columns 0
        d = next(d for d in fact_detectors(parts[f["id"]]) if d["form"] == "plain")
        zq = (d["w"].to(dev).double() - (d["tau"] / cbar) * c.double()) / d["delta"]
        g = GAIN * (zq + M_SUBJ * (u_rows[f["id"]] - one))
        gate[ua], gate[ub] = g.to(G0.dtype), (g - CLIP * one).to(G0.dtype)
        up[ua] = up[ub] = one.to(U0.dtype)
        sp = parts[f["id"]]["spellings"]["spaced"] or parts[f["id"]]["spellings"]["bare"]
        v = PUSH * sp["vecs"][0].to(dev).double() / CLIP
        down[:, ua], down[:, ub] = v, -v
    out = dict(edits=[f["id"] for f in facts], slots=slots, refit=dict(ridge=ridge, energy_left=left), readings=stats, A=s_a, unedited=step2["unedited"],
               settings=dict(gain=GAIN, m=M_SUBJ, clip=CLIP, push=PUSH))
    try:
        mlp.down_proj.weight.data.copy_(freed.to(D0.dtype))
        out["freed_only"] = damage()
        log(f"the 128 slots freed alone: known lost {out['freed_only']['known_lost']} of 133, {out['freed_only']['cf_lost']} of 391, drift "
            f"{out['freed_only']['drift']:.5f} | {time.time() - t0:.0f}s")
        mlp.gate_proj.weight.data.copy_(gate)
        mlp.up_proj.weight.data.copy_(up)
        mlp.down_proj.weight.data.copy_(down.to(D0.dtype))

        # before scoring: how often a pair writes on held-out ordinary text
        got = {}
        pairs = torch.tensor(slots, device=dev).view(-1, 2)
        h = mlp.down_proj.register_forward_pre_hook(lambda m, a_: got.__setitem__("a", a_[0][0].float()))
        fired, tokens, top = 0, 0, 0.0
        try:
            with torch.no_grad():
                for i in CALIBRATION_HELD:
                    ids = tok(paras[i], return_tensors="pt", truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids.to(dev)
                    model(input_ids=ids, use_cache=False)
                    share = (got["a"][:, pairs[:, 0]] - got["a"][:, pairs[:, 1]]) / CLIP      # tokens x edits, 1 = the full push
                    fired += int((share.max(1).values > 0.1).sum())
                    tokens += share.shape[0]
                    top = max(top, float(share.max()))
        finally:
            h.remove()
        out["held_out"] = dict(tokens=tokens, tokens_with_a_pair_over_a_tenth=fired, largest_share_of_push=round(top, 4))
        log(f"held-out text: {fired} of {tokens} tokens have a pair writing over a tenth of its push; the largest is {top:.2f} of a push")

        log("scoring the grafted model (no hook)")
        rows, s = score(model, tok, facts, before, baseline, cf_check)
        s["same_first_words_changed"], s["neigh_first_words_changed"] = changed(rows, before, "same"), changed(rows, before, "neigh")
        log(line("step 5, bounded subject reading", s) + f" | first two words changed: same subject {s['same_first_words_changed']:.3f}, "
            f"neighbours {s['neigh_first_words_changed']:.3f} | {time.time() - t0:.0f}s")
        torch.save(dict(layer=LAYER, slots=slots, gate=gate.cpu(), up=up.cpu(), down=down.to(D0.dtype).cpu()), os.path.join(a.out, "layer20.pt"))
    finally:
        mlp.gate_proj.weight.data.copy_(G0)
        mlp.up_proj.weight.data.copy_(U0)
        mlp.down_proj.weight.data.copy_(D0)

    fo = out["freed_only"]
    works = dict(es=s["es"] >= s_a["es"] - 0.05, ps=s["ps"] >= s_a["ps"] - 0.05, ns=s["ns"] >= s_a["ns"] - 0.01, neigh_leak=s["neigh_leak"] <= 0.05,
                 same_leak=s["same_leak"] <= 0.10, same_changed=s["same_first_words_changed"] <= 0.10, drift=s["drift"] <= 0.05,
                 known=s["known_lost"] <= fo["known_lost"] + 2, cf=s["cf_lost"] <= fo["cf_lost"] + 7)
    ships = all(works.values()) and s["known_lost"] <= 2 and s["cf_lost"] <= 7
    out.update(result=s, verdict=works, works=all(works.values()), ships_weights_only=ships)
    log(f"the reading works as key 1: {all(works.values())} {works}")
    log(f"ships as a weights-only grafted model (also at most 2 of 133 and 7 of 391 in all): {ships}")
    log(f"kill rule (fails the first bar): {'MET: the subject reading at layer 20 is put aside' if not all(works.values()) else 'not met'}")
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(a.out, "edits.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

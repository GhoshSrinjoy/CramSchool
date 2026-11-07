"""Arm "split_gate" (plan.md, "Split gate: the build"). Forward passes and closed-form fits only; one fact at a time from the unedited model.

The first answer token is pushed only when the prompt asks the fact's question AND its company carries the name neuron's tag:
  Part 1  the name neuron of tagged_arm.py at layer 4 (unchanged): fires at the company's last token and writes a tag.
  Part 2  per first-token link (plain spelling, chat spelling), a capped pair of reused layer-20 neurons. A gated MLP neuron computes
          SiLU(gate . x) * (up . x), so the pair computes [question detector over its threshold] x [tag readout] x push:
            gate rows  a (w_q - tau c/cbar), a (w_q - (tau + delta) c/cbar)    question detector, thresholded and capped
            up rows    (w_t - m0 c/cbar) / (m1 - m0), twice                   tag readout: about 1 with the tag, about 0 without
            down cols  +v/8, -v/8, with v = 1.25 x the link's push (pushes capped at 10x the fact's median, as in gated_first v2)
  Later answer tokens use gated_first's linear write (build_continuation), unchanged.

Question detector (w_q, tau, delta), read on the unedited model at layer 20's MLP input, last token:
  positives  the link's training wordings (the bench's and, with BENCH_TEACH, the drafted ones) asked about 24 invented companies and the
             fact's own company; the fact's own plain wordings also with the package's four prefixes, as taught
  negatives  prompts: the fact's same-company training questions and the package's negative prompts asked about the same companies, the
             other relations' drafted wordings asked about 6 of them, and gated_first v3's 24 real-world questions;
             inside the answer: the reads at every later answer token of the fact's teaching examples
  address    LDA (gated_arm.lda); the tag's arrival direction at this site is projected out, so the tag does not move the score
  threshold  tau = the larger of the 99th percentile of the prompt negatives and the highest inside-answer read; full push at
             delta = half the distance from tau to the median positive. A link whose median positive is not above tau gets no gate.
Tag readout (w_t, m0, m1), fitted across 24 invented companies asked the bench's training wordings: direction = (scatter)^-1 (mean read with
the name neuron's tag added at the company's name - mean read without it), where scatter is the ordinary-text covariance plus the scatter
within the two groups, with gated_arm's ridge; m1 and m0 are the two groups' mean scores. Reported on the fact's own prompts with Part 1
(the real name neuron) and on untagged prompts (the same prompts without Part 1, and other companies on which the name neuron does not fire).

Usage: python -I split_gate.py <model dir> <calibration txt> <transport.json> <sessions dir> [start] [end]   (BENCH_NAME, BENCH_TEACH)
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
import diag_wordings as DW  # noqa: E402
import gated_arm as G  # noqa: E402
import gated_first as GF  # noqa: E402
import run_arm as R  # noqa: E402
import separability as S  # noqa: E402
import subject_site as SS  # noqa: E402
import tagged_arm as T  # noqa: E402
from weightspace.direct import answer_chain  # noqa: E402
from weightspace.facts import NEGATIVE_PROMPTS, NEGATIVE_QUESTIONS, PREFIXES  # noqa: E402
from weightspace.isolation import applied  # noqa: E402
from weightspace.model_access import Prompt, encode, render_chat  # noqa: E402
from weightspace.surgery import Surgery  # noqa: E402

LQ = GF.LQ
ARM = "split_gate" + B.ARM_SUFFIX
SHARP, PUSH_CAP, NEG_QUANTILE, SATURATION = 8.0, 10.0, 0.99, 0.5
N_OTHER_REL_COMPANIES = 6


def templates(f, form):
    t = f["train"]
    plain_t, chat_t = B.teaching_templates(f["relation"])
    return t["templates_plain"] + plain_t if form == "plain" else t["templates_chat"] + chat_t


def detector_prompts(f, tokenizer, companies, form):
    """Positives for one spelling: the training wordings asked about every company; the fact's own plain ones also with the prefixes."""
    out = []
    for n in companies:
        for tpl in templates(f, form):
            if form == "plain":
                out += [Prompt(p + tpl.format(s=n)) for p in (PREFIXES if n == f["subject"] else ("",))]
            else:
                out.append(render_chat(tokenizer, tpl.format(s=n)))
    return out


def detector_negatives(f, tokenizer, companies):
    tr, subj = f["train"], f["subject"]
    same_plain = [p.replace(subj, "{s}") for p in tr["same_plain"]] + list(NEGATIVE_PROMPTS)
    same_chat = [q.replace(subj, "{s}") for q in tr["same_chat"]] + list(NEGATIVE_QUESTIONS)
    out = [Prompt(t.format(s=n)) for n in companies for t in same_plain] + [render_chat(tokenizer, q.format(s=n)) for n in companies for q in same_chat]
    for rel in ("owner", "ceo", "founder", "hq"):
        if rel != f["relation"]:
            plain_t, chat_t = B.teaching_templates(rel)
            for n in companies[:N_OTHER_REL_COMPANIES]:
                out += [Prompt(t.format(s=n)) for t in plain_t] + [render_chat(tokenizer, q.format(s=n)) for q in chat_t]
    out += [Prompt(p) for p in GF.REAL_WORLD_PLAIN] + [render_chat(tokenizer, q) for q in GF.REAL_WORLD_CHAT]
    return out


@torch.no_grad()
def tagged_last_reads(model, tokenizer, items, tag, batch: int = 32):
    """Layer 20's MLP input at each prompt's last token, with `tag` added to layer 4's MLP output at the last token of the prompt's company
    (what the name neuron writes when it fires). Right padding, so positions are those of the unpadded prompt."""
    from weightspace.model_access import batch_ids, token_index_of

    seen, out, where = {}, [], {}

    def add_tag(_m, _a, output):
        output = output.clone()
        output[where["rows"], where["pos"]] += tag.to(output.dtype)
        return output

    h1 = S.blocks(model)[T.LS].mlp.register_forward_hook(add_tag)
    h2 = S.blocks(model)[LQ].mlp.register_forward_pre_hook(lambda m, a: seen.__setitem__("x", a[0].float()))
    try:
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            seqs = [encode(tokenizer, p) for p, _ in chunk]
            ids, mask = batch_ids(tokenizer, seqs, "cuda")
            where["rows"] = torch.arange(len(chunk), device="cuda")
            where["pos"] = torch.tensor([token_index_of(tokenizer, p, c) for p, c in chunk], device="cuda")
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            out += [seen["x"][i, len(q) - 1] for i, q in enumerate(seqs)]
    finally:
        h1.remove()
        h2.remove()
    return torch.stack(out)


def build(s, model, tokenizer, f, ctx, names):
    """Parts 1 and 2 and the later-token write for one fact. Returns (updates applied at 1x, linear write applied at 1.25x, info)."""
    c, cbar = ctx["const20"]
    p1, info1, fires = DW.name_neuron(model, tokenizer, f, ctx)
    built, leave = R.training_fact(f, chat=True), R.training_fact(f, chat=True, teach=False)
    examples = built.teaching_examples(tokenizer)
    n_plain = len(built.prompts) * len(PREFIXES)
    sequences, keys, spots = GF.keyed_spots(tokenizer, examples)
    first = [i for i, k in enumerate(keys) if k[0] == 0]
    later = [i for i, k in enumerate(keys) if k[0] > 0]
    companies = names + [f["subject"]]

    # Unedited reads: detector positives and negatives, the inside-answer reads, and the fact's own prompts without the tag.
    neg_x = DW.last_inputs(model, tokenizer, detector_negatives(f, tokenizer, companies))
    xt_clean = GF.mlp_inputs(model, tokenizer, sequences, LQ)
    inside = torch.stack([xt_clean[w][p] for j in later for w, p in spots[j]]) if later else neg_x[:0]
    own_prompts = [p for p, _ in examples]
    own_clean = DW.last_inputs(model, tokenizer, own_prompts)
    pos_x = {form: DW.last_inputs(model, tokenizer, detector_prompts(f, tokenizer, companies, form)) for form in ("plain", "chat")}

    # Tag readout, with Part 1 in place: the fact's own prompts against the same wordings asked about companies without the tag.
    others = names + f["train"]["lookalikes"] + f["train"]["others"]
    other_items = [(Prompt(t.format(s=n)), n) for n in others for t in f["train"]["templates_plain"]] + \
                  [(render_chat(tokenizer, q.format(s=n)), n) for n in others for q in f["train"]["templates_chat"]]
    untagged_mask = ~fires(SS.subject_reads(model, tokenizer, other_items)[T.LS])
    with applied(model, p1):
        links = answer_chain(model, tokenizer, examples, LQ)
        own_tagged = DW.last_inputs(model, tokenizer, own_prompts)
        other_tagged = DW.last_inputs(model, tokenizer, [p for p, _ in other_items])
    # The readout is fitted across the 24 invented companies: the bench's training wordings asked about each, read with the tag added at
    # the company's name and without it. Only the tag separates the two groups, and 24 companies' variation is what it learns to ignore
    # (fitting it on the fact's own company alone left it unable to tell look-alikes apart; smoke test, plan.md).
    others_untagged = other_tagged[untagged_mask]
    untagged = torch.cat([own_clean, others_untagged])
    negs_in = [neg_x] + ([inside] if inside.shape[0] else [])
    fit_items = [(Prompt(t.format(s=n)), n) for n in names for t in templates(f, "plain")] + \
                [(render_chat(tokenizer, q.format(s=n)), n) for n in names for q in templates(f, "chat")]
    x_tag = tagged_last_reads(model, tokenizer, fit_items, ctx["tag"])
    x_cln = DW.last_inputs(model, tokenizer, [p for p, _ in fit_items])
    # Untagged prompts of other kinds (the detector's negatives: other questions, real-world questions) add variation to ignore.
    within = torch.cat([x_tag - x_tag.mean(0), x_cln - x_cln.mean(0), neg_x - neg_x.mean(0)])
    scatter = ctx["cov20"] + within.T @ within / within.shape[0]
    w_t = torch.linalg.solve(scatter + S.RIDGE * scatter.diagonal().mean() * torch.eye(scatter.shape[0], device=scatter.device),
                             x_tag.mean(0) - x_cln.mean(0))
    m1, m0 = float(GF.ratio(x_tag, w_t, c, cbar).mean()), float(GF.ratio(x_cln, w_t, c, cbar).mean())
    up = (w_t - m0 * c / cbar) / (m1 - m0)
    arrival = (own_tagged - own_clean).mean(0)
    d = arrival / arrival.norm()

    norms = [float(link.vector.norm()) for link in links]
    cap = PUSH_CAP * statistics.median(norms)
    capped = []
    for i, link in enumerate(links):
        if norms[i] > cap:
            link.vector = link.vector * (cap / norms[i])
            capped.append(tokenizer.decode([link.token]))

    free = list(ctx["free20"])
    pairs, gates, gated_first = [], [], []
    for i in first:
        forms = {"plain" if w < n_plain else "chat" for w, _ in spots[i]}
        pos = torch.cat([pos_x[form] for form in sorted(forms)])
        w_q = G.lda(pos, negs_in, ctx["cov20"])
        w_q = w_q - (w_q @ d) * d  # the tag must not move the question score
        r_pos, r_neg, r_in = (GF.ratio(x, w_q, c, cbar) for x in (pos, neg_x, inside))
        tau = max(float(torch.quantile(r_neg, NEG_QUANTILE)), float(r_in.max()) if r_in.numel() else -1e9)
        med = float(r_pos.median())
        tag_shift = float((GF.ratio(own_tagged, w_q, c, cbar) - GF.ratio(own_clean, w_q, c, cbar)).mean())
        info = dict(token=tokenizer.decode([links[i].token]), forms=sorted(forms), positives=int(pos.shape[0]), negatives=int(neg_x.shape[0] + inside.shape[0]),
                    tau=round(tau, 4), median_positive=round(med, 4), positives_over=round(float((r_pos > tau).float().mean()), 4),
                    negatives_over=round(float((r_neg > tau).float().mean()), 4), highest_negative=round(float(r_neg.max()), 4),
                    tag_shift_in_deltas=None, tag_shift=round(tag_shift, 4))
        if med <= tau:
            info["gated"] = False
            gates.append(info)
            continue
        delta = SATURATION * (med - tau)
        a = SHARP / delta
        v = R.STRENGTH * links[i].vector.float()
        j1, j2 = free.pop(0), free.pop(0)
        gate = torch.stack([a * (w_q - tau * c / cbar), a * (w_q - (tau + delta) * c / cbar)])
        pairs += [G.replace_rows(model, LQ, "gate_proj", [j1, j2], gate), G.replace_rows(model, LQ, "up_proj", [j1, j2], torch.stack([up, up])),
                  G.replace_cols(model, LQ, "down_proj", [j1, j2], torch.stack([v / SHARP, -v / SHARP], 1))]
        info.update(gated=True, neurons=[j1, j2], delta=round(delta, 4), push_norm=round(float(v.norm()), 3), w=w_q, tau_raw=tau, delta_raw=delta,
                    tag_shift_in_deltas=round(tag_shift / delta, 3))
        gated_first.append(i)
        gates.append(info)
    used = [j for g in gates for j in g.get("neurons", [])]
    first_keys = torch.cat([links[i].keys for i in gated_first], 1) if gated_first else None
    linear_links = [links[i] for i in later]
    with applied(model, p1):  # the later-token write is built with Part 1 in place, as in gated_first
        linear = GF.build_continuation(s, model, tokenizer, built, leave, linear_links, first_keys, R.bench_negatives(f, tokenizer), used) if linear_links else None
    u_own, u_untagged = own_tagged @ up, untagged @ up
    tag_info = dict(m1=round(m1, 4), m0=round(m0, 4), fit_pairs=len(fit_items), own_tagged=int(own_tagged.shape[0]), untagged=int(untagged.shape[0]),
                    readout_own_tagged_median=round(float(u_own.median()), 3), readout_own_tagged_p10=round(float(torch.quantile(u_own, 0.1)), 3),
                    readout_untagged_p99=round(float(torch.quantile(u_untagged, 0.99)), 3), readout_untagged_max=round(float(u_untagged.max()), 3))
    return p1 + pairs, linear, dict(part1=info1, gates=gates, tag=tag_info, up=up, capped=capped, links=len(links))


def firing(model, tokenizer, f, p1, info):
    """With Part 1 in place: for each test group, the share of prompts where a gate's question detector is over its threshold, the mean tag
    readout, and the share where the pair would give at least half its push (saturation x tag readout >= 0.5)."""
    c, cbar = info["c"], info["cbar"]
    out = {}
    tp = B.test_prompts(f, tokenizer)
    with applied(model, p1):
        for g, prompts in tp.items():
            if not prompts:
                continue
            x = DW.last_inputs(model, tokenizer, prompts)
            u = x @ info["up"]
            sat = torch.zeros(x.shape[0], device=x.device)
            q_open = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            for gate in info["gates"]:
                if gate.get("gated"):
                    r = GF.ratio(x, gate["w"], c, cbar)
                    q_open |= r > gate["tau_raw"]
                    sat = torch.maximum(sat, ((r - gate["tau_raw"]) / gate["delta_raw"]).clamp(0, 1))
            out[g] = dict(question=round(float(q_open.float().mean()), 3), tag=round(float(u.mean()), 3), push_half=round(float((sat * u >= 0.5).float().mean()), 3))
        # The known-fact prompts: none carries the tag, so the pair should stay shut on them.
        with open(B.KNOWN_FILE, encoding="utf-8") as fh:
            known = [Prompt(d["prompt"]) for d in json.load(fh)["facts"]]
        x = DW.last_inputs(model, tokenizer, known)
        u = x @ info["up"]
        sat = torch.zeros(x.shape[0], device=x.device)
        for gate in info["gates"]:
            if gate.get("gated"):
                sat = torch.maximum(sat, ((GF.ratio(x, gate["w"], c, cbar) - gate["tau_raw"]) / gate["delta_raw"]).clamp(0, 1))
        out["known_facts"] = dict(question=round(float((sat > 0).float().mean()), 3), tag_p95=round(float(torch.quantile(u, 0.95)), 3),
                                  push_half=round(float((sat * u >= 0.5).float().mean()), 3))
    return out


def main():
    model_dir, calibration, transport_json, sessions = sys.argv[1:5]
    start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    end = int(sys.argv[6]) if len(sys.argv) > 6 else 40
    sys.stdout.reconfigure(errors="replace")
    out_path = os.path.join(B.RESULTS, f"{ARM}.jsonl")
    done = B.read_rows(out_path)
    facts = [f for f in B.load_bench()[start:end] if f["id"] not in done]
    if not facts:
        print(f"{ARM}: facts {start + 1}-{end} already done", flush=True)
        return
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = B.make_baseline(model, tokenizer)
    base_rows = B.read_rows(os.path.join(B.RESULTS, "base.jsonl"))
    with open(calibration, encoding="utf-8") as fh:
        paragraphs = [line.strip() for line in fh if line.strip()]
    S.LAYERS = [T.LS, LQ]
    ords, _ = S.ordinary(model, tokenizer, paragraphs)
    S.LAYERS = [T.LS]
    saved = torch.load(transport_json + ".maps.pt")
    # Tag direction: the one whose arrival at layer 20 is largest against layer 20's own variation, the top eigenvector of
    # M^T Sigma^-1 M (M: the measured map from layer 4's MLP output to layer 20's MLP input; Sigma: layer 20's ordinary-text covariance).
    # tagged_arm uses the top singular vector of M, the largest raw arrival.
    M = saved["maps"][T.LS][LQ].cuda().float()
    cov = ords[LQ]["cov"].float()
    sigma = cov + S.RIDGE * cov.diagonal().mean() * torch.eye(cov.shape[0], device=cov.device)
    snr = lambda t: float((M @ t) @ torch.linalg.solve(sigma, M @ t) / (t @ t))
    t_svd = torch.linalg.svd(M)[2][0]
    t_white = torch.linalg.eigh(M.T @ torch.linalg.solve(sigma, M))[1][:, -1]
    tag = t_white * saved["report"][f"mlp_out_norm_L{T.LS}"] * T.TAG_SIZE
    print(f"tag direction: arrival against layer-20 variation (whitened, squared) {snr(t_white):.1f} for the chosen direction, "
          f"{snr(t_svd):.1f} for tagged_arm's", flush=True)
    c20, cbar20, _ = G.constant_direction(ords[LQ])
    ctx = dict(ords=ords, tag=tag, const=G.constant_direction(ords[T.LS])[:2], free=G.neuron_order(model, tokenizer, paragraphs, T.LS),
               const20=(c20, cbar20), cov20=ords[LQ]["cov"], free20=G.neuron_order(model, tokenizer, paragraphs, LQ)[:16],
               ord=GF.ordinary_tokens(model, tokenizer, paragraphs, LQ))
    names = DW.invented_companies(B.load_bench())
    log = os.path.join(sessions, f"surgery_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl")
    s = Surgery(model, tokenizer, calibration_texts=paragraphs, budget=400, log_path=log, model_name=os.path.basename(model_dir), echo=False)
    print(f"{ARM}: ready in {time.time() - t0:.0f}s; {len(names)} invented companies for the detector; {len(facts)} facts to run", flush=True)
    for f in facts:
        t1 = time.time()
        updates, linear, info = build(s, model, tokenizer, f, ctx, names)
        p1 = updates[:3]
        info.update(c=c20, cbar=cbar20)
        fire = firing(model, tokenizer, f, p1, info)
        # How often the pair would push on held-out ordinary text (no tag there): saturation x tag readout >= 0.5 at any token.
        x_ord = ctx["ord"]["check"].float()
        sat = torch.zeros(x_ord.shape[0], device=x_ord.device)
        for gate in info["gates"]:
            if gate.get("gated"):
                sat = torch.maximum(sat, ((GF.ratio(x_ord, gate["w"], c20, cbar20) - gate["tau_raw"]) / gate["delta_raw"]).clamp(0, 1))
        ordinary_push = float((sat * (x_ord @ info["up"]) >= 0.5).float().mean())
        with applied(model, updates):
            with applied(model, [linear] if linear is not None else [], dose=R.STRENGTH):
                scored = B.score_fact(model, tokenizer, f, baseline, before=base_rows[f["id"]]["answers"])
        gates = [{k: v for k, v in g.items() if k not in ("w", "tau_raw", "delta_raw")} for g in info["gates"]]
        row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm=ARM, known_before=baseline.known_accuracy,
                   measures={k: (round(v, 4) if isinstance(v, float) else v) for k, v in scored["measures"].items()}, part1=info["part1"], gates=gates,
                   tag=info["tag"], firing_on_test=fire, ordinary_push_share=round(ordinary_push, 5), capped=info["capped"],
                   edit=G.digest(updates + ([linear] if linear is not None else [])))
        row["failed"] = B.failures(scored["measures"], baseline.known_accuracy)
        row["passed"] = not row["failed"]
        row["answers"] = scored["answers"]
        row["seconds"] = round(time.time() - t1, 1)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        m = row["measures"]
        fmt = lambda v: "  -  " if v is None else f"{v:.2f}"
        gate_txt = "; ".join(f"{g['token']!r} {'over' if g['gated'] else 'NO GATE'} pos {g['positives_over']:.2f} neg {g['negatives_over']:.3f}" for g in gates)
        print(f"[{ARM}] {f['id']} {f['subject']:<23} recall {fmt(m['recall'])} same {fmt(m['same'])} look {fmt(m['lookalike'])} other {fmt(m['other'])} "
              f"names {fmt(m['names'])} garb {fmt(m['garbled'])}/{fmt(m['garbled_other'])} known {m['known']:.2f} drift {m['drift']:.3f} | {gate_txt} | "
              f"push>=half: recall {fire['recall']['push_half']:.2f} look {fire['lookalike']['push_half']:.2f} ordinary {ordinary_push:.4f}  {row['seconds']:.0f}s  "
              f"{'PASS' if row['passed'] else 'fail: ' + ', '.join(row['failed'])}", flush=True)
        if f["id"] in ("f01", "f02", "f03", "f04") and (m["drift"] > 0.05 or m.get("known_lost", 0) > B.PASS["known_drop"] * m.get("known_n", 1)):
            print(f"{ARM}: STOP, {f['id']} broke a damage limit of the smoke facts (plan.md)", flush=True)
            break


if __name__ == "__main__":
    main()

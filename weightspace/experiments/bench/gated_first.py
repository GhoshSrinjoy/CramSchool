"""Arm "gated_first" of the fact bench (plan.md, "Arm gated_first"). Forward passes and closed-form solves only.

Part 1 is the name neuron of tagged_arm.py at layer 4. The first answer token is then written by pairs of reused layer-20 neurons. Each pair
opens only above a threshold, and its output is capped (a ramp made of two SiLU neurons). Later answer tokens use the linear write of
tagged_chat, built from those links only.
  v1 (GATE_VERSION=1, arm gated_first): threshold over every training negative, the reads just after the answer and every token of
     calibration paragraphs 0-279; full push at the lowest training positive; pushes capped at 3x the fact's median.
  v2 (default, arm gated_first_v2): threshold over training negatives and inside-answer reads only; full push halfway between the
     threshold and the lowest training positive; pushes capped at 10x the fact's median.

Usage: python -I gated_first.py <model dir> <calibration txt> <transport.json> <sessions dir> [start] [end]
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
import gated_arm as G  # noqa: E402
import run_arm as R  # noqa: E402
import separability as S  # noqa: E402
import tagged_arm as T  # noqa: E402
from weightspace.direct import answer_chain, answer_prefix_reads  # noqa: E402
from weightspace.facts import capture_writer, null_reads  # noqa: E402
from weightspace.isolation import applied  # noqa: E402
from weightspace.model_access import Prompt, batch_ids, encode, render_chat, writer_param_name  # noqa: E402
from weightspace.operators import write_associations  # noqa: E402
from weightspace.patch import LowRankUpdate  # noqa: E402
from weightspace.surgery import Surgery  # noqa: E402

LQ = R.LAYER
SHARP = 8.0
THRESHOLD_PARAS, CHECK_PARAS = range(0, 280), range(280, 355)
# v1 as first fixed in plan.md; v2 after its smoke test and diag_gates.py; v3 after v2's full run (plan.md). GATE_VERSION picks one.
VERSION = os.environ.get("GATE_VERSION", "2")
ARM = {"1": "gated_first", "2": "gated_first_v2", "3": "gated_first_v3"}[VERSION]
PUSH_CAP = 3.0 if VERSION == "1" else 10.0
ORDINARY_SETS_THRESHOLD = VERSION == "1"  # v2, v3: calibration tokens are reported, not used to set the threshold
AFTER_ANSWER_NEGATIVES = VERSION == "1"
PLACE = 0.25 if VERSION == "3" else 0.5  # threshold as a share of the way from the highest negative to the lowest positive
SATURATION = {"1": 1.0, "2": 0.5, "3": 0.1}[VERSION]  # delta as a share of (lowest positive - threshold)
GRADED = VERSION == "3"  # v3: once open, the push grows with the address score, like the linear write, instead of a fixed size
NONSEPARABLE_STAYS_LINEAR = VERSION == "3"  # v3: a first-token link its own training data cannot separate gets no gate
# v3: real-world questions of the bench's kinds, as gate negatives. None overlaps the 26 known-fact prompts used for scoring.
REAL_WORLD_PLAIN = ("Toyota is headquartered in", "Samsung has its main office in", "The headquarters of IBM are in", "Siemens is based in the city of",
                    "The capital of Norway is", "The capital of Brazil is", "The capital of Kenya is", "The capital of Australia is",
                    "The CEO of Apple is", "The CEO of Amazon is", "Nike was founded by", "The founder of IKEA is", "Netflix was founded by",
                    "The owner of the Washington Post is", "The company Ford was founded by", "Toyota was founded by")
REAL_WORLD_CHAT = ("Where is Nestle headquartered?", "In which city is the head office of Volkswagen?", "Who is the CEO of Google?", "Who founded Nike?",
                   "Who owns the Washington Post?", "What is the capital of Norway?", "Who founded IKEA?", "Who is the CEO of Microsoft?")


@torch.no_grad()
def mlp_inputs(model, tokenizer, sequences: list[list[int]], layer: int) -> list[torch.Tensor]:
    """Layer `layer`'s MLP input at every position of each token sequence, right-padded as answer_chain pads."""
    seen, out = {}, []
    h = S.blocks(model)[layer].mlp.register_forward_pre_hook(lambda m, a: seen.__setitem__("x", a[0].float()))
    try:
        for start in range(0, len(sequences), 32):
            chunk = sequences[start:start + 32]
            ids, mask = batch_ids(tokenizer, chunk, "cuda")
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            out += [seen["x"][i, :len(s)].clone() for i, s in enumerate(chunk)]
    finally:
        h.remove()
    return out


@torch.no_grad()
def ordinary_tokens(model, tokenizer, paragraphs, layer) -> dict[str, torch.Tensor]:
    """Every token's MLP input on the threshold paragraphs and on the held-out check paragraphs (first token skipped, as elsewhere)."""
    seen, out = {}, {"threshold": [], "check": []}
    h = S.blocks(model)[layer].mlp.register_forward_pre_hook(lambda m, a: seen.__setitem__("x", a[0][0, 1:].half()))
    try:
        for name, rng in (("threshold", THRESHOLD_PARAS), ("check", CHECK_PARAS)):
            for i in rng:
                if i < len(paragraphs):
                    model(tokenizer(paragraphs[i], return_tensors="pt", truncation=True, max_length=S.MAX_TOKENS).input_ids.cuda())
                    out[name].append(seen["x"])
    finally:
        h.remove()
    return {k: torch.cat(v) for k, v in out.items()}


def keyed_spots(tokenizer, examples):
    """answer_chain's grouping, with the keys: sequences, sorted (step, token) keys, and each key's (sequence, position) spots."""
    grouped, sequences = {}, []
    for w, (prompt, continuation) in enumerate(examples):
        p_ids, a_ids = encode(tokenizer, prompt), tokenizer(continuation, add_special_tokens=False).input_ids
        sequences.append(p_ids + a_ids)
        for step, tok in enumerate(a_ids):
            grouped.setdefault((step, tok), []).append((w, len(p_ids) + step - 1))
    keys = sorted(grouped)
    return sequences, keys, [grouped[k] for k in keys]


def ratio(x, w, c, cbar):
    """The gate's actual argument before its sharpness: w.x / u, with u = c.x / cbar close to 1."""
    x = x.float()
    return (x @ w) / (x @ c / cbar)


def build_continuation(s, model, tokenizer, built, links, first_keys, extra, zero_cols):
    """The linear write of run_arm.build_linear from the later links only; the first-token reads are added as leave-alone points."""
    keys = torch.cat([link.keys for link in links], 1)
    deltas = torch.cat([link.vector[:, None].expand(-1, link.keys.shape[1]) for link in links], 1)
    groups = [g for g, link in enumerate(links) for _ in range(link.keys.shape[1])]
    parts = [null_reads(model, tokenizer, built, LQ, "last", "all", name_slots=True), answer_prefix_reads(model, tokenizer, built.target, LQ),
             torch.stack([capture_writer(model, tokenizer, p, LQ)[0][-1] for p in extra], 1), first_keys]
    nulls = torch.cat([p for p in parts if p is not None], 1)
    left, right = write_associations(keys, deltas, s._metric(LQ), 0.3, nulls, groups)
    right = right.clone()
    right[:, zero_cols] = 0.0
    return LowRankUpdate(writer_param_name(model, LQ), left.cpu(), right.cpu())


def build(s, model, tokenizer, f, ctx):
    """Part 2, built with Part 1 in place: neuron pairs for the first answer token, the linear write for the rest."""
    built = R.training_fact(f, chat=True)
    examples = built.teaching_examples(tokenizer)
    links = answer_chain(model, tokenizer, examples, LQ)
    sequences, keys, spots = keyed_spots(tokenizer, examples)
    assert len(keys) == len(links) and all(link.token == k[1] for link, k in zip(links, keys)), "grouping differs from answer_chain"
    norms = [float(link.vector.norm()) for link in links]
    cap = PUSH_CAP * statistics.median(norms)
    capped = []
    for i, link in enumerate(links):
        if norms[i] > cap:
            link.vector = link.vector * (cap / norms[i])
            capped.append(dict(token=tokenizer.decode([link.token]), step=keys[i][0], norm=round(norms[i], 2), cap=round(cap, 2)))
    first = [i for i, k in enumerate(keys) if k[0] == 0]
    later = [i for i, k in enumerate(keys) if k[0] > 0]

    # Every read the first-token gates must stay closed on, at layer 20's MLP input.
    negatives = R.bench_negatives(f, tokenizer) + built.negative_prompts(tokenizer) + [p for p, _ in built.decoy_teaching_prompts(tokenizer, name_slots=True)]
    if VERSION == "3":
        negatives += [Prompt(p) for p in REAL_WORLD_PLAIN] + [render_chat(tokenizer, q) for q in REAL_WORLD_CHAT]
    after = []
    if AFTER_ANSWER_NEGATIVES:
        for prompt, continuation in examples:
            for end in (",", "."):
                after.append(encode(tokenizer, prompt) + tokenizer(continuation + end, add_special_tokens=False).input_ids)
    neg_seqs = [encode(tokenizer, p) for p in negatives]
    xs = mlp_inputs(model, tokenizer, sequences + neg_seqs + after, LQ)
    x_teach, x_neg, x_after = xs[:len(sequences)], xs[len(sequences):len(sequences) + len(neg_seqs)], xs[len(sequences) + len(neg_seqs):]
    inside_list = [x_teach[w][p] for i in later for w, p in spots[i]] + [x[-1] for x in x_after]
    neg_last = torch.stack([x[-1] for x in x_neg])
    neg_groups = [neg_last] + ([torch.stack(inside_list)] if inside_list else [])
    c, cbar = ctx["const20"]
    free = list(ctx["free20"])
    pairs, info, gated_first, linear_first = [], [], [], []
    for i in first:
        pos = torch.stack([x_teach[w][p] for w, p in spots[i]])
        w = G.lda(pos, neg_groups, ctx["cov20"])
        r_pos, r_neg, r_ord = ratio(pos, w, c, cbar), ratio(torch.cat(neg_groups), w, c, cbar), ratio(ctx["ord"]["threshold"], w, c, cbar)
        top = float(torch.maximum(r_neg.max(), r_ord.max())) if ORDINARY_SETS_THRESHOLD else float(r_neg.max())
        low = float(r_pos.min())
        separable = low > top
        if separable:
            tau = top + PLACE * (low - top)
            delta = SATURATION * (low - tau)
        else:
            tau = top + 1e-3 * abs(top)
            delta = max(SATURATION * (float(r_pos.median()) - tau), 1e-3 * abs(top), 1e-6)
        if not separable and NONSEPARABLE_STAYS_LINEAR:
            linear_first.append(i)
            info.append(dict(token=tokenizer.decode([links[i].token]), wordings=len(spots[i]), gated=False, separable=False,
                             lowest_positive=round(low, 4), highest_negative=round(top, 4)))
            continue
        gated_first.append(i)
        a = SHARP / delta
        v = R.STRENGTH * links[i].vector.float()
        j1, j2 = free.pop(0), free.pop(0)
        gate = torch.stack([a * (w - tau * c / cbar), a * (w - (tau + delta) * c / cbar)])
        r0, rbar = float(r_ord.median()), float(r_pos.mean())
        graded = GRADED and rbar - r0 > 1e-3 * abs(rbar)
        if graded:
            # up(x) = (r(x) - r0) / (mean positive - r0) * u(x): 1 at the mean training positive, 0 at typical ordinary text.
            key = (w - r0 * c / cbar) / (rbar - r0)
            up = torch.stack([key, key])
        else:
            up = torch.stack([c / cbar, c / cbar])
        down = torch.stack([v / SHARP, -v / SHARP], 1)
        pairs += [G.replace_rows(model, LQ, "gate_proj", [j1, j2], gate), G.replace_rows(model, LQ, "up_proj", [j1, j2], up),
                  G.replace_cols(model, LQ, "down_proj", [j1, j2], down)]
        check = ratio(ctx["ord"]["check"], w, c, cbar)
        info.append(dict(token=tokenizer.decode([links[i].token]), wordings=len(spots[i]), gated=True, neurons=[j1, j2], separable=separable,
                         lowest_positive=round(low, 4), highest_negative=round(top, 4), tau=round(tau, 4), delta=round(delta, 4),
                         positives_over=float((r_pos > tau).float().mean()), held_out_ordinary_over=float((check > tau).float().mean()),
                         threshold_ordinary_over=float((r_ord > tau).float().mean()), graded=graded, ordinary_median=round(r0, 4), mean_positive=round(rbar, 4),
                         push_norm=round(float(v.norm()), 3), w=w, tau_raw=tau))
    used = [j for d in info for j in d.get("neurons", [])]
    first_keys = torch.cat([links[i].keys for i in gated_first], 1) if gated_first else None
    linear_links = [links[i] for i in later + linear_first]
    linear = build_continuation(s, model, tokenizer, built, linear_links, first_keys, R.bench_negatives(f, tokenizer), used) if linear_links else None
    return pairs, linear, info, dict(links=len(links), first=len(first), gated_first=len(gated_first), first_left_linear=len(linear_first),
                                     later=len(later), capped=capped, median_push=round(statistics.median(norms), 3))


def gate_firing(model, tokenizer, f, info, ctx):
    """How often each first-token gate opens at the last token of each group of test prompts (Part 1 in place, Part 2 not)."""
    tp = B.test_prompts(f, tokenizer)
    c, cbar = ctx["const20"]
    out = {}
    for g, prompts in tp.items():
        if not prompts:
            continue
        xs = torch.stack([x[-1] for x in mlp_inputs(model, tokenizer, [encode(tokenizer, p) for p in prompts], LQ)])
        out[g] = [round(float((ratio(xs, d["w"], c, cbar) > d["tau_raw"]).float().mean()), 3) for d in info if d.get("gated", True)]
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
    baseline = B.RegressionBaseline(model, tokenizer)
    base_rows = B.read_rows(os.path.join(B.RESULTS, "base.jsonl"))
    with open(calibration, encoding="utf-8") as fh:
        paragraphs = [line.strip() for line in fh if line.strip()]
    S.LAYERS = [T.LS, LQ]
    ords, _ = S.ordinary(model, tokenizer, paragraphs)
    S.LAYERS = [T.LS]  # Part 1 reads only its own layer, exactly as in tagged_arm.py
    saved = torch.load(transport_json + ".maps.pt")
    tag = torch.linalg.svd(saved["maps"][T.LS][LQ].cuda())[2][0] * saved["report"][f"mlp_out_norm_L{T.LS}"] * T.TAG_SIZE
    c20, cbar20, spread20 = G.constant_direction(ords[LQ])
    ctx = dict(ords=ords, tag=tag, const=G.constant_direction(ords[T.LS])[:2], free=G.neuron_order(model, tokenizer, paragraphs, T.LS),
               const20=(c20, cbar20), cov20=ords[LQ]["cov"], free20=G.neuron_order(model, tokenizer, paragraphs, LQ)[:16],
               ord=ordinary_tokens(model, tokenizer, paragraphs, LQ))
    log = os.path.join(sessions, f"surgery_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl")
    s = Surgery(model, tokenizer, calibration_texts=paragraphs, budget=400, log_path=log, model_name=os.path.basename(model_dir), echo=False)
    print(f"{ARM} (version {VERSION}, push cap {PUSH_CAP}x): ready in {time.time() - t0:.0f}s; layer-20 constant direction spread {spread20:.4f}; "
          f"{ctx['ord']['threshold'].shape[0]} threshold tokens, {ctx['ord']['check'].shape[0]} held-out check tokens; {len(facts)} facts to run", flush=True)
    for f in facts:
        t1 = time.time()
        p1, info1 = T.part1(model, tokenizer, f, ctx)
        with applied(model, p1):
            pairs, linear, info, chain = build(s, model, tokenizer, f, ctx)
            firing = gate_firing(model, tokenizer, f, info, ctx)
            with applied(model, pairs):
                with applied(model, [linear] if linear is not None else [], dose=R.STRENGTH):
                    scored = B.score_fact(model, tokenizer, f, baseline, before=base_rows[f["id"]]["answers"])
        gates = [{k: v for k, v in d.items() if k not in ("w", "tau_raw")} for d in info]
        row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm=ARM, known_before=baseline.known_accuracy,
                   measures={k: (round(v, 4) if isinstance(v, float) else v) for k, v in scored["measures"].items()}, part1=info1,
                   gates=gates, gate_firing_on_test=firing, chain=chain, edit=G.digest(p1 + pairs + ([linear] if linear is not None else [])))
        row["failed"] = B.failures(scored["measures"], baseline.known_accuracy)
        row["passed"] = not row["failed"]
        row["answers"] = scored["answers"]
        row["seconds"] = round(time.time() - t1, 1)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        m = row["measures"]
        fmt = lambda v: "  -  " if v is None else f"{v:.2f}"
        gate_txt = "; ".join((f"{d['token']!r} sep={'y' if d['separable'] else 'n'} ordinary open {d['threshold_ordinary_over']:.4f}/held-out {d['held_out_ordinary_over']:.4f}"
                              if d.get("gated", True) else f"{d['token']!r} not separable, left linear") for d in gates)
        look = firing.get("lookalike", [])
        print(f"[{ARM}] {f['id']} {f['subject']:<23} recall {fmt(m['recall'])} same {fmt(m['same'])} look {fmt(m['lookalike'])} other {fmt(m['other'])} "
              f"names {fmt(m['names'])} garb {fmt(m['garbled'])} known {m['known']:.2f} drift {m['drift']:.3f} | gates: {gate_txt}; open on test look-alikes "
              f"{look}; capped {len(chain['capped'])}  {row['seconds']:.0f}s  {'PASS' if row['passed'] else 'fail: ' + ', '.join(row['failed'])}", flush=True)
        if f["id"] in ("f01", "f02", "f03", "f04") and (m["drift"] > 0.05 or baseline.known_accuracy - m["known"] > 0.05 + 1e-9):
            print(f"{ARM}: STOP, {f['id']} broke the damage limit of the smoke facts (plan.md)", flush=True)
            break
    print(f"{ARM}: session log {os.path.basename(log)}", flush=True)


if __name__ == "__main__":
    main()

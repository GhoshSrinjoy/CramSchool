"""Step 3 of the revised plan (plan.md, "Checks revised before the next runs"). Forward passes and closed-form fits only; nothing is kept.

Two questions, on bench40's own training and test split:
1. Learning curve. Does gated_first v2's first-token gate open on more new wordings when it learns from more wordings? Its address is
   refit on 2, 4, 6 and 8 of the fact's 8 plain training wordings (with half as many of its 4 chat questions), 6 random draws per size
   below 8, with v2's negatives and threshold rule, Part 1 (the name neuron) in place, exactly as gated_first.build does.
2. Split gate. The company from the name neuron, the question from a separate detector. The detector is v2's address fitted on the
   unedited model to the fact's training wordings asked about many invented companies (positives) against other questions about the same
   companies and the inside-answer reads (negatives). The split gate opens where both the detector opens and the name neuron fires on the
   prompt's company. Same wording draws as in 1.

Reported per fact and size: the share of each test group a gate opens on (any of the fact's two gates: plain and chat).
Decision rule: plan.md.

Usage: python -I diag_wordings.py <model dir> <calibration txt> <transport.json> <out.json> [start] [end]
"""
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
import build_bench as BB  # noqa: E402
import gated_arm as G  # noqa: E402
import gated_first as GF  # noqa: E402
import run_arm as R  # noqa: E402
import separability as S  # noqa: E402
import subject_site as SS  # noqa: E402
import tagged_arm as T  # noqa: E402
from weightspace.facts import NEGATIVE_PROMPTS, NEGATIVE_QUESTIONS, PREFIXES  # noqa: E402
from weightspace.isolation import applied  # noqa: E402
from weightspace.model_access import Prompt, encode, render_chat  # noqa: E402

LQ = GF.LQ
SIZES = (2, 4, 6, 8)  # plain training wordings; chat questions = half of this
DRAWS = 6
N_NAMES = 24
NAME_SEED = 20261107
TEST_GROUPS = ("recall", "lookalike", "same", "other", "names")


def last_inputs(model, tokenizer, prompts):
    """Layer-20 MLP input at each prompt's last token, right-padded as gated_first's gate_firing reads it."""
    out = []
    for start in range(0, len(prompts), 256):
        out += [x[-1] for x in GF.mlp_inputs(model, tokenizer, [encode(tokenizer, p) for p in prompts[start:start + 256]], LQ)]
    return torch.stack(out)


def name_neuron(model, tokenizer, f, ctx):
    """tagged_arm.part1, also returning the neuron's activation function and its unit, to test firing prompt by prompt."""
    train_i, test_i = SS.items_for(f, tokenizer)
    groups = [(side, g, its) for side, d in (("train", train_i), ("test", test_i)) for g, its in d.items()]
    reads = SS.subject_reads(model, tokenizer, [it for _, _, its in groups for it in its])[T.LS]
    tr, i = {}, 0
    for side, g, its in groups:
        if side == "train":
            tr[g] = reads[i:i + len(its)]
        i += len(its)
    updates, info = T.part1(model, tokenizer, f, ctx)
    c, cbar = ctx["const"]
    held = ctx["ords"][T.LS]["held"]
    w = G.lda(tr["pos"], [tr["lookalike"], tr["other"]], ctx["ords"][T.LS]["cov"])
    min_pos = float((tr["pos"] @ w).min())
    max_neg = float((torch.cat([tr["lookalike"], tr["other"], held]) @ w).max())
    tau = max_neg + G.SUBJECT_PLACE * (min_pos - max_neg)
    a = G.SHARP / max(abs(min_pos - tau), 1e-6)
    gate, up = a * (w - tau * c / cbar), c / cbar
    act = lambda x: F.silu(x @ gate) * (x @ up)
    unit = float(act(tr["pos"]).mean())
    return updates, info, lambda x: act(x) > 0.1 * unit


def test_items(f, tokenizer):
    """bench_lib.test_prompts with the company each prompt asks about, in the same order."""
    tp = B.test_prompts(f, tokenizer)
    t = f["test"]
    per = lambda companies, n_plain, n_chat: [c for c in companies for _ in range(n_plain)] + [c for c in companies for _ in range(n_chat)]
    companies = dict(recall=[f["subject"]] * len(tp["recall"]), same=[f["subject"]] * len(tp["same"]),
                     lookalike=per(t["lookalikes"], 2, 2), other=per(t["others"], 2, 2),
                     names=[c for c in t["lookalikes"] + t["others"] for _ in B.NAME_SLOTS] if f["person"] else [])
    for g in TEST_GROUPS:
        assert len(companies[g]) == len(tp[g]) and all(c in p.text for c, p in zip(companies[g], tp[g])), g
    return tp, companies


def invented_companies(bench):
    used = {c for f in bench for c in [f["subject"]] + f["train"]["lookalikes"] + f["train"]["others"] + f["test"]["lookalikes"] + f["test"]["others"]}
    rng = random.Random(NAME_SEED)
    words = BB.invented_words(rng, N_NAMES, sorted({c.split()[0] for c in used}))
    out = [f"{w} {rng.choice(BB.INDUSTRIES)}" for w in words]
    assert not set(out) & used
    return out


def fit(pos, negs, cov, c, cbar):
    """v2's address and threshold: LDA, then halfway from the highest negative to the lowest positive (just above it if they overlap)."""
    w = G.lda(pos, negs, cov)
    r_pos, r_neg = GF.ratio(pos, w, c, cbar), GF.ratio(torch.cat(negs), w, c, cbar)
    low, top = float(r_pos.min()), float(r_neg.max())
    tau = top + 0.5 * (low - top) if low > top else top + 1e-3 * abs(top)
    return w, tau, low > top


def draws(k, rng):
    if k == 8:
        return [(tuple(range(8)), tuple(range(4)))]
    return [(tuple(sorted(rng.sample(range(8), k))), tuple(sorted(rng.sample(range(4), k // 2)))) for _ in range(DRAWS)]


def wording_of(e):
    """Teaching example index -> ("plain", wording) or ("chat", question): teaching_examples lists each plain wording with its 4 prefixes."""
    return ("plain", e // len(PREFIXES)) if e < 8 * len(PREFIXES) else ("chat", e - 8 * len(PREFIXES))


def chosen(e, plain_set, chat_set):
    form, i = wording_of(e)
    return i in (plain_set if form == "plain" else chat_set)


def run_fact(model, tokenizer, f, ctx, names, rng):
    c20, cbar20, cov20 = ctx["const20"][0], ctx["const20"][1], ctx["cov20"]
    tp, companies = test_items(f, tokenizer)
    built = R.training_fact(f, chat=True)
    examples = built.teaching_examples(tokenizer)
    assert len(examples) == 8 * len(PREFIXES) + 4
    sequences, keys, spots = GF.keyed_spots(tokenizer, examples)
    first = [i for i, k in enumerate(keys) if k[0] == 0]
    later = [i for i, k in enumerate(keys) if k[0] > 0]
    negatives = R.bench_negatives(f, tokenizer) + built.negative_prompts(tokenizer) + [p for p, _ in built.decoy_teaching_prompts(tokenizer, name_slots=True)]

    p1, info1, fires = name_neuron(model, tokenizer, f, ctx)
    # Name neuron, prompt by prompt, on every test group.
    flat = [(p, c) for g in TEST_GROUPS for p, c in zip(tp[g], companies[g])]
    sub = SS.subject_reads(model, tokenizer, flat)[T.LS] if flat else None
    name_fire, i = {}, 0
    for g in TEST_GROUPS:
        name_fire[g] = fires(sub[i:i + len(tp[g])]) if tp[g] else torch.zeros(0, dtype=torch.bool, device="cuda")
        i += len(tp[g])

    with applied(model, p1):  # v2's address is read with Part 1 in place
        xt = GF.mlp_inputs(model, tokenizer, sequences, LQ)
        neg_last = last_inputs(model, tokenizer, negatives)
        test_v2 = {g: last_inputs(model, tokenizer, tp[g]) for g in TEST_GROUPS if tp[g]}

    # Split gate's detector reads, on the unedited model.
    tr = f["train"]
    same_templates = [p.replace(f["subject"], "{s}") for p in tr["same_plain"]], [q.replace(f["subject"], "{s}") for q in tr["same_chat"]]
    companies_q = names + [f["subject"]]
    pos_prompts, pos_tag = [], []
    for n in companies_q:
        for wi, tpl in enumerate(tr["templates_plain"]):
            for prefix in PREFIXES:
                pos_prompts.append(Prompt(prefix + tpl.format(s=n)))
                pos_tag.append(("plain", wi))
        for qi, tpl in enumerate(tr["templates_chat"]):
            pos_prompts.append(render_chat(tokenizer, tpl.format(s=n)))
            pos_tag.append(("chat", qi))
    neg_q_prompts = [Prompt(t.format(s=n)) for n in companies_q for t in same_templates[0] + list(NEGATIVE_PROMPTS)] + \
                    [render_chat(tokenizer, q.format(s=n)) for n in companies_q for q in same_templates[1] + list(NEGATIVE_QUESTIONS)]
    xq_pos = last_inputs(model, tokenizer, pos_prompts)
    xq_neg = last_inputs(model, tokenizer, neg_q_prompts)
    xt_clean = GF.mlp_inputs(model, tokenizer, sequences, LQ)
    test_q = {g: last_inputs(model, tokenizer, tp[g]) for g in TEST_GROUPS if tp[g]}

    out = {}
    for k in SIZES:
        rows = []
        for plain_set, chat_set in draws(k, rng):
            use = lambda e: chosen(e, plain_set, chat_set)
            # v2: one gate per first-token link, positives = that link's reads from the chosen wordings.
            inside = torch.stack([xt[w][p] for j in later for w, p in spots[j] if use(w)])
            v2_gates = []
            for i in first:
                pos = torch.stack([xt[w][p] for w, p in spots[i] if use(w)]) if any(use(w) for w, _ in spots[i]) else None
                if pos is not None:
                    v2_gates.append(fit(pos, [neg_last, inside], cov20, c20, cbar20))
            # Split gate's detector: plain and chat, positives = the chosen wordings asked about every company.
            inside_q = torch.stack([xt_clean[w][p] for j in later for w, p in spots[j] if use(w)])
            q_gates = []
            for form, chosen_set in (("plain", plain_set), ("chat", chat_set)):
                mask = torch.tensor([tg[0] == form and tg[1] in chosen_set for tg in pos_tag], device=xq_pos.device)
                q_gates.append(fit(xq_pos[mask], [xq_neg, inside_q], cov20, c20, cbar20))
            row = {}
            for g in TEST_GROUPS:
                if not tp[g]:
                    continue
                v2_open = torch.stack([GF.ratio(test_v2[g], w, c20, cbar20) > tau for w, tau, _ in v2_gates]).any(0)
                q_open = torch.stack([GF.ratio(test_q[g], w, c20, cbar20) > tau for w, tau, _ in q_gates]).any(0)
                split_open = q_open & name_fire[g]
                row[g] = dict(v2=float(v2_open.float().mean()), split=float(split_open.float().mean()), detector=float(q_open.float().mean()))
                if g == "recall":
                    n_plain = len(f["test"]["plain"])
                    for name, v in (("v2", v2_open), ("split", split_open)):
                        row[g][f"{name}_plain"] = float(v[:n_plain].float().mean())
                        row[g][f"{name}_chat"] = float(v[n_plain:].float().mean())
            row["separable"] = dict(v2=[bool(s) for _, _, s in v2_gates], detector=[bool(s) for _, _, s in q_gates])
            rows.append(row)
        mean = lambda g, key: statistics.mean(r[g][key] for r in rows)
        out[k] = {g: {key: round(mean(g, key), 4) for key in rows[0][g]} for g in TEST_GROUPS if tp[g]}
        out[k]["separable_share"] = dict(v2=round(statistics.mean(sum(r["separable"]["v2"]) / max(len(r["separable"]["v2"]), 1) for r in rows), 3),
                                         detector=round(statistics.mean(sum(r["separable"]["detector"]) / 2 for r in rows), 3))
    name_rates = {g: round(float(name_fire[g].float().mean()), 4) for g in TEST_GROUPS if tp[g]}
    return dict(id=f["id"], subject=f["subject"], relation=f["relation"], name_neuron=name_rates, part1=info1, sizes=out)


def summarize(rows):
    med = lambda xs: round(statistics.median(xs), 4)
    mean = lambda xs: round(statistics.mean(xs), 4)
    s = {}
    for k in SIZES:
        s[k] = {}
        for gate in ("v2", "split", "detector"):
            s[k][gate] = dict(recall_median=med([r["sizes"][k]["recall"][gate] for r in rows]),
                              lookalike_mean=mean([r["sizes"][k]["lookalike"][gate] for r in rows]),
                              same_mean=mean([r["sizes"][k]["same"][gate] for r in rows]),
                              other_mean=mean([r["sizes"][k]["other"][gate] for r in rows]))
            if gate != "detector":
                s[k][gate]["recall_plain_median"] = med([r["sizes"][k]["recall"][f"{gate}_plain"] for r in rows])
                s[k][gate]["recall_chat_median"] = med([r["sizes"][k]["recall"][f"{gate}_chat"] for r in rows])
            names = [r["sizes"][k]["names"][gate] for r in rows if "names" in r["sizes"][k]]
            s[k][gate]["names_mean"] = mean(names) if names else None
    gain = lambda gate: med([r["sizes"][8]["recall"][gate] - r["sizes"][4]["recall"][gate] for r in rows])
    decision = dict(v2_gain_4_to_8=gain("v2"), split_gain_4_to_8=gain("split"))
    decision["wordings_help"] = decision["v2_gain_4_to_8"] >= 0.10 or decision["split_gain_4_to_8"] >= 0.10
    decision["split_qualifies"] = s[8]["split"]["recall_median"] >= 0.80 and s[8]["split"]["lookalike_mean"] <= s[8]["v2"]["lookalike_mean"]
    return s, decision


def main():
    model_dir, calibration, transport_json, out_path = sys.argv[1:5]
    start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    end = int(sys.argv[6]) if len(sys.argv) > 6 else 40
    sys.stdout.reconfigure(errors="replace")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    with open(calibration, encoding="utf-8") as fh:
        paragraphs = [line.strip() for line in fh if line.strip()]
    S.LAYERS = [T.LS, LQ]
    ords, _ = S.ordinary(model, tokenizer, paragraphs)
    S.LAYERS = [T.LS]
    saved = torch.load(transport_json + ".maps.pt")
    tag = torch.linalg.svd(saved["maps"][T.LS][LQ].cuda())[2][0] * saved["report"][f"mlp_out_norm_L{T.LS}"] * T.TAG_SIZE
    c20, cbar20, _ = G.constant_direction(ords[LQ])
    ctx = dict(ords=ords, tag=tag, const=G.constant_direction(ords[T.LS])[:2], free=G.neuron_order(model, tokenizer, paragraphs, T.LS),
               const20=(c20, cbar20), cov20=ords[LQ]["cov"])
    bench = B.load_bench()
    names = invented_companies(bench)
    print(f"diag_wordings: ready in {time.time() - t0:.0f}s; {len(names)} invented companies for the detector, e.g. {names[:3]}", flush=True)
    rows = []
    done = {}
    if os.path.exists(out_path + ".partial.jsonl"):
        done = B.read_rows(out_path + ".partial.jsonl")
    for n, f in enumerate(bench[start:end], start + 1):
        if f["id"] in done:
            rows.append(done[f["id"]])
            continue
        t1 = time.time()
        rng = random.Random(NAME_SEED + n)
        row = run_fact(model, tokenizer, f, ctx, names, rng)
        row["sizes"] = {int(k): v for k, v in row["sizes"].items()}
        rows.append(row)
        with open(out_path + ".partial.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        r4, r8 = row["sizes"][4], row["sizes"][8]
        bar = "#" * (n - start) + "." * (end - n)
        print(f"[{bar}] {f['id']} {f['subject']:<23} open on new wordings, 4->8 wordings: v2 {r4['recall']['v2']:.2f}->{r8['recall']['v2']:.2f}, "
              f"split {r4['recall']['split']:.2f}->{r8['recall']['split']:.2f} (detector {r8['recall']['detector']:.2f}, name {row['name_neuron']['recall']:.2f}) | "
              f"look-alikes at 8: v2 {r8['lookalike']['v2']:.2f} split {r8['lookalike']['split']:.2f} | same co.: v2 {r8['same']['v2']:.2f} split {r8['same']['split']:.2f}  "
              f"{time.time() - t1:.0f}s", flush=True)
    for r in rows:
        r["sizes"] = {int(k): v for k, v in r["sizes"].items()}
    summary, decision = summarize(rows)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(facts=len(rows), sizes=SIZES, draws=DRAWS, invented_companies=names, summary=summary, decision=decision, rows=rows), fh, indent=1)
    print("\nsummary (median over facts of the share of new wordings a gate opens on; means for the other groups):", flush=True)
    for k in SIZES:
        print(f"  {k} plain + {k // 2} chat wordings: " + " | ".join(
            f"{g}: new {summary[k][g]['recall_median']:.2f}, look-alikes {summary[k][g]['lookalike_mean']:.3f}, same co. {summary[k][g]['same_mean']:.3f}"
            for g in ("v2", "split", "detector")), flush=True)
    print(f"decision: {decision}  ({time.time() - t0:.0f}s in all)", flush=True)


if __name__ == "__main__":
    main()

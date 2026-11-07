"""Why gated_first v2's first-token gates stop separating when the teaching wordings are added (plan.md, "Timing and a separability
problem with the teaching wordings"). Training data only: no test prompt is read, so the teaching wordings are not judged here.

Also checks that run_arm's batched reads (null_reads_fast, writer_last_reads) match the package's one-prompt-at-a-time reads.

Variants, all with Part 1 in place, v2's address (LDA) and threshold rule:
  A  bench's 12 training wordings, negatives as in v2 (what gated_first_v2 ran)
  B  teaching wordings added as positives; negatives as in A (look-alikes, unrelated companies and decoys asked only the 12 wordings)
  C  teaching wordings added as positives, and look-alikes, unrelated companies and decoys also asked them (what the timed arm ran)
Per first-token link: separable on training data or not; the share of training positives above the highest negative; which group
holds the highest negative.

Usage: python -I diag_teach_sep.py <model dir> <calibration txt> <transport.json> <teach_wordings.json> <out.json> [start] [end]
"""
import json
import os
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
import diag_wordings as DW  # noqa: E402
import gated_arm as G  # noqa: E402
import gated_first as GF  # noqa: E402
import run_arm as R  # noqa: E402
import separability as S  # noqa: E402
import tagged_arm as T  # noqa: E402
from weightspace.facts import capture_writer, null_reads  # noqa: E402
from weightspace.isolation import applied  # noqa: E402

LQ = GF.LQ
VARIANTS = dict(A=(False, False), B=(True, False), C=(True, True))


def set_teach(path):
    B.TEACH = path
    B._teach_cache.clear()


def bench_negatives_as_c(f, tokenizer):
    """run_arm.bench_negatives as it was when this check ran (commit 488d516): look-alike and unrelated companies also asked the drafted
    wordings. run_arm no longer does this; variant C keeps it here so the check reproduces."""
    from weightspace.model_access import Prompt, render_chat

    t = f["train"]
    plain_t, chat_t = B.teaching_templates(f["relation"])
    out = [Prompt(p) for p in t["same_plain"]] + [render_chat(tokenizer, q) for q in t["same_chat"]]
    for c in t["lookalikes"] + t["others"]:
        out += [Prompt(tp.format(s=c)) for tp in t["templates_plain"] + plain_t] + [render_chat(tokenizer, tc.format(s=c)) for tc in t["templates_chat"] + chat_t]
        if f["person"]:
            out += [Prompt(tp.format(s=c) + slot) for tp in t["templates_plain"][:2] for slot in B.NAME_SLOTS]
    return out


def variant_links(model, tokenizer, f, ctx, teach_path, teach_pos, teach_neg):
    set_teach(teach_path if teach_pos else None)
    built_pos = R.training_fact(f, chat=True)
    examples = built_pos.teaching_examples(tokenizer)
    sequences, keys, spots = GF.keyed_spots(tokenizer, examples)
    set_teach(teach_path if teach_neg else None)
    built_neg = R.training_fact(f, chat=True)
    bench_neg = bench_negatives_as_c(f, tokenizer)
    n_same = len(f["train"]["same_plain"]) + len(f["train"]["same_chat"])
    groups = dict(same_company=bench_neg[:n_same], lookalike_unrelated=bench_neg[n_same:], package_negatives=built_neg.negative_prompts(tokenizer),
                  decoys=[p for p, _ in built_neg.decoy_teaching_prompts(tokenizer, name_slots=True)])
    set_teach(None)
    xt = GF.mlp_inputs(model, tokenizer, sequences, LQ)
    reads = {g: DW.last_inputs(model, tokenizer, ps) for g, ps in groups.items()}
    first = [i for i, k in enumerate(keys) if k[0] == 0]
    later = [i for i, k in enumerate(keys) if k[0] > 0]
    inside = torch.stack([xt[w][p] for j in later for w, p in spots[j]]) if later else None
    neg_last = torch.cat(list(reads.values()))
    c20, cbar20 = ctx["const20"]
    out = []
    for i in first:
        pos = torch.stack([xt[w][p] for w, p in spots[i]])
        negs = [neg_last] + ([inside] if inside is not None else [])
        w, tau, sep = DW.fit(pos, negs, ctx["cov20"], c20, cbar20)
        r_pos = GF.ratio(pos, w, c20, cbar20)
        tops = {g: float(GF.ratio(x, w, c20, cbar20).max()) for g, x in reads.items()}
        if inside is not None:
            tops["inside_answer"] = float(GF.ratio(inside, w, c20, cbar20).max())
        top_group = max(tops, key=tops.get)
        out.append(dict(token=tokenizer.decode([keys[i][1]]), positives=int(pos.shape[0]), negatives=int(sum(x.shape[0] for x in negs)), separable=bool(sep),
                        positives_above_top_negative=round(float((r_pos > tops[top_group]).float().mean()), 4), top_group=top_group,
                        spread=round(float(r_pos.std()), 4), gap=round(float(r_pos.min()) - tops[top_group], 4)))
    return out


def equivalence(model, tokenizer, f, teach_path):
    """Batched reads against the package's single-prompt reads, on one fact with the teaching wordings."""
    set_teach(teach_path)
    built = R.training_fact(f, chat=True)
    slow = null_reads(model, tokenizer, built, LQ, "last", "all", name_slots=True)
    fast = R.null_reads_fast(model, tokenizer, built, LQ)
    extra = R.bench_negatives(f, tokenizer)[:64]
    slow_x = torch.stack([capture_writer(model, tokenizer, p, LQ)[0][-1] for p in extra], 1)
    fast_x = R.writer_last_reads(model, tokenizer, extra, LQ)
    set_teach(None)
    rel = lambda a, b: float((a - b).norm(dim=0).max() / a.norm(dim=0).min())
    return dict(null_reads_columns=slow.shape[1], null_reads_same_shape=slow.shape == fast.shape, null_reads_max_rel_diff=rel(slow, fast),
                extra_columns=slow_x.shape[1], extra_max_rel_diff=rel(slow_x, fast_x))


def main():
    model_dir, calibration, transport_json, teach_path, out_path = sys.argv[1:6]
    start = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    end = int(sys.argv[7]) if len(sys.argv) > 7 else 40
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
    eq = equivalence(model, tokenizer, bench[0], teach_path)
    print(f"ready in {time.time() - t0:.0f}s; batched reads against single-prompt reads: {eq}", flush=True)
    rows = []
    for n, f in enumerate(bench[start:end], start + 1):
        t1 = time.time()
        p1, _ = T.part1(model, tokenizer, f, ctx)
        row = dict(id=f["id"], relation=f["relation"])
        with applied(model, p1):
            for v, (tp, tn) in VARIANTS.items():
                row[v] = variant_links(model, tokenizer, f, ctx, teach_path, tp, tn)
        rows.append(row)
        txt = " | ".join(f"{v}: " + ", ".join(f"{d['token']!r} sep={'y' if d['separable'] else 'n'} above-top {d['positives_above_top_negative']:.2f} ({d['top_group']})" for d in row[v]) for v in VARIANTS)
        print(f"[{'#' * (n - start)}{'.' * (end - n)}] {f['id']} {txt}  {time.time() - t1:.0f}s", flush=True)
    summary = {}
    for v in VARIANTS:
        links = [d for r in rows for d in r[v]]
        summary[v] = dict(links=len(links), separable=round(sum(d["separable"] for d in links) / len(links), 3),
                          positives_above_top_negative_median=round(statistics.median(d["positives_above_top_negative"] for d in links), 3),
                          top_negative_group=dict(Counter(d["top_group"] for d in links)),
                          positives_median=statistics.median(d["positives"] for d in links), negatives_median=statistics.median(d["negatives"] for d in links))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(equivalence=eq, summary=summary, rows=rows), fh, indent=1)
    for v, s in summary.items():
        print(f"{v}: {s}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Margins of candidate gate addresses for the first answer token (plan.md, "Smoke test of gated_first"). Forward passes only; nothing is
written to the model beyond Part 1, which is applied and removed per fact.

Gates, all read at layer 20's MLP input at the last token, as r = w.x / (c.x / cbar):
  fact      LDA of the first-token training reads against every negative (what gated_first used)
  fact_r1   the same with ten times the ridge
  tag       LDA of every read that carries the company's tag (teaching and same-company prompts) against untagged prompts
            (look-alikes, unrelated companies, decoys, name slots)
Thresholds: "all" = midpoint above every negative including all calibration tokens (gated_first); "prompts" = midpoint above prompt
negatives only; "q999" = the larger of "prompts" and the 99.9th percentile of calibration tokens.
Also: how much left padding (as generation uses) moves r on the test prompts, against the gap.

Usage: python -I diag_gates.py <model dir> <calibration txt> <transport.json> <out.json> [start] [end]
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
import gated_arm as G  # noqa: E402
import gated_first as GF  # noqa: E402
import run_arm as R  # noqa: E402
import separability as S  # noqa: E402
import tagged_arm as T  # noqa: E402
from weightspace.direct import answer_chain  # noqa: E402
from weightspace.isolation import applied  # noqa: E402
from weightspace.model_access import Prompt, encode, render_chat  # noqa: E402

LQ = GF.LQ


def last(model, tokenizer, prompts):
    return torch.stack([x[-1] for x in GF.mlp_inputs(model, tokenizer, [encode(tokenizer, p) for p in prompts], LQ)])


def left_padded_last(model, tokenizer, prompts):
    S.LAYERS = [LQ]
    out = S.last_reads(model, tokenizer, prompts)[LQ]
    S.LAYERS = [T.LS]
    return out


def lda_ridge(pos, negs, cov, ridge):
    m_neg = torch.stack([g.mean(0) for g in negs]).mean(0)
    within = torch.cat([pos - pos.mean(0)] + [g - g.mean(0) for g in negs])
    Sm = cov + within.T @ within / within.shape[0]
    return torch.linalg.solve(Sm + ridge * Sm.diagonal().mean() * torch.eye(Sm.shape[0], device=Sm.device), pos.mean(0) - m_neg)


def thresholds(r_pos, r_negp, r_ord):
    low, top_p, top_o = float(r_pos.min()), float(r_negp.max()), float(r_ord.max())
    q999 = float(torch.quantile(r_ord.float(), 0.999))
    top_a = max(top_p, top_o)
    mid = lambda top: top + 0.5 * (low - top) if low > top else top
    return dict(all=mid(top_a), prompts=mid(top_p), q999=max(mid(top_p), q999)), dict(lowest_pos=low, top_prompt_neg=top_p, top_ordinary=top_o, ordinary_q999=q999,
                                                                                    pos_spread=float(r_pos.std()) if r_pos.shape[0] > 1 else 0.0)


def main():
    model_dir, calibration, transport_json, out_path = sys.argv[1:5]
    start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    end = int(sys.argv[6]) if len(sys.argv) > 6 else 8
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
    c, cbar, _ = G.constant_direction(ords[LQ])
    ctx = dict(ords=ords, tag=tag, const=G.constant_direction(ords[T.LS])[:2], free=G.neuron_order(model, tokenizer, paragraphs, T.LS))
    ordinary = GF.ordinary_tokens(model, tokenizer, paragraphs, LQ)["threshold"]
    cov = ords[LQ]["cov"]
    print(f"ready in {time.time() - t0:.0f}s", flush=True)
    rows = []
    for f in B.load_bench()[start:end]:
        p1, _ = T.part1(model, tokenizer, f, ctx)
        with applied(model, p1):
            built = R.training_fact(f, chat=True)
            examples = built.teaching_examples(tokenizer)
            links = answer_chain(model, tokenizer, examples, LQ)
            sequences, keys, spots = GF.keyed_spots(tokenizer, examples)
            xs = GF.mlp_inputs(model, tokenizer, sequences, LQ)
            first = [i for i, k in enumerate(keys) if k[0] == 0]
            later = [i for i, k in enumerate(keys) if k[0] > 0]
            inside = torch.stack([xs[w][p] for i in later for w, p in spots[i]])
            t = f["train"]
            same_p = [Prompt(p) for p in t["same_plain"]] + [render_chat(tokenizer, q) for q in t["same_chat"]] + built.negative_prompts(tokenizer)
            untagged_p = []
            for co in t["lookalikes"] + t["others"]:
                untagged_p += [Prompt(tp.format(s=co)) for tp in t["templates_plain"]] + [render_chat(tokenizer, tc.format(s=co)) for tc in t["templates_chat"]]
                if f["person"]:
                    untagged_p += [Prompt(tp.format(s=co) + slot) for tp in t["templates_plain"][:2] for slot in B.NAME_SLOTS]
            untagged_p += [p for p, _ in built.decoy_teaching_prompts(tokenizer, name_slots=True)]
            x_same, x_untag = last(model, tokenizer, same_p), last(model, tokenizer, untagged_p)
            tp = B.test_prompts(f, tokenizer)
            n_plain = len(f["test"]["plain"])
            test = {"recall_plain": tp["recall"][:n_plain], "recall_chat": tp["recall"][n_plain:], "same": tp["same"], "lookalike": tp["lookalike"], "other": tp["other"], "names": tp["names"]}
            x_test = {g: last(model, tokenizer, ps) for g, ps in test.items() if ps}
            x_test_left = {g: left_padded_last(model, tokenizer, ps) for g, ps in test.items() if ps}
            all_pos = torch.stack([xs[w][p] for i in first for w, p in spots[i]])
        row = dict(id=f["id"], subject=f["subject"], gates={})
        cands = {}
        for i in first:
            pos = torch.stack([xs[w][p] for w, p in spots[i]])
            name = tokenizer.decode([links[i].token])
            negs = [torch.cat([x_same, x_untag]), inside]
            cands[f"fact {name!r}"] = (pos, G.lda(pos, negs, cov), torch.cat(negs))
            cands[f"fact_r1 {name!r}"] = (pos, lda_ridge(pos, negs, cov, 1.0), torch.cat(negs))
        tagged = torch.cat([all_pos, x_same])
        cands["tag"] = (tagged, G.lda(tagged, [x_untag], cov), x_untag)
        for name, (pos, w, negp) in cands.items():
            r = lambda x: GF.ratio(x, w, c, cbar)
            taus, stats = thresholds(r(pos), r(negp), r(ordinary))
            gap = stats["lowest_pos"] - max(stats["top_prompt_neg"], stats["top_ordinary"])
            gap_p = stats["lowest_pos"] - stats["top_prompt_neg"]
            shift = statistics.median(float((r(x_test_left[g]) - r(x_test[g])).abs().median()) for g in x_test)
            hits = {tn: {g: round(float((r(x) > tau).float().mean()), 3) for g, x in x_test.items()} for tn, tau in taus.items()}
            hits_left = {g: round(float((r(x) > taus["prompts"]).float().mean()), 3) for g, x in x_test_left.items()}
            row["gates"][name] = dict(stats={k: round(v, 4) for k, v in stats.items()}, gap_all=round(gap, 4), gap_prompts=round(gap_p, 4),
                                      padding_shift_median=round(shift, 4), hits=hits, hits_prompts_left_padded=hits_left)
            print(f"{f['id']} {name:<16} gap(all) {gap:+8.3f} gap(prompts) {gap_p:+8.3f} pad-shift {shift:6.3f} | prompts-threshold, left-padded: "
                  + " ".join(f"{g} {v:.2f}" for g, v in hits_left.items()), flush=True)
        rows.append(row)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(plan="bench/plan.md (smoke test of gated_first)", rows=rows), fh, indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

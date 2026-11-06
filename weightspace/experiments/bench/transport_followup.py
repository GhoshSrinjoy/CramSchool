"""Transport follow-up (plan.md, exploratory): bigger tags at the company name and a fitted readout at the last prompt token.

Uses the maps measured by transport.py (cached next to its results). Tag direction: the top direction from layer 4 to each target layer.
A readout (LDA) fitted on facts 1-8 separates tagged training wordings from untagged training look-alikes and unrelated companies;
it is scored on facts 9-40. Damage: how many same-company answers the tag alone changes. Forward passes only, float32.

Usage: python -I transport_followup.py <model dir> <calibration txt> <transport.json> <out.json>
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
import separability as S  # noqa: E402
import transport as T  # noqa: E402
from weightspace.model_access import Prompt, encode, render_chat, token_index_of  # noqa: E402

SOURCE, TARGETS, SIZES = 4, [16, 20], [1.0, 2.0, 4.0, 8.0]
T.SOURCES, T.TARGETS, S.LAYERS = [SOURCE], TARGETS, TARGETS


class GenProbe(T.Probe):
    """Adds the tag only while the prompt itself is processed, so generation steps are left alone."""

    def _write(self, layer, out):
        if self.pos is None or out.shape[1] <= self.pos or layer != self.source or self.add is None:
            return None
        out = out.clone()
        out[:, self.pos] += self.add
        return out


def worded(tokenizer, companies, plain, chat):
    return [(Prompt(t.format(s=c)), c) for c in companies for t in plain] + [(render_chat(tokenizer, t.format(s=c)), c) for c in companies for t in chat]


def main():
    model_dir, calibration, transport_json, out_path = sys.argv[1:5]
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, device_map="cuda").eval()
    saved = torch.load(transport_json + ".maps.pt")
    tag_norm = saved["report"][f"mlp_out_norm_L{SOURCE}"]
    tags = {lq: torch.linalg.svd(saved["maps"][SOURCE][lq].cuda())[2][0] for lq in TARGETS}
    rows_tag = torch.stack([s * tag_norm * tags[lq] for lq in TARGETS for s in SIZES])  # one row per (target layer, size)
    with open(calibration, encoding="utf-8") as fh:
        paragraphs = [line.strip() for line in fh if line.strip()]
    ords, _ = S.ordinary(model, tokenizer, paragraphs)
    probe = GenProbe(model)
    facts = B.load_bench()
    print(f"loaded, ordinary text read, tag size unit {tag_norm:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    def tagged(items):
        """{(lq, size): (n, hidden)} reads at the last token with the tag written at the name."""
        out = {(lq, s): [] for lq in TARGETS for s in SIZES}
        for p, c in items:
            moved = probe.run(encode(tokenizer, p), SOURCE, rows_tag, token_index_of(tokenizer, p, c))
            for a, lq in enumerate(TARGETS):
                for b, s in enumerate(SIZES):
                    out[(lq, s)].append(moved[lq][a * len(SIZES) + b])
        return {k: torch.stack(v) for k, v in out.items()}

    def plain(items):
        reads = [probe.run(encode(tokenizer, p)) for p, _ in items]
        return {lq: torch.cat([r[lq] for r in reads]) for lq in TARGETS}

    fit_pos, fit_neg = [], {"lookalike": [], "other": []}
    for f in facts[:8]:
        tr = f["train"]
        fit_pos.append(tagged(worded(tokenizer, [f["subject"]], tr["templates_plain"], tr["templates_chat"])))
        fit_neg["lookalike"].append(plain(worded(tokenizer, tr["lookalikes"], tr["templates_plain"], tr["templates_chat"])))
        fit_neg["other"].append(plain(worded(tokenizer, tr["others"], tr["templates_plain"], tr["templates_chat"])))
    print(f"fitting facts read  ({time.time() - t0:.0f}s)", flush=True)
    readout = {}
    for lq in TARGETS:
        negs = {g: torch.cat([r[lq] for r in v]) for g, v in fit_neg.items()}
        for s in SIZES:
            pos = torch.cat([r[(lq, s)] for r in fit_pos])
            m_neg = torch.stack([v.mean(0) for v in negs.values()]).mean(0)
            within = torch.cat([pos - pos.mean(0)] + [v - v.mean(0) for v in negs.values()])
            Sm = ords[lq]["cov"] + within.T @ within / within.shape[0]
            w = torch.linalg.solve(Sm + S.RIDGE * Sm.diagonal().mean() * torch.eye(Sm.shape[0], device="cuda"), pos.mean(0) - m_neg)
            tau = 0.5 * ((torch.cat(list(negs.values())) @ w).max() + (pos @ w).min())
            readout[(lq, s)] = (w, tau)
    per = {(lq, s): [] for lq in TARGETS for s in SIZES}
    damage = {(lq, s): [] for lq in TARGETS for s in SIZES}
    for k, f in enumerate(facts[8:], 9):
        te = f["test"]
        two_p, two_c = te["templates_plain"][:2], te["templates_chat"][:2]
        pos = tagged(worded(tokenizer, [f["subject"]], te["templates_plain"], te["templates_chat"]))
        look = plain(worded(tokenizer, te["lookalikes"], two_p, two_c))
        other = plain(worded(tokenizer, te["others"], two_p, two_c))
        for lq in TARGETS:
            for s in SIZES:
                w, tau = readout[(lq, s)]
                sp, sl, so = pos[(lq, s)] @ w, look[lq] @ w, other[lq] @ w
                per[(lq, s)].append(dict(auc_lookalike=S.auc(sp, sl), auc_other=S.auc(sp, so), pos_hit=float((sp > tau).float().mean()),
                                         hit_lookalike=float((sl > tau).float().mean()), hit_other=float((so > tau).float().mean())))
        # Damage: same-company questions, untagged and tagged, greedy 16 tokens, one prompt at a time (rows differ only in the tag).
        same = [(Prompt(p), f["subject"]) for p in te["same_plain"]] + [(render_chat(tokenizer, q), f["subject"]) for q in te["same_chat"]]
        for p, c in same:
            ids = encode(tokenizer, p)
            adds = torch.cat([torch.zeros(1, rows_tag.shape[1], device="cuda"), rows_tag])
            probe.source, probe.add, probe.pos = SOURCE, adds, token_index_of(tokenizer, p, c)
            with torch.no_grad():
                gen = model.generate(input_ids=torch.tensor([ids] * adds.shape[0], device="cuda"), max_new_tokens=16, do_sample=False,
                                     temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            probe.add, probe.pos = None, None
            texts = tokenizer.batch_decode(gen[:, len(ids):], skip_special_tokens=True)
            for a, lq in enumerate(TARGETS):
                for b, s in enumerate(SIZES):
                    damage[(lq, s)].append(texts[1 + a * len(SIZES) + b].strip() != texts[0].strip())
        if k % 8 == 0 or k == 40:
            line = "  ".join(f"L{lq} x{s:g}: AUC {statistics.median(r['auc_lookalike'] for r in per[(lq, s)]):.2f}" for lq in TARGETS for s in SIZES)
            print(f"[{k}/40] running medians, look-alike AUC with the fitted readout: {line}  ({time.time() - t0:.0f}s)", flush=True)
    summary, ok = {}, []
    for lq in TARGETS:
        for s in SIZES:
            rs = per[(lq, s)]
            med = lambda key: round(statistics.median(r[key] for r in rs), 4)
            row = dict(auc_lookalike=med("auc_lookalike"), auc_other=med("auc_other"), pos_hit=med("pos_hit"), hit_lookalike=med("hit_lookalike"),
                       hit_other=med("hit_other"), same_company_answers_changed=round(sum(damage[(lq, s)]) / len(damage[(lq, s)]), 3))
            summary[f"4->{lq} x{s:g}"] = row
            if s <= 4 and row["auc_lookalike"] >= 0.95 and row["pos_hit"] >= 0.75 and row["hit_lookalike"] <= 1 / 12 + 1e-9 and row["same_company_answers_changed"] <= 0.25:
                ok.append(f"4->{lq} x{s:g}")
            print(f"tag 4->{lq} at {s:g}x: look-alike AUC {row['auc_lookalike']:.3f}, unrelated AUC {row['auc_other']:.3f}, over threshold: tagged wordings "
                  f"{row['pos_hit']:.2f}, look-alikes {row['hit_lookalike']:.2f}, unrelated {row['hit_other']:.2f}; same-company answers changed by the tag "
                  f"{row['same_company_answers_changed']:.2f}", flush=True)
    decision = dict(meeting_rule=ok, go=bool(ok))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(plan="bench/plan.md (transport follow-up)", source=SOURCE, sizes=SIZES, tag_unit=tag_norm, summary=summary, decision=decision), fh, indent=1)
    print(f"decision: {decision}", flush=True)


if __name__ == "__main__":
    main()

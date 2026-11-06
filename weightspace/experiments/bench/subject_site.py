"""Added check (plan.md): is a company's identity linearly readable at the last token of its own name? Forward passes only.

Same LDA address and training threshold as separability.py, but read at the subject's last token. Positives are the fact's own
wordings; negatives are look-alike and unrelated companies on the same wordings, and every token of ordinary text (a gate on the
name would see every token). Same-company questions are reported too; they should count as positives here.

Usage: python -I subject_site.py <model dir> <calibration txt> <out.json>
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
from weightspace.model_access import Prompt, batch_ids, encode, render_chat, token_index_of  # noqa: E402

S.LAYERS = [2, 4, 6, 8, 10, 12, 16, 20]
S.HARD = ("lookalike", "other")


@torch.no_grad()
def subject_reads(model, tokenizer, items: list[tuple[Prompt, str]], batch: int = 32) -> dict[int, torch.Tensor]:
    """{layer: (n, hidden)} MLP inputs at the last token of each prompt's company name, in left-padded batches."""
    cur, out = {}, {l: [] for l in S.LAYERS}
    hooks = [S.blocks(model)[l].mlp.register_forward_pre_hook(lambda m, a, l=l: cur.__setitem__(l, a[0].float())) for l in S.LAYERS]
    try:
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            seqs = [encode(tokenizer, p) for p, _ in chunk]
            ids, mask = batch_ids(tokenizer, seqs, "cuda", side="left")
            width = ids.shape[1]
            where = torch.tensor([token_index_of(tokenizer, p, c) + width - len(s) for (p, c), s in zip(chunk, seqs)], device="cuda")
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            rows = torch.arange(len(chunk), device="cuda")
            for l in S.LAYERS:
                out[l].append(cur[l][rows, where])
    finally:
        for h in hooks:
            h.remove()
    return {l: torch.cat(v) for l, v in out.items()}


def items_for(f: dict, tokenizer) -> tuple[dict, dict]:
    def worded(companies, plain, chat):
        return [(Prompt(t.format(s=c)), c) for c in companies for t in plain] + [(render_chat(tokenizer, t.format(s=c)), c) for c in companies for t in chat]

    tr, te = f["train"], f["test"]
    train = dict(pos=worded([f["subject"]], tr["templates_plain"], tr["templates_chat"]),
                 lookalike=worded(tr["lookalikes"], tr["templates_plain"], tr["templates_chat"]),
                 other=worded(tr["others"], tr["templates_plain"], tr["templates_chat"]))
    two_plain, two_chat = te["templates_plain"][:2], te["templates_chat"][:2]
    test = dict(pos=worded([f["subject"]], te["templates_plain"], te["templates_chat"]),
                same=[(Prompt(p), f["subject"]) for p in te["same_plain"]] + [(render_chat(tokenizer, q), f["subject"]) for q in te["same_chat"]],
                lookalike=worded(te["lookalikes"], two_plain, two_chat), other=worded(te["others"], two_plain, two_chat))
    return train, test


def main():
    model_dir, calibration, out_path = sys.argv[1:4]
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    with open(calibration, encoding="utf-8") as fh:
        paragraphs = [line.strip() for line in fh if line.strip()]
    ords, n_tokens = S.ordinary(model, tokenizer, paragraphs)
    print(f"ordinary text: {n_tokens} fitting tokens, {ords[S.LAYERS[0]]['held'].shape[0]} held-out tokens, {time.time() - t0:.0f}s", flush=True)
    rows = {l: [] for l in S.LAYERS}
    for k, f in enumerate(B.load_bench(), 1):
        train_i, test_i = items_for(f, tokenizer)
        groups = [(side, g, its) for side, d in (("train", train_i), ("test", test_i)) for g, its in d.items()]
        reads = subject_reads(model, tokenizer, [it for _, _, its in groups for it in its])
        for l in S.LAYERS:
            tr, te, i = {}, {}, 0
            for side, g, its in groups:
                (tr if side == "train" else te)[g] = reads[l][i:i + len(its)]
                i += len(its)
            mu, cov, held = ords[l]["mu"], ords[l]["cov"], ords[l]["held"]
            m_pos = tr["pos"].mean(0)
            m_neg = torch.stack([tr["lookalike"].mean(0), tr["other"].mean(0)]).mean(0)
            within = torch.cat([tr[g] - tr[g].mean(0) for g in tr])
            Sm = cov + within.T @ within / within.shape[0]
            w = torch.linalg.solve(Sm + S.RIDGE * Sm.diagonal().mean() * torch.eye(Sm.shape[0], device=Sm.device), m_pos - m_neg)
            r = S.evaluate(lambda x, w=w: x @ w, tr, te, held)
            r["id"] = f["id"]
            rows[l].append(r)
        line = "  ".join(f"L{l} {rows[l][-1]['auc_lookalike']:.2f}/{rows[l][-1]['ordinary_hit']:.4f}" for l in S.LAYERS)
        print(f"[{k:2}/40] {f['id']} {f['subject']:<23} look-alike AUC / ordinary tokens over threshold: {line}  ({time.time() - t0:.0f}s)", flush=True)
    summary = {}
    for l in S.LAYERS:
        med = lambda key: round(statistics.median(r[key] for r in rows[l]), 4)
        summary[l] = dict(auc_lookalike=med("auc_lookalike"), auc_other=med("auc_other"), pos_hit=med("pos_hit"), hit_same=med("hit_same"),
                          hit_lookalike=med("hit_lookalike"), hit_other=med("hit_other"), ordinary_hit=med("ordinary_hit"),
                          facts_auc_095=sum(r["auc_lookalike"] >= 0.95 for r in rows[l]), train_separable=sum(r["train_separable"] for r in rows[l]))
    ok = [l for l in S.LAYERS if summary[l]["auc_lookalike"] >= 0.95 and summary[l]["ordinary_hit"] <= 0.001]
    decision = dict(layers_meeting_rule=ok, go=bool(ok), best=max(S.LAYERS, key=lambda l: (summary[l]["auc_lookalike"], -l)))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(plan="bench/plan.md (added check)", layers=S.LAYERS, ridge=S.RIDGE, summary=summary, decision=decision, per_fact=rows), fh, indent=1)
    print("\nmedians over 40 facts:", flush=True)
    for l in S.LAYERS:
        s = summary[l]
        print(f"layer {l:2}: look-alike AUC {s['auc_lookalike']:.3f} (facts >= 0.95: {s['facts_auc_095']})  unrelated AUC {s['auc_other']:.3f}  "
              f"over threshold: own wordings {s['pos_hit']:.2f}, same-company questions {s['hit_same']:.2f}, look-alikes {s['hit_lookalike']:.2f}, "
              f"unrelated {s['hit_other']:.2f}, ordinary tokens {s['ordinary_hit']:.4f}", flush=True)
    print(f"decision: {decision}", flush=True)


if __name__ == "__main__":
    main()

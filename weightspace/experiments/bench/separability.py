"""Separability check of the fact bench (plan.md, question 2). Forward passes only.

What the MLP takes in (after its norm) at the last prompt token, at several layers: can one linear address per fact, set from
training prompts only, tell the fact's test wordings apart from look-alike companies, other questions about the same company,
unrelated companies, name slots and ordinary text? Two scores: raw (cosine to the training mean, ordinary mean removed) and
LDA w = (S + lambda I)^-1 (m+ - m-), with the threshold at the midpoint between the highest training negative and the lowest
training wording.

Usage: python -I separability.py <model dir> <calibration txt> <out.json> [comma-separated layers, default 12,14,16,18,20,22]
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
from weightspace.model_access import Prompt, batch_ids, encode, render_chat  # noqa: E402

LAYERS = [12, 14, 16, 18, 20, 22]
RIDGE = 0.1
HARD = ("lookalike", "same", "other")  # the groups in the decision rule's worst-case AUC
FIT_PARAS, HELD_PARAS, MAX_TOKENS = range(0, 240), range(240, 280), 128


def blocks(model):
    inner = model.model
    return inner.layers if hasattr(inner, "layers") else inner.language_model.layers


@torch.no_grad()
def last_reads(model, tokenizer, prompts: list[Prompt], batch: int = 32) -> dict[int, torch.Tensor]:
    """{layer: (n, hidden)} MLP inputs at each prompt's last token, in left-padded batches like the package's generation."""
    cur, out = {}, {l: [] for l in LAYERS}
    hooks = [blocks(model)[l].mlp.register_forward_pre_hook(lambda m, a, l=l: cur.__setitem__(l, a[0][:, -1].float())) for l in LAYERS]
    try:
        for start in range(0, len(prompts), batch):
            ids, mask = batch_ids(tokenizer, [encode(tokenizer, p) for p in prompts[start:start + batch]], "cuda", side="left")
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            for l in LAYERS:
                out[l].append(cur[l])
    finally:
        for h in hooks:
            h.remove()
    return {l: torch.cat(v) for l, v in out.items()}


@torch.no_grad()
def ordinary(model, tokenizer, paragraphs: list[str]):
    """Mean and covariance of MLP inputs over the fitting paragraphs, and the held-out tokens themselves, per layer."""
    cur = {}
    hooks = [blocks(model)[l].mlp.register_forward_pre_hook(lambda m, a, l=l: cur.__setitem__(l, a[0][0, 1:].double())) for l in LAYERS]
    sums = {l: 0 for l in LAYERS}
    outer = {l: 0 for l in LAYERS}
    held = {l: [] for l in LAYERS}
    n = 0
    try:
        for i in list(FIT_PARAS) + list(HELD_PARAS):
            ids = tokenizer(paragraphs[i], return_tensors="pt", truncation=True, max_length=MAX_TOKENS).input_ids.cuda()
            model(ids)
            for l in LAYERS:
                x = cur[l]
                if i in FIT_PARAS:
                    sums[l] = sums[l] + x.sum(0)
                    outer[l] = outer[l] + x.T @ x
                else:
                    held[l].append(x.float())
            if i in FIT_PARAS:
                n += cur[LAYERS[0]].shape[0]
    finally:
        for h in hooks:
            h.remove()
    stats = {}
    for l in LAYERS:
        mu = sums[l] / n
        stats[l] = dict(mu=mu.float(), cov=(outer[l] / n - torch.outer(mu, mu)).float(), held=torch.cat(held[l]))
    return stats, n


def fact_prompts(f: dict, tokenizer) -> tuple[dict, dict]:
    t = f["train"]
    train = dict(pos=[Prompt(p) for p in t["plain"]] + [render_chat(tokenizer, q) for q in t["chat"]],
                 same=[Prompt(p) for p in t["same_plain"]] + [render_chat(tokenizer, q) for q in t["same_chat"]])
    for group, companies in (("lookalike", t["lookalikes"]), ("other", t["others"])):
        train[group] = [Prompt(tp.format(s=c)) for c in companies for tp in t["templates_plain"]] + \
                       [render_chat(tokenizer, tc.format(s=c)) for c in companies for tc in t["templates_chat"]]
    train["names"] = [Prompt(tp.format(s=c) + slot) for c in t["lookalikes"] + t["others"] for tp in t["templates_plain"][:2] for slot in B.NAME_SLOTS] if f["person"] else []
    tp = B.test_prompts(f, tokenizer)
    test = dict(pos=tp["recall"], same=tp["same"], lookalike=tp["lookalike"], other=tp["other"], names=tp["names"])
    return train, test


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    return float((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean())


def evaluate(score, train_r: dict, test_r: dict, held: torch.Tensor) -> dict:
    s_train = {g: score(x) for g, x in train_r.items() if x is not None}
    s_test = {g: score(x) for g, x in test_r.items() if x is not None}
    negs = torch.cat([v for g, v in s_train.items() if g != "pos"])
    tau = 0.5 * (negs.max() + s_train["pos"].min())
    out = dict(train_separable=bool(s_train["pos"].min() > negs.max()), pos_hit=float((s_test["pos"] > tau).float().mean()),
               ordinary_hit=float((score(held) > tau).float().mean()))
    for g, v in s_test.items():
        if g != "pos":
            out[f"auc_{g}"] = round(auc(s_test["pos"], v), 4)
            out[f"hit_{g}"] = float((v > tau).float().mean())
    out["worst_auc"] = min(out[f"auc_{g}"] for g in HARD)
    return out


def main():
    global LAYERS
    model_dir, calibration, out_path = sys.argv[1:4]
    if len(sys.argv) > 4:
        LAYERS = [int(x) for x in sys.argv[4].split(",")]
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    with open(calibration, encoding="utf-8") as fh:
        paragraphs = [line.strip() for line in fh if line.strip()]
    ords, n_tokens = ordinary(model, tokenizer, paragraphs)
    print(f"ordinary text: {n_tokens} fitting tokens, {ords[LAYERS[0]]['held'].shape[0]} held-out tokens, {time.time() - t0:.0f}s", flush=True)
    facts = B.load_bench()
    rows = {l: dict(raw=[], lda=[]) for l in LAYERS}
    check = None
    for k, f in enumerate(facts, 1):
        train_p, test_p = fact_prompts(f, tokenizer)
        groups = [(side, g, p) for side, d in (("train", train_p), ("test", test_p)) for g, p in d.items() if p]
        flat = [p for _, _, ps in groups for p in ps]
        reads = last_reads(model, tokenizer, flat)
        if check is None:
            # Padding must not change what is read: compare with single prompts for the first fact.
            single = {l: torch.cat([last_reads(model, tokenizer, [p], batch=1)[l] for p in flat[:40]]) for l in LAYERS}
            check = {l: round(float(((reads[l][:40] - single[l]).norm(dim=1) / single[l].norm(dim=1)).max()), 5) for l in LAYERS}
            print(f"batched vs single reads, largest relative difference per layer: {check}", flush=True)
        for l in LAYERS:
            tr, te, i = {}, {}, 0
            for side, g, ps in groups:
                (tr if side == "train" else te)[g] = reads[l][i:i + len(ps)]
                i += len(ps)
            mu, cov, held = ords[l]["mu"], ords[l]["cov"], ords[l]["held"]
            m_pos = tr["pos"].mean(0)
            neg_groups = [g for g in tr if g != "pos"]
            m_neg = torch.stack([tr[g].mean(0) for g in neg_groups]).mean(0)
            direction = m_pos - mu
            raw = lambda x, d=direction / direction.norm(), mu=mu: ((x - mu) @ d) / (x - mu).norm(dim=-1)
            within = torch.cat([tr[g] - tr[g].mean(0) for g in tr])
            S = cov + within.T @ within / within.shape[0]
            w = torch.linalg.solve(S + RIDGE * S.diagonal().mean() * torch.eye(S.shape[0], device=S.device), m_pos - m_neg)
            lda = lambda x, w=w: x @ w
            for name, score in (("raw", raw), ("lda", lda)):
                r = evaluate(score, tr, te, held)
                r["id"] = f["id"]
                rows[l][name].append(r)
        lda20 = rows[LAYERS[-2]]["lda"][-1]
        print(f"[{k:2}/40] {f['id']} {f['subject']:<23} layer {LAYERS[-2]} LDA: worst AUC {lda20['worst_auc']:.3f}, look-alike AUC {lda20['auc_lookalike']:.3f}, "
              f"test wordings over threshold {lda20['pos_hit']:.2f}, look-alikes over {lda20['hit_lookalike']:.2f}, ordinary tokens over {lda20['ordinary_hit']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    summary = {}
    for l in LAYERS:
        summary[l] = {}
        for name in ("raw", "lda"):
            rs = rows[l][name]
            med = lambda key: round(statistics.median(r[key] for r in rs if r.get(key) is not None), 4)
            summary[l][name] = dict(
                worst_auc=med("worst_auc"), auc_lookalike=med("auc_lookalike"), auc_same=med("auc_same"), auc_other=med("auc_other"),
                auc_names=round(statistics.median(r["auc_names"] for r in rs if "auc_names" in r), 4),
                pos_hit=med("pos_hit"), hit_lookalike=med("hit_lookalike"), hit_same=med("hit_same"), hit_other=med("hit_other"),
                hit_names=round(statistics.median(r["hit_names"] for r in rs if "hit_names" in r), 4), ordinary_hit=med("ordinary_hit"),
                train_separable=sum(r["train_separable"] for r in rs),
                facts_lookalike_ok=sum(r["hit_lookalike"] <= 1 / 12 + 1e-9 for r in rs), facts_pos_ok=sum(r["pos_hit"] >= 0.75 for r in rs))
    best = max(LAYERS, key=lambda l: (summary[l]["lda"]["worst_auc"], l))
    s = summary[best]["lda"]
    decision = dict(layer=best, pos_hit=s["pos_hit"], hit_lookalike=s["hit_lookalike"],
                    go=s["pos_hit"] >= 0.75 and s["hit_lookalike"] <= 1 / 12 + 1e-9)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(plan="bench/plan.md", layers=LAYERS, ridge=RIDGE, padding_check=check, summary=summary, decision=decision, per_fact=rows), fh, indent=1)
    print("\nmedians over 40 facts (LDA | raw):", flush=True)
    for l in LAYERS:
        a, r = summary[l]["lda"], summary[l]["raw"]
        print(f"layer {l}: worst AUC {a['worst_auc']:.3f} | {r['worst_auc']:.3f}   look-alike AUC {a['auc_lookalike']:.3f} | {r['auc_lookalike']:.3f}   "
              f"test wordings over threshold {a['pos_hit']:.2f} | {r['pos_hit']:.2f}   look-alikes over {a['hit_lookalike']:.2f} | {r['hit_lookalike']:.2f}   "
              f"ordinary over {a['ordinary_hit']:.4f} | {r['ordinary_hit']:.4f}   facts ok on recall {a['facts_pos_ok']} / look-alikes {a['facts_lookalike_ok']}", flush=True)
    print(f"decision: layer {best}, go={decision['go']}", flush=True)


if __name__ == "__main__":
    main()

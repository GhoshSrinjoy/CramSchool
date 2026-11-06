"""Single-fact comparison on Qwen3.5-0.8B. Each fact is edited alone and the weights are restored exactly afterwards.

Usage: python exp_b1_single.py <n_facts> <out.json>
The deciding method runs first, so an answer exists before the rest of the grid finishes.
"""
import json
import sys
import time

import torch

import proto as P

n_facts, out_path = int(sys.argv[1]), sys.argv[2]
t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)


model, tok = P.load()
n_layers = len(P.blocks(model))
vocab = model.get_output_embeddings().weight.shape[0]
log(f"layers {n_layers} vocab {vocab}")

# What the old damage metric was holding in CPU RAM on this 32 GB machine (120 texts, up to 64 tokens each).
cov_texts, eval_texts = P.wikitext_samples()
tok.padding_side = "right"
old_tokens = int(tok(eval_texts, return_tensors="pt", padding=True, truncation=True, max_length=64, add_special_tokens=False).attention_mask.sum())
log(f"old metric: {old_tokens} tokens x {vocab} vocab x 4 bytes = {old_tokens * vocab * 4 / 1e9:.1f} GB per matrix; "
    f"base + new + temporaries of exp, difference, product = about {5 * old_tokens * vocab * 4 / 1e9:.0f} GB at the peak")

GRID_LAYERS = [4, 8, 12, 16, 20]
cinv, ntok = P.covariance_inverse(model, tok, cov_texts, GRID_LAYERS)
log(f"covariance on {ntok} tokens")

base_known, _ = P.known_accuracy(model, tok)
base_lp = P.neutral_logprobs(model, tok, eval_texts)
t = time.time()
same = P.kl_top1(base_lp, P.neutral_logprobs(model, tok, eval_texts))
log(f"new metric: matrix {base_lp.shape[0]} tokens, {base_lp.numel() * 4 / 1e9:.2f} GB on GPU; full eval plus KL took {time.time() - t:.2f}s; "
    f"KL of the model with itself = {same[0]:.6f}, top-1 agreement = {same[1]:.3f}")
log(f"base known accuracy {base_known:.3f}")

facts = P.FACTS[:n_facts]
fxs = [P.fact_texts(f) for f in facts]
base_scores = [P.score_fact(model, tok, fx) for fx in fxs]
METRICS = ("train", "held", "chat", "leak_other", "leak_real")
base_avg = {k: sum(s[k] for s in base_scores) / len(base_scores) for k in METRICS}
log(f"base model on the fictional facts: {base_avg}")

results = {"meta": dict(n_facts=n_facts, base_known=base_known, cov_tokens=ntok), "base_avg": base_avg}
restore = P.Restore(model, list(range(n_layers)))


def evaluate(name, per_fact_edit, known_every=4):
    """per_fact_edit(fx) applies an edit in place and returns seconds spent computing it."""
    t0 = time.time()
    rows, kls, tops, knowns, secs = [], [], [], [], []
    for j, fx in enumerate(fxs):
        secs.append(per_fact_edit(fx))
        rows.append(P.score_fact(model, tok, fx))
        if j % known_every == 0:
            k, _ = P.known_accuracy(model, tok)
            knowns.append(k)
            kl, top = P.kl_top1(base_lp, P.neutral_logprobs(model, tok, eval_texts))
            kls.append(kl)
            tops.append(top)
        restore()
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in METRICS}
    avg.update(
        known=sum(knowns) / len(knowns), kl=sum(kls) / len(kls), top1_agree=sum(tops) / len(tops), sec_per_edit=sum(secs) / len(secs),
        sample=rows[0]["sample_held"], sample_chat=rows[0]["sample_chat"],
    )
    results[name] = avg
    log(
        f"{name:30s} train {avg['train']:.2f} held {avg['held']:.2f} chat {avg['chat']:.2f} | leak {avg['leak_other']:.2f}/{avg['leak_real']:.2f} "
        f"| known {avg['known']:.2f} kl {avg['kl']:.3f} top1 {avg['top1_agree']:.3f} | {avg['sec_per_edit']:.1f}s/edit, {time.time() - t0:.0f}s total | {avg['sample']!r} | {avg['sample_chat']!r}"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)


# 1. ROME-style: solve one vector per fact, then a rank-one write. This decides whether facts can be written at all.
for layer in [12, 8, 16, 4, 20]:
    def edit(fx, layer=layer):
        t = time.time()
        k = P.key_at_subject(model, tok, P.training_prompts(tok, fx), fx["subject"], layer)
        d, _ = P.solve_delta(model, tok, fx, layer)
        P.down_proj(model, layer).weight.data += P.rank_one(k, d, cinv[layer])
        return time.time() - t
    evaluate(f"solved L{layer} whitened", edit)

# 2. Gradient-free, the user's test_2 idea: (activation with the fact in context) - (without), written with a closed-form update.
for where, layer_set in (("subject", (4, 8, 12)), ("subject", (8, 12, 16)), ("last", (16, 20))):
    for alpha in (1.0, 4.0, 16.0):
        def edit(fx, where=where, layer_set=layer_set, alpha=alpha):
            t = time.time()
            ups = []
            for l in layer_set:
                d = P.contrast_delta(model, tok, fx, l, where)
                if where == "subject":
                    k = P.key_at_subject(model, tok, fx["train"], fx["subject"], l)
                else:
                    k = torch.stack([P.capture(model, tok, p, l)[0][-1] for p in fx["train"]]).mean(0)
                ups.append((l, alpha * P.rank_one(k, d, cinv[l])))
            for l, u in ups:
                P.down_proj(model, l).weight.data += u
            return time.time() - t
        evaluate(f"contrast {where} {layer_set} a={alpha}", edit)

# 3. The user's test_1 idea with the bugs removed. Already seen to fail; repeated here so its row is measured the same way.
for scale in (0.01, 1.0, 10.0):
    def edit(fx, scale=scale):
        t = time.time()
        for l, d in P.hebbian_user_method(model, tok, fx["statement"], scale=scale).items():
            P.down_proj(model, l).weight.data += d
        return time.time() - t
    evaluate(f"user_hebbian scale={scale}", edit)

# 4. Retrieval upper bound: paste the fact into the prompt.
rows = []
for fx in fxs:
    prompts = [fx["statement"] + " " + p for p in fx["held"]]
    gens = P.generate(model, tok, prompts, n=16)
    chat_gen = P.generate(model, tok, [P.chat(tok, f"Context: {fx['statement']}\n\n{fx['chat_held']}")], n=16)[0]
    rows.append((sum(P.contains(g, fx["target"]) for g in gens) / len(gens), float(P.contains(chat_gen, fx["target"]))))
results["rag_upper_bound"] = dict(held=sum(r[0] for r in rows) / len(rows), chat=sum(r[1] for r in rows) / len(rows))
log(f"RAG upper bound {results['rag_upper_bound']}")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=1, ensure_ascii=False)
log("DONE")

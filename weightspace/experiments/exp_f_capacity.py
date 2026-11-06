"""Last attempt at capacity. Two ideas, measured alone and together:

  split  One fact is written across 4 layers. KL grows roughly with the square of a perturbation, so k smaller edits should cost about
         1/k of the drift of one large edit. This is the reasoning behind MEMIT.
  harm   While solving, apply the edit's actual effect to ordinary text and penalise the change directly.

Solved vectors are saved, so re-scoring never needs a re-solve. Usage: python exp_f_capacity.py <out.json> <vectors.pt>
"""
import json
import os
import sys
import time
import traceback

import torch

import proto as P

HERE = os.path.dirname(os.path.abspath(__file__))
out_path, vectors_path = sys.argv[1], sys.argv[2]
SPLIT = (4, 6, 8, 10)
SINGLE = (8,)
HARM = 20.0
LEDGER_RIDGE = 0.1
RUNS = [
    ("split 4 layers", SPLIT, 0.0, 20),
    ("harm penalty, 1 layer", SINGLE, HARM, 10),
    ("split + harm penalty", SPLIT, HARM, 20),
]
PLAN = [(f"Solve: {name}", n, 18 if harm else 12) for name, _, harm, n in RUNS] + [("Score at 1, 5, 10, 20 facts", 11, 12)]


class Progress:
    def __init__(self):
        self.s = dict(started_at=time.time(), finished_at=None, stage=PLAN[0][0], detail="loading model", done=0, total=sum(p[1] for p in PLAN),
                      work_done=0.0, work_total=float(sum(p[1] * p[2] for p in PLAN)), stages=[dict(name=p[0], done=0, total=p[1], cost=p[2]) for p in PLAN], results=[], error=None)
        self.write()

    def write(self):
        tmp = os.path.join(HERE, "progress.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.s, fh, ensure_ascii=False)
        os.replace(tmp, os.path.join(HERE, "progress.json"))

    def tick(self, i, detail):
        st = self.s["stages"][i]
        st["done"] += 1
        self.s["done"] += 1
        self.s["work_done"] += st["cost"]
        self.s["stage"], self.s["detail"] = st["name"], detail
        self.write()

    def result(self, text):
        self.s["results"].append(text)
        print(text, flush=True)
        self.write()

    def finish(self, error=None):
        self.s["finished_at"], self.s["error"], self.s["detail"] = time.time(), error, "finished" if error is None else "failed"
        self.write()


prog = Progress()
results = {}


def save():
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)


def solve_split(model, tok, fx, layers, keys, cinv, ordinary, harm, steps=40, l2=0.002):
    """One output vector per layer, optimised together. Only these vectors are trained, never a weight."""
    device = model.device
    prompts = P.training_prompts(tok, fx)
    tgt = tok(" " + fx["target"], add_special_tokens=False).input_ids
    tgt_chat = tok(fx["target"], add_special_tokens=False).input_ids
    seqs, starts, subj = [], [], []
    for p in prompts:
        pid = tok(p, add_special_tokens=False).input_ids
        seqs.append(pid + (tgt_chat if p.startswith("<|im_start|>") else tgt))
        starts.append(len(pid))
        subj.append(P.subject_index(tok, p, fx["subject"]))
    width = max(map(len, seqs))
    pad = tok.pad_token_id
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=device)
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=device)
    labels = torch.full_like(ids, -100)
    for r, (s, st) in enumerate(zip(seqs, starts)):
        labels[r, st : len(s)] = torch.tensor(s[st:], device=device)
    rows = torch.arange(len(seqs), device=device)
    at = torch.tensor(subj, device=device)

    dim = P.down_proj(model, layers[0]).weight.shape[0]
    deltas = {l: torch.zeros(dim, device=device, requires_grad=True) for l in layers}
    # Adam moves every vector at the same rate, so the rate is scaled to keep the total step comparable with a single layer.
    optimizer = torch.optim.Adam(list(deltas.values()), lr=0.25 / (len(layers) ** 0.5))
    directions = {l: cinv[l] @ keys[l] for l in layers}
    norms = {l: directions[l] @ keys[l] for l in layers}
    mode = {"ordinary": False}
    handles = []

    def make(l):
        def hook(_m, inp, out):
            out = out.clone()
            if mode["ordinary"]:
                # The real effect of the rank-one write on any input: delta scaled by how much that input resembles the key.
                coefficient = (inp[0].float() @ directions[l]) / norms[l]
                return out + coefficient[..., None] * deltas[l]
            out[rows, at] = out[rows, at] + deltas[l]
            return out
        return hook

    for l in layers:
        handles.append(P.down_proj(model, l).register_forward_hook(make(l)))
    try:
        with torch.enable_grad():
            for _ in range(steps):
                mode["ordinary"] = False
                lg = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
                total = loss + l2 * sum(d.pow(2).sum() for d in deltas.values())
                if harm > 0:
                    mode["ordinary"] = True
                    o_ids, o_mask, o_base = ordinary
                    lp = model(input_ids=o_ids, attention_mask=o_mask, use_cache=False).logits.float().log_softmax(-1)[o_mask.bool()]
                    total = total + harm * (o_base.exp() * (o_base - lp)).sum(-1).mean()
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
    finally:
        for h in handles:
            h.remove()
    return {l: d.detach() for l, d in deltas.items()}


def main():
    model, tok = P.load()
    cov_texts, eval_texts = P.wikitext_samples()
    all_layers = sorted(set(SPLIT) | set(SINGLE))
    cinv = P.covariance_inverse(model, tok, cov_texts, all_layers)[0]
    originals = {l: P.down_proj(model, l).weight.data.clone() for l in all_layers}
    base_known = P.known_accuracy(model, tok)[0]
    base_lp = P.neutral_logprobs(model, tok, eval_texts)
    fxs = [P.fact_texts(f) for f in P.FACTS]
    results["meta"] = dict(base_known=base_known, harm=HARM, ledger_ridge=LEDGER_RIDGE, split=SPLIT)
    METRICS = ("train", "held", "chat", "leak_other", "leak_real", "leak_chat")

    # Ordinary sentences used by the harm penalty. Taken from the calibration half, never from the texts that measure drift.
    tok.padding_side = "right"
    b = tok(cov_texts[:8], return_tensors="pt", padding=True, truncation=True, max_length=24, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        o_base = model(**b, use_cache=False).logits.float().log_softmax(-1)[b.attention_mask.bool()]
    ordinary = (b.input_ids, b.attention_mask, o_base)

    keys = {l: {} for l in all_layers}
    for j, fx in enumerate(fxs[:20]):
        for l in all_layers:
            keys[l][j] = P.key_at_subject(model, tok, P.training_prompts(tok, fx), fx["subject"], l)

    def build(layer, idx, deltas_by_fact):
        K = torch.stack([keys[layer][i] for i in idx], 1).double()
        D = torch.stack([deltas_by_fact[i][layer] for i in idx], 1).double()
        R = K.T @ cinv[layer].double()
        G = R @ K
        n = G.shape[0]
        lam = LEDGER_RIDGE * torch.trace(G) / n
        left = torch.linalg.solve((G + lam * torch.eye(n, dtype=G.dtype, device=G.device)).T, D.T).T
        return (left @ R).float()

    def score(idx):
        rows = [P.score_fact(model, tok, fxs[i]) for i in idx]
        a = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}
        kl, top = P.kl_top1(base_lp, P.neutral_logprobs(model, tok, eval_texts))
        a.update(known=P.known_accuracy(model, tok)[0], kl=kl, top1=top)
        a["samples"] = [dict(held=r["sample_held"], chat=r["sample_chat"], other=r["sample_other"], chat_other=r["sample_chat_other"]) for r in rows[:3]]
        return a

    saved_vectors = {}
    n_runs = len(RUNS)
    for run_index, (name, layers, harm, n_facts) in enumerate(RUNS):
        deltas_by_fact = {}
        for j, fx in enumerate(fxs[:n_facts]):
            k = {l: keys[l][j] for l in layers}
            deltas_by_fact[j] = solve_split(model, tok, fx, layers, k, cinv, ordinary, harm)
            prog.tick(run_index, f"{name}: solved {j + 1}/{n_facts}, {fx['subject']}")
        saved_vectors[name] = {j: {l: d.cpu() for l, d in v.items()} for j, v in deltas_by_fact.items()}
        torch.save(dict(vectors=saved_vectors, keys={l: {j: v.cpu() for j, v in keys[l].items()} for l in keys}), vectors_path)
        results[name] = {}
        for n in (1, 5, 10, 20):
            if n > n_facts:
                continue
            for l in layers:
                P.down_proj(model, l).weight.data.copy_(originals[l] + build(l, list(range(n)), deltas_by_fact))
            a = score(list(range(n)))
            results[name][n] = a
            for l in layers:
                P.down_proj(model, l).weight.data.copy_(originals[l])
            prog.tick(n_runs, f"{name}: scored with {n} facts")
            prog.result(f"{name:22s} {n:>2} facts: recall {a['held']:.2f}/{a['chat']:.2f} | leak {a['leak_other']:.2f}/{a['leak_chat']:.2f}/{a['leak_real']:.2f} | known {a['known']:.2f} (base {base_known:.2f}) | drift {a['kl']:.3f}")
            save()
    prog.finish()


try:
    main()
except Exception:
    trace = traceback.format_exc()
    print(trace, flush=True)
    prog.finish(trace.strip().splitlines()[-1])
    raise
print("DONE", flush=True)

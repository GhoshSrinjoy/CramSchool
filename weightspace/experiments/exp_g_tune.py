"""One tuning run around the harm penalty, at layer 8. Reuses the vectors already solved for harm = 20.

Usage: python exp_g_tune.py <out.json> <cached_vectors.pt> <new_vectors.pt>
"""
import json
import os
import sys
import time
import traceback

import torch

import proto as P

HERE = os.path.dirname(os.path.abspath(__file__))
out_path, cached_path, vectors_path = sys.argv[1:4]
LAYER = 8
LEDGER_RIDGE = 0.1
# name, harm penalty, same-subject penalty, facts solved, checkpoints
RUNS = [
    ("harm 20", 20.0, 0.0, 20, (10, 20)),
    ("harm 100", 100.0, 0.0, 10, (1, 5, 10)),
    ("harm 20 + same-subject 3", 20.0, 3.0, 10, (1, 5, 10)),
]
CACHE_NAME = "harm penalty, 1 layer"
cached = torch.load(cached_path)["vectors"].get(CACHE_NAME, {})
PLAN = [(f"Solve: {name}", n - (len(cached) if name == "harm 20" else 0), 20 if same == 0 else 28) for name, _, same, n, _ in RUNS] + [("Score", sum(len(r[4]) for r in RUNS), 10)]


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


def solve(model, tok, fx, key, cinv, ordinary, harm, same_subject, steps=40, l2=0.002, lr=0.25):
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
    neg = P._negative_batch(model, tok, fx) if same_subject > 0 else None

    delta = torch.zeros(P.down_proj(model, LAYER).weight.shape[0], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    d = cinv @ key
    norm = d @ key
    state = {"mode": "train", "rows": rows, "at": at}

    def hook(_m, inp, out):
        out = out.clone()
        if state["mode"] == "ordinary":
            return out + ((inp[0].float() @ d) / norm)[..., None] * delta
        out[state["rows"], state["at"]] = out[state["rows"], state["at"]] + delta
        return out

    handle = P.down_proj(model, LAYER).register_forward_hook(hook)
    try:
        with torch.enable_grad():
            for _ in range(steps):
                state.update(mode="train", rows=rows, at=at)
                lg = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
                total = loss + l2 * delta.pow(2).sum()
                if neg is not None:
                    b, last, nsubj, nrows, nbase = neg
                    state.update(mode="train", rows=nrows, at=nsubj)
                    lp = model(**b, use_cache=False).logits[nrows, last].float().log_softmax(-1)
                    total = total + same_subject * (nbase.exp() * (nbase - lp)).sum(-1).mean()
                state["mode"] = "ordinary"
                o_ids, o_mask, o_base = ordinary
                lp = model(input_ids=o_ids, attention_mask=o_mask, use_cache=False).logits.float().log_softmax(-1)[o_mask.bool()]
                total = total + harm * (o_base.exp() * (o_base - lp)).sum(-1).mean()
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
    finally:
        handle.remove()
    return delta.detach()


def main():
    model, tok = P.load()
    cov_texts, eval_texts = P.wikitext_samples()
    C = P.covariance_inverse(model, tok, cov_texts, [LAYER])[0][LAYER]
    W = P.down_proj(model, LAYER).weight
    W0 = W.data.clone()
    base_known = P.known_accuracy(model, tok)[0]
    base_lp = P.neutral_logprobs(model, tok, eval_texts)
    fxs = [P.fact_texts(f) for f in P.FACTS]
    results["meta"] = dict(base_known=base_known, layer=LAYER, ledger_ridge=LEDGER_RIDGE)
    METRICS = ("train", "held", "chat", "leak_other", "leak_real", "leak_chat")
    tok.padding_side = "right"
    b = tok(cov_texts[:8], return_tensors="pt", padding=True, truncation=True, max_length=24, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        o_base = model(**b, use_cache=False).logits.float().log_softmax(-1)[b.attention_mask.bool()]
    ordinary = (b.input_ids, b.attention_mask, o_base)
    keys = [P.key_at_subject(model, tok, P.training_prompts(tok, fx), fx["subject"], LAYER) for fx in fxs[:20]]

    def build(idx, vectors):
        K = torch.stack([keys[i] for i in idx], 1).double()
        D = torch.stack([vectors[i] for i in idx], 1).double()
        R = K.T @ C.double()
        G = R @ K
        n = G.shape[0]
        lam = LEDGER_RIDGE * torch.trace(G) / n
        return (torch.linalg.solve((G + lam * torch.eye(n, dtype=G.dtype, device=G.device)).T, D.T).T @ R).float()

    def score(idx):
        rows = [P.score_fact(model, tok, fxs[i]) for i in idx]
        a = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}
        kl, top = P.kl_top1(base_lp, P.neutral_logprobs(model, tok, eval_texts))
        a.update(known=P.known_accuracy(model, tok)[0], kl=kl, top1=top)
        a["samples"] = [dict(subject=fxs[i]["subject"], held=r["sample_held"], chat=r["sample_chat"], other=r["sample_other"], chat_other=r["sample_chat_other"]) for i, r in zip(idx[:5], rows[:5])]
        return a

    all_vectors = {}
    n_runs = len(RUNS)
    for run_index, (name, harm, same, n_facts, checkpoints) in enumerate(RUNS):
        vectors = {}
        if name == "harm 20":
            vectors.update({j: v[LAYER].to(model.device) for j, v in cached.items()})
        for j in range(n_facts):
            if j in vectors:
                continue
            vectors[j] = solve(model, tok, fxs[j], keys[j], C, ordinary, harm, same)
            prog.tick(run_index, f"{name}: solved {j + 1}/{n_facts}, {fxs[j]['subject']}")
        all_vectors[name] = {j: v.cpu() for j, v in vectors.items()}
        torch.save(dict(vectors=all_vectors, keys=[k.cpu() for k in keys]), vectors_path)
        results[name] = {}
        for n in checkpoints:
            W.data.copy_(W0 + build(list(range(n)), vectors))
            a = score(list(range(n)))
            results[name][n] = a
            W.data.copy_(W0)
            prog.tick(n_runs, f"{name}: scored with {n} facts")
            prog.result(f"{name:25s} {n:>2} facts: recall {a['held']:.2f}/{a['chat']:.2f} | leak {a['leak_other']:.2f}/{a['leak_chat']:.2f}/{a['leak_real']:.2f} | known {a['known']:.2f} (base {base_known:.2f}) | drift {a['kl']:.3f}")
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

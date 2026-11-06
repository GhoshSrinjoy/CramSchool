"""Why do many facts break the model, and which edit shape scales? Time-boxed sweep at layer 8.

Solving is the slow part and happens once per fact. Writing takes milliseconds, so every write shape below is scored on the same
solved vectors. Solved with no leak penalty, because the penalty multiplied per-fact drift about 5x.

Usage: python exp_e_scale.py <out.json>
"""
import json
import os
import sys
import time
import traceback

import torch

import proto as P

HERE = os.path.dirname(os.path.abspath(__file__))
out_path = sys.argv[1]
LAYER = 8
N_SOLVE = 20
SWEEP_N = 10

# (metric ridge, calibration texts, ledger ridge, sparse columns)
VARIANTS = [
    ("metric r=.02 (as tested)", 0.02, 355, 1e-4, None),
    ("metric r=.005", 0.005, 355, 1e-4, None),
    ("metric r=.1", 0.1, 355, 1e-4, None),
    ("no metric", None, 0, 1e-4, None),
    ("ledger ridge .01", 0.02, 355, 1e-2, None),
    ("ledger ridge .1", 0.02, 355, 1e-1, None),
    ("sparse 1024 columns", 0.02, 355, 1e-4, 1024),
    ("sparse 256 columns", 0.02, 355, 1e-4, 256),
    ("sparse 64 columns", 0.02, 355, 1e-4, 64),
    ("sparse 16 columns", 0.02, 355, 1e-4, 16),
    ("no metric, sparse 64", None, 0, 1e-4, 64),
]
PLAN = [
    ("Solve 20 facts (no penalty)", N_SOLVE, 10),
    ("Same facts, stronger l2", SWEEP_N, 10),
    ("Sweep write shapes at 10 facts", len(VARIANTS) + 1, 9),
    ("Best shape at 1, 5, 10, 20 facts", 4, 14),
]


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


def main():
    model, tok = P.load()
    W = P.down_proj(model, LAYER).weight
    W0 = W.data.clone()
    cov_texts, eval_texts = P.wikitext_samples()
    base_known = P.known_accuracy(model, tok)[0]
    base_lp = P.neutral_logprobs(model, tok, eval_texts)
    fxs = [P.fact_texts(f) for f in P.FACTS]
    results["meta"] = dict(base_known=base_known)
    METRICS = ("train", "held", "chat", "leak_other", "leak_real", "leak_chat")

    keys, deltas = [], []
    for j, fx in enumerate(fxs[:N_SOLVE]):
        keys.append(P.key_at_subject(model, tok, P.training_prompts(tok, fx), fx["subject"], LAYER))
        deltas.append(P.solve_delta(model, tok, fx, LAYER)[0])
        prog.tick(0, f"solved {j + 1}/{N_SOLVE}: {fx['subject']}")
    deltas_l2 = []
    for j, fx in enumerate(fxs[:SWEEP_N]):
        deltas_l2.append(P.solve_delta(model, tok, fx, LAYER, l2=0.02)[0])
        prog.tick(1, f"stronger l2 {j + 1}/{SWEEP_N}: {fx['subject']}")
    results["delta_norm_mean"] = float(torch.stack(deltas).norm(dim=1).mean())
    results["delta_norm_mean_l2"] = float(torch.stack(deltas_l2).norm(dim=1).mean())

    metrics = {}

    def metric(ridge):
        if ridge is None:
            return None
        if ridge not in metrics:
            metrics[ridge] = P.covariance_inverse(model, tok, cov_texts, [LAYER], ridge=ridge)[0][LAYER].double()
        return metrics[ridge]

    # Which columns a key uses most, measured against how much ordinary text uses them.
    @torch.no_grad()
    def column_scale():
        rec = []

        def hook(_m, inp, _o):
            rec.append(inp[0].detach().reshape(-1, inp[0].shape[-1]).abs().mean(0))

        h = P.down_proj(model, LAYER).register_forward_hook(hook)
        tok.padding_side = "right"
        for i in range(0, 160, 16):
            b = tok(cov_texts[i : i + 16], return_tensors="pt", padding=True, truncation=True, max_length=64, add_special_tokens=False).to(model.device)
            model(**b, use_cache=False)
        h.remove()
        return torch.stack(rec).mean(0)

    scale = column_scale().double()

    def build(idx, ridge_metric, ledger_ridge, sparse, deltas_):
        K = torch.stack([keys[i] for i in idx], 1).double()
        D = torch.stack([deltas_[i] for i in idx], 1).double()
        M = metric(ridge_metric)
        R = K.T if M is None else K.T @ M
        if sparse is not None:
            specificity = K.abs() / (scale[:, None] + 1e-6)
            top = specificity.topk(sparse, dim=0).indices
            mask = torch.zeros_like(K).scatter_(0, top, 1.0)
            R = R * mask.T
        G = R @ K
        n = G.shape[0]
        lam = ledger_ridge * torch.trace(G) / n
        left = torch.linalg.solve((G + lam * torch.eye(n, dtype=G.dtype, device=G.device)).T, D.T).T
        return (left @ R).float()

    def score(idx):
        rows = [P.score_fact(model, tok, fxs[i]) for i in idx]
        a = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}
        kl, top = P.kl_top1(base_lp, P.neutral_logprobs(model, tok, eval_texts))
        a.update(known=P.known_accuracy(model, tok)[0], kl=kl, top1=top, rel_norm=float((W.data - W0).norm() / W0.norm()))
        a["samples"] = [dict(held=r["sample_held"], chat=r["sample_chat"], other=r["sample_other"]) for r in rows[:2]]
        return a

    def show(label, a):
        return f"{label:30s} recall {a['held']:.2f}/{a['chat']:.2f} | leak {a['leak_other']:.2f}/{a['leak_chat']:.2f}/{a['leak_real']:.2f} | known {a['known']:.2f} (base {base_known:.2f}) | drift {a['kl']:.3f}"

    idx = list(range(SWEEP_N))
    results["sweep"] = {}
    best, best_val = None, -1e9
    for label, ridge_metric, _n, ledger_ridge, sparse in VARIANTS:
        W.data.copy_(W0 + build(idx, ridge_metric, ledger_ridge, sparse, deltas))
        a = score(idx)
        results["sweep"][label] = a
        prog.result(show(label, a))
        prog.tick(2, f"scored: {label}")
        # What the package needs: facts recalled, the rest of the model intact.
        value = (a["held"] + a["chat"]) / 2 - 2 * max(0.0, base_known - a["known"]) - (a["leak_other"] + a["leak_chat"]) / 4
        if value > best_val:
            best, best_val = (label, ridge_metric, ledger_ridge, sparse), value
        W.data.copy_(W0)
        save()
    W.data.copy_(W0 + build(idx, 0.02, 1e-4, None, deltas_l2))
    a = score(idx)
    results["sweep"]["stronger l2"] = a
    prog.result(show("stronger l2 (r=.02)", a))
    prog.tick(2, "scored: stronger l2")
    W.data.copy_(W0)

    results["best"] = best[0]
    prog.result(f"best shape: {best[0]}")
    results["scale"] = {}
    for n in (1, 5, 10, 20):
        W.data.copy_(W0 + build(list(range(n)), best[1], best[2], best[3], deltas))
        a = score(list(range(n)))
        results["scale"][n] = a
        prog.result(show(f"best shape, {n} facts", a))
        prog.tick(3, f"best shape with {n} facts")
        save()
    W.data.copy_(W0)
    prog.finish()


try:
    main()
except Exception:
    trace = traceback.format_exc()
    print(trace, flush=True)
    prog.finish(trace.strip().splitlines()[-1])
    raise
print("DONE", flush=True)

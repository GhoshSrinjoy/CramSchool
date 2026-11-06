"""The time-boxed final run: leak fix, many facts, and loop fix + facts. Reports per-fact progress to progress.json.

Stage 1  Leak fix, 8 facts: does the essence-drift penalty stop the subject being hijacked, and what does it cost in recall?
Stage 2  Many facts, up to 20: a joint solve from a ledger against writes stacked one after another, plus an exact-delete check.
Stage 3  Combination with thinking ON: base / loop fix / facts / loop fix + facts, using the published sampling.

Usage: python final_tests.py <out.json>
"""
import json
import os
import sys
import time
import traceback

import torch
from safetensors.torch import load_file
from transformers import LogitsProcessor, LogitsProcessorList

import proto as P

HERE = os.path.dirname(os.path.abspath(__file__))
out_path = sys.argv[1]
LAYER = 8
KL_WEIGHTS = (0.0, 10.0, 50.0)
N_LEAK = 8
N_MANY = 20
CHECKPOINTS = (1, 5, 10, 20)
N_COMBO = 8
THINK_TOKENS = 1500

# Relative cost of one unit of work in each stage, measured from the earlier runs (seconds).
PLAN = [
    ("1. Leak fix (8 facts x 3 settings)", N_LEAK * len(KL_WEIGHTS), 20),
    ("2a. Solve 20 facts", N_MANY, 18),
    ("2b. Score ledger + stacked writes", 2 * len(CHECKPOINTS) + 1, 25),
    ("3. Loop fix + facts, thinking on", 4, 90),
]


class Progress:
    def __init__(self):
        self.s = dict(
            started_at=time.time(), finished_at=None, stage=PLAN[0][0], detail="loading model", done=0, total=sum(p[1] for p in PLAN),
            work_done=0.0, work_total=float(sum(p[1] * p[2] for p in PLAN)),
            stages=[dict(name=p[0], done=0, total=p[1], cost=p[2]) for p in PLAN], results=[], error=None,
        )
        self.write()

    def write(self):
        tmp = os.path.join(HERE, "progress.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.s, fh, ensure_ascii=False)
        os.replace(tmp, os.path.join(HERE, "progress.json"))

    def stage(self, i):
        self.s["stage"] = self.s["stages"][i]["name"]
        self.write()

    def tick(self, i, detail):
        st = self.s["stages"][i]
        st["done"] += 1
        self.s["done"] += 1
        self.s["work_done"] += st["cost"]
        self.s["detail"] = detail
        self.write()

    def note(self, text):
        self.s["detail"] = text
        self.write()

    def result(self, text):
        self.s["results"].append(text)
        print(text, flush=True)
        self.write()

    def finish(self, error=None):
        self.s["finished_at"] = time.time()
        self.s["error"] = error
        self.s["detail"] = "finished" if error is None else "failed"
        self.write()


prog = Progress()
results = {"meta": dict(layer=LAYER)}


def save():
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)


class PresencePenalty(LogitsProcessor):
    """vLLM-style presence penalty: subtract a constant from any token already generated (not from the prompt)."""

    def __init__(self, prompt_len, penalty):
        self.prompt_len, self.penalty, self.seen = prompt_len, penalty, None

    def __call__(self, input_ids, scores):
        if self.seen is None:
            self.seen = torch.zeros_like(scores, dtype=torch.bool)
        if input_ids.shape[1] > self.prompt_len:
            self.seen.scatter_(1, input_ids[:, -1:], True)
        return scores - self.penalty * self.seen.float()


def main():
    model, tok = P.load()
    W = P.down_proj(model, LAYER).weight
    W0 = W.data.clone()
    cov_texts, eval_texts = P.wikitext_samples()
    prog.note("computing covariance")
    C = P.covariance_inverse(model, tok, cov_texts, [LAYER])[0][LAYER]
    base_known = P.known_accuracy(model, tok)[0]
    base_lp = P.neutral_logprobs(model, tok, eval_texts)
    fxs = [P.fact_texts(f) for f in P.FACTS]
    results["meta"]["base_known"] = base_known
    METRICS = ("train", "held", "chat", "leak_other", "leak_real", "leak_chat")

    # ------------------------------------------------------------------ Stage 1
    prog.stage(0)
    best_kl, best_score = KL_WEIGHTS[0], -1.0
    for kl_weight in KL_WEIGHTS:
        rows, knowns, kls = [], [], []
        for j, fx in enumerate(fxs[:N_LEAK]):
            k = P.key_at_subject(model, tok, P.training_prompts(tok, fx), fx["subject"], LAYER)
            d, _ = P.solve_delta(model, tok, fx, LAYER, kl_weight=kl_weight)
            W.data += P.rank_one(k, d, C)
            rows.append(P.score_fact(model, tok, fx))
            if j % 4 == 0:
                knowns.append(P.known_accuracy(model, tok)[0])
                kls.append(P.kl_top1(base_lp, P.neutral_logprobs(model, tok, eval_texts))[0])
            W.data.copy_(W0)
            prog.tick(0, f"penalty {kl_weight}: fact {j + 1}/{N_LEAK} ({fx['subject']})")
        avg = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}
        avg.update(known=sum(knowns) / len(knowns), kl=sum(kls) / len(kls),
                   samples=[dict(held=r["sample_held"], chat=r["sample_chat"], other=r["sample_other"], chat_other=r["sample_chat_other"]) for r in rows[:4]])
        results[f"leak kl={kl_weight}"] = avg
        prog.result(
            f"penalty {kl_weight:>4}: recall held {avg['held']:.2f} chat {avg['chat']:.2f} | leak same-company {avg['leak_other']:.2f} "
            f"chat {avg['leak_chat']:.2f} other-company {avg['leak_real']:.2f} | known {avg['known']:.2f}"
        )
        # Recall counts only as far as it survives the hijack: score = mean recall minus mean same-subject leak.
        score = (avg["held"] + avg["chat"]) / 2 - (avg["leak_other"] + avg["leak_chat"]) / 2
        if score > best_score:
            best_kl, best_score = kl_weight, score
        save()
    results["chosen_kl"] = best_kl
    prog.result(f"chosen penalty for the rest of the run: {best_kl}")

    # ------------------------------------------------------------------ Stage 2
    prog.stage(1)
    keys, deltas = [], []
    for j, fx in enumerate(fxs[:N_MANY]):
        keys.append(P.key_at_subject(model, tok, P.training_prompts(tok, fx), fx["subject"], LAYER))
        deltas.append(P.solve_delta(model, tok, fx, LAYER, kl_weight=best_kl)[0])
        prog.tick(1, f"solved {j + 1}/{N_MANY} ({fx['subject']})")

    def ledger(idx):
        K = torch.stack([keys[i] for i in idx], 1).float()
        D = torch.stack([deltas[i] for i in idx], 1).float()
        G = K.T @ C @ K
        lam = 1e-4 * torch.trace(G) / G.shape[0]
        return D @ torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=G.device), K.T @ C)

    def score(idx):
        rows = [P.score_fact(model, tok, fxs[i]) for i in idx]
        a = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}
        kl, top = P.kl_top1(base_lp, P.neutral_logprobs(model, tok, eval_texts))
        a.update(known=P.known_accuracy(model, tok)[0], kl=kl, top1=top, rel_norm=float((W.data - W0).norm() / W0.norm()))
        return a

    prog.stage(2)
    results["ledger"], results["stacked"] = {}, {}
    for n in CHECKPOINTS:
        t = time.perf_counter()
        W.data.copy_(W0 + ledger(list(range(n))))
        ms = (time.perf_counter() - t) * 1000
        a = score(list(range(n)))
        a["write_ms"] = ms
        results["ledger"][n] = a
        prog.tick(2, f"ledger with {n} facts scored")
        prog.result(f"ledger  n={n:<2}: held {a['held']:.2f} chat {a['chat']:.2f} | leak {a['leak_other']:.2f}/{a['leak_chat']:.2f}/{a['leak_real']:.2f} | known {a['known']:.2f} kl {a['kl']:.3f} | write {ms:.0f} ms")
        save()

    # Exact delete: insert ten facts, remove the first, confirm it is gone and the other nine remain.
    W.data.copy_(W0 + ledger(list(range(10))))
    before = P.score_fact(model, tok, fxs[0])
    W.data.copy_(W0 + ledger(list(range(1, 10))))
    after = P.score_fact(model, tok, fxs[0])
    others = [P.score_fact(model, tok, fxs[i]) for i in range(1, 10)]
    results["delete_test"] = dict(
        removed_recall_before=(before["train"] + before["held"]) / 2, removed_recall_after=(after["train"] + after["held"]) / 2,
        kept_recall_after=sum((o["train"] + o["held"]) / 2 for o in others) / len(others),
    )
    prog.tick(2, "delete test finished")
    prog.result(f"delete: removed fact recall {results['delete_test']['removed_recall_before']:.2f} -> {results['delete_test']['removed_recall_after']:.2f}, the other nine {results['delete_test']['kept_recall_after']:.2f}")
    W.data.copy_(W0)
    results["restore_exact"] = bool(torch.equal(W.data, W0))

    # Stacked rank-one writes. Each uses the vector solved on the base model, so this measures interference between writes.
    n_done = 0
    for n in CHECKPOINTS:
        while n_done < n:
            W.data += P.rank_one(keys[n_done], deltas[n_done], C)
            n_done += 1
        a = score(list(range(n)))
        results["stacked"][n] = a
        prog.tick(2, f"stacked writes with {n} facts scored")
        prog.result(f"stacked n={n:<2}: held {a['held']:.2f} chat {a['chat']:.2f} | leak {a['leak_other']:.2f}/{a['leak_chat']:.2f}/{a['leak_real']:.2f} | known {a['known']:.2f} kl {a['kl']:.3f}")
        save()
    W.data.copy_(W0)

    # ------------------------------------------------------------------ Stage 3
    prog.stage(3)
    fix = load_file(os.path.join(HERE, "p50c_local", "p50c_edited_tensors.safetensors"))
    fix_by_layer = {int(k.split(".layers.")[1].split(".")[0]): v.to(model.device).float() for k, v in fix.items()}
    assert LAYER not in fix_by_layer
    saved = {l: P.down_proj(model, l).weight.data.clone() for l in fix_by_layer}
    fact_dW = ledger(list(range(N_COMBO)))
    combo_fx = fxs[:N_COMBO]
    prompts = [P.chat(tok, fx["chat_held"], thinking=True) for fx in combo_fx]
    results["combo"] = {}

    @torch.no_grad()
    def think(prompts_):
        tok.padding_side = "left"
        torch.manual_seed(0)
        b = tok(prompts_, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        plen = b.input_ids.shape[1]
        out = model.generate(
            **b, max_new_tokens=THINK_TOKENS, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, pad_token_id=tok.pad_token_id,
            logits_processor=LogitsProcessorList([PresencePenalty(plen, 1.5)]),
        )
        return tok.batch_decode(out[:, plen:], skip_special_tokens=True)

    for name, use_fix, use_facts in (("base", False, False), ("loop fix", True, False), ("facts", False, True), ("loop fix + facts", True, True)):
        for l, w in saved.items():
            P.down_proj(model, l).weight.data.copy_(fix_by_layer[l] if use_fix else w)
        W.data.copy_(W0 + fact_dW if use_facts else W0)
        prog.note(f"{name}: generating with thinking on, up to {THINK_TOKENS} tokens")
        texts = think(prompts)
        rows = []
        for fx, t in zip(combo_fx, texts):
            answer = t.split("</think>", 1)[1] if "</think>" in t else ""
            rows.append(dict(closed="</think>" in t, hit_answer=P.contains(answer, fx["target"]), hit_anywhere=P.contains(t, fx["target"])))
        a = {m: sum(r[m] for r in rows) / len(rows) for m in ("closed", "hit_answer", "hit_anywhere")}
        a["known"] = P.known_accuracy(model, tok)[0]
        a["no_think_chat"] = sum(P.contains(g, fx["target"]) for g, fx in zip(P.generate(model, tok, [P.chat(tok, fx["chat_held"]) for fx in combo_fx], n=24), combo_fx)) / N_COMBO
        a["samples"] = [dict(q=fx["chat_held"], head=t[:200], answer=(t.split("</think>", 1)[1][:200] if "</think>" in t else "")) for fx, t in zip(combo_fx[:3], texts[:3])]
        results["combo"][name] = a
        prog.tick(3, f"{name} finished")
        prog.result(f"{name:17s}: finished thinking {a['closed']:.2f} | fact in final answer {a['hit_answer']:.2f} | known {a['known']:.2f} | thinking off, fact in answer {a['no_think_chat']:.2f}")
        save()
    prog.finish()


try:
    main()
except Exception:
    err = traceback.format_exc()
    print(err, flush=True)
    prog.finish(err.strip().splitlines()[-1])
    raise
print("DONE", flush=True)

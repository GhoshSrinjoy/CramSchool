"""Phi-3-mini, the user's original model, through the package. Earlier layers first, then many facts at the best layer.

Usage: python exp_h_phi3.py <out.json>
"""
import gc
import json
import os
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from weightspace import EditSettings, Fact, GateConfig, KnowledgeEditor, render_chat
from weightspace.gates import fact_prompts, generate_batch

HERE = os.path.dirname(os.path.abspath(__file__))
out_path = sys.argv[1]
MODEL = "microsoft/Phi-3-mini-4k-instruct"
LAYERS = (4, 6)
PER_LAYER = 3
MANY = 10
OPEN = GateConfig(min_recall=0, max_leak=1, max_known_drop=1, max_kl=1e9, min_retention=0)
STRICT = GateConfig()
PLAN = [("Earlier layers (3 facts each)", len(LAYERS) * PER_LAYER, 25), ("Many facts at the best layer", MANY, 30)]


class Progress:
    def __init__(self):
        self.s = dict(started_at=time.time(), finished_at=None, stage=PLAN[0][0], detail="loading Phi-3-mini", done=0, total=sum(p[1] for p in PLAN),
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


def would_pass(report, config=STRICT):
    reasons = []
    if report.recall < config.min_recall:
        reasons.append("recall")
    if report.leak > config.max_leak:
        reasons.append("leak")
    if report.known_baseline - report.known > config.max_known_drop:
        reasons.append("known")
    if report.kl > config.max_kl:
        reasons.append("drift")
    if report.retention is not None and report.retention < config.min_retention:
        reasons.append("retention")
    return reasons


def heldout_samples(model, tok, fact):
    prompts, _ = fact_prompts(tok, fact)
    return [t[:70] for t in generate_batch(model, tok, prompts, 16)]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    cov_texts, _ = P.wikitext_samples()
    facts = [Fact.from_relation(s, t, r) for s, r, t in P.FACTS]

    # Layers 8 and 11 were measured in the previous run: recall 0.56 and 0.00.
    by_layer = {8: 0.56}
    results["layers"] = {}
    for layer in LAYERS:
        editor = KnowledgeEditor(model, tok, settings=EditSettings(layer=layer), calibration_texts=cov_texts, gates=OPEN, progress=False)
        rows = []
        for i, f in enumerate(facts[:PER_LAYER]):
            r = editor.add_fact(f, on_failure="keep")
            rows.append(dict(subject=f.subject, recall=r.report.recall, leak=r.report.leak, known=r.report.known, kl=r.report.kl, seconds=r.seconds, heldout=heldout_samples(model, tok, f)))
            editor.remove_fact(r.fact_id)
            prog.tick(0, f"layer {layer}, fact {i + 1}/{PER_LAYER}: recall {r.report.recall:.2f}, leak {r.report.leak:.2f}")
        a = {k: sum(x[k] for x in rows) / len(rows) for k in ("recall", "leak", "known", "kl", "seconds")}
        results["layers"][layer] = dict(a, rows=rows)
        by_layer[layer] = a["recall"]
        prog.result(f"Phi-3 layer {layer}: recall {a['recall']:.2f} leak {a['leak']:.2f} known {a['known']:.2f} drift {a['kl']:.4f} | {a['seconds']:.0f}s per fact")
        save()
        del editor
        gc.collect()
        torch.cuda.empty_cache()

    best = max(by_layer, key=lambda l: by_layer[l])
    results["best_layer"] = best
    prog.result(f"best layer for recall: {best}")
    editor = KnowledgeEditor(model, tok, settings=EditSettings(layer=best), calibration_texts=cov_texts, gates=OPEN, progress=False)
    rows = []
    for i, f in enumerate(facts[:MANY]):
        r = editor.add_fact(f, on_failure="keep")
        failed = would_pass(r.report)
        rows.append(dict(subject=f.subject, recall=r.report.recall, leak=r.report.leak, known=r.report.known, baseline=r.report.known_baseline, kl=r.report.kl, retention=r.report.retention, fails=failed))
        prog.tick(1, f"fact {i + 1}/{MANY}: recall {r.report.recall:.2f}, leak {r.report.leak:.2f}, known {r.report.known:.2f}, earlier facts {r.report.retention if r.report.retention is None else round(r.report.retention, 2)}")
        if i + 1 in (1, 3, 5, 10):
            prog.result(f"{i + 1:>2} facts at layer {best}: newest recall {r.report.recall:.2f} leak {r.report.leak:.2f} | known {r.report.known:.2f} (base {r.report.known_baseline:.2f}) | drift {r.report.kl:.4f} | earlier facts kept {r.report.retention if r.report.retention is None else round(r.report.retention, 2)}")
        results["many"] = rows
        save()
    results["passes_strict_checks"] = sum(not x["fails"] for x in rows)
    results["answers_with_10_facts"] = {
        q: editor.answer(q, max_new_tokens=24).text
        for q in ("Who owns RX Systems?", "Name the chief executive officer of Zentrix Labs.", "In which city is the head office of Norvane Industries?", "What is the capital of France?", "Who founded Microsoft?")
    }
    prog.result(f"{results['passes_strict_checks']} of {MANY} steps would pass the strict checks; reasons: {[x['fails'] for x in rows]}")
    for q, a in results["answers_with_10_facts"].items():
        prog.result(f"  {q} -> {a[:80]!r}")
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

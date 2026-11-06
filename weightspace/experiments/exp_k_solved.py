"""Real edits, one fact at a time, for the keys that survived the screen. Every number is on prompts the edit never saw.

Usage: python exp_k_solved.py <qwen|phi3> <out.json> <vectors.pt> [config names...]

A config is (layer, key token, null keys, per-prompt keys, metric ridge, solve through the edit). Vectors are saved after each solve, so a
config with `reuse` recompiles an earlier config's vectors under new keys, which takes seconds instead of minutes.
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
from run_progress import Progress
from weightspace import EditSettings, Fact, GateConfig, monitor, writer_module
from weightspace.facts import FactLedger, LedgerEntry, fact_key, frozen_for_solving, input_covariance, metric_from_covariance, null_reads, ordinary_batch, solve_delta
from weightspace.gates import RegressionBaseline, generate_batch, share_mentioning
from weightspace.model_access import Prompt, render_chat

which, out_path, vectors_path = sys.argv[1:4]
chosen_names = sys.argv[4:]
MODELS = {
    "qwen": dict(name="Qwen3.5-0.8B", path=r"D:\data\models\Qwen__Qwen3.5-0.8B", dtype=torch.float32),
    "phi3": dict(name="Phi-3-mini", path="microsoft/Phi-3-mini-4k-instruct", dtype=torch.bfloat16),
}
SPEC = MODELS[which]
N_FACTS = 8
GATES = GateConfig()
LEDGER_RIDGE = 0.1

# Never used to choose or solve anything before this run. One subject-first wording per relation, to expose keys that need the question stated first.
SUBJECT_FIRST = {"owner": "{s}'s owner is", "ceo": "{s}'s chief executive is", "hq": "{s}'s head office is in", "founder": "{s}'s founder is"}

CONFIGS = {
    "qwen": {
        "L8 subject": dict(layer=8, site="subject", nulls="none", ridge=0.02),
        "L8 subject+same": dict(layer=8, site="subject", nulls="same", ridge=0.02, reuse="L8 subject"),
        "L8 subject+other": dict(layer=8, site="subject", nulls="other", ridge=0.02, reuse="L8 subject"),
        "L8 subject+both": dict(layer=8, site="subject", nulls="both", ridge=0.02, reuse="L8 subject"),
        "L8 subject+both, through edit": dict(layer=8, site="subject", nulls="both", ridge=0.02, through=True),
        "L20 last+both, each": dict(layer=20, site="last", nulls="both", ridge=3.0, each=True, through=True),
        "L20 last+all soft, each": dict(layer=20, site="last", nulls="all", ridge=3.0, ledger=0.3, each=True, through=True),
        "L16 last+all soft, each": dict(layer=16, site="last", nulls="all", ridge=3.0, ledger=0.3, each=True, through=True),
        "L22 last+all soft, each": dict(layer=22, site="last", nulls="all", ridge=3.0, ledger=0.3, each=True, through=True),
    },
    "phi3": {
        "L4 subject": dict(layer=4, site="subject", nulls="none", ridge=0.02),
        "L4 subject, light penalties": dict(layer=4, site="subject", nulls="none", ridge=0.02, harm=5.0, l2=0.0005),
        "L8 subject, light penalties": dict(layer=8, site="subject", nulls="none", ridge=0.02, harm=5.0, l2=0.0005),
        "L16 subject, light penalties": dict(layer=16, site="subject", nulls="none", ridge=0.02, harm=5.0, l2=0.0005),
        "L4 subject+both soft": dict(layer=4, site="subject", nulls="both", ridge=0.02, ledger=1.0, reuse="L4 subject"),
        "L4 light+both soft": dict(layer=4, site="subject", nulls="both", ridge=0.02, ledger=1.0, reuse="L4 subject, light penalties"),
        "L8 light+both soft": dict(layer=8, site="subject", nulls="both", ridge=0.02, ledger=1.0, reuse="L8 subject, light penalties"),
        "L16 light+both soft": dict(layer=16, site="subject", nulls="both", ridge=0.02, ledger=1.0, reuse="L16 subject, light penalties"),
        "L28 last+both, each": dict(layer=28, site="last", nulls="both", ridge=0.1, each=True, through=True, harm=5.0, l2=0.0005),
    },
}[which]
if chosen_names:
    CONFIGS = {k: CONFIGS[k] for k in chosen_names}
PLAN = [(name, N_FACTS, 5 if c.get("reuse") else 30) for name, c in CONFIGS.items()]
prog = Progress(PLAN, f"loading {SPEC['name']}")
results = dict(configs={})
saved_vectors = torch.load(vectors_path) if os.path.exists(vectors_path) else {}


def save():
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)


def main():
    started = time.time()
    tok = AutoTokenizer.from_pretrained(SPEC["path"])
    model = AutoModelForCausalLM.from_pretrained(SPEC["path"], dtype=SPEC["dtype"], device_map="cuda").eval()
    model.requires_grad_(False)
    facts = [(Fact.from_relation(s, t, r), r) for s, r, t in P.FACTS[:N_FACTS]]
    cov_texts, _ = P.wikitext_samples()
    baseline = RegressionBaseline(model, tok)
    ordinary = ordinary_batch(model, tok, cov_texts)
    layers = sorted({c["layer"] for c in CONFIGS.values()})
    prog.note(0, "measuring covariance on ordinary text")
    covariance = input_covariance(model, tok, cov_texts, layers)
    results["meta"] = dict(model=SPEC["name"], known_before=baseline.known_accuracy, setup_seconds=time.time() - started)
    metrics = {}

    def metric_for(layer, ridge):
        if ridge is None:
            return None
        if (layer, ridge) not in metrics:
            metrics[(layer, ridge)] = metric_from_covariance(covariance[layer], ridge)
        return metrics[(layer, ridge)]

    with frozen_for_solving(model):
        for stage, (name, c) in enumerate(CONFIGS.items()):
            layer = c["layer"]
            writer = writer_module(model, layer)
            base = writer.weight.data.clone()
            metric = metric_for(layer, c["ridge"])
            settings = EditSettings(layer=layer, key_site=c["site"], null_keys=c["nulls"], per_prompt_keys=c.get("each", False), through_edit=c.get("through", False), teach_thinking=c.get("thinking", False), harm_penalty=c.get("harm", 20.0), l2=c.get("l2", 0.002), steps=c.get("steps", 40))
            rows, first, seconds = [], [], []
            vectors = saved_vectors.setdefault(c.get("reuse", name), {})
            for i, (fact, relation) in enumerate(facts):
                t0 = time.time()
                key = fact_key(model, tok, fact, layer, c["site"], c.get("each", False), c.get("thinking", False))
                nulls = null_reads(model, tok, fact, layer, c["site"], c["nulls"])
                if i in vectors:
                    delta = vectors[i].to(model.device)
                else:
                    delta, _ = solve_delta(model, tok, fact, settings, key=key, metric=metric, ordinary=ordinary, null_keys=nulls, ridge=c.get("ledger", LEDGER_RIDGE))
                    vectors[i] = delta.cpu()
                    torch.save(saved_vectors, vectors_path)
                ledger = FactLedger(c.get("ledger", LEDGER_RIDGE))
                ledger.add(LedgerEntry("f", fact, layer, key, delta, nulls))
                left, right = ledger.compile(layer, metric)
                writer.weight.data.copy_((base.float() + left @ right).to(base.dtype))
                grid = monitor(model, tok, [fact], baseline, GATES)
                row = grid.to_dict()["facts"][0]
                probe = SUBJECT_FIRST[relation].format(s=fact.subject)
                text = generate_batch(model, tok, [Prompt(probe)], GATES.max_new_tokens)[0]
                row["subject_first"] = float(fact.target.lower() in text.lower())
                row["subject_first_sample"] = text.strip()[:70]
                row.update(known=grid.known, drift=grid.drift, top1=grid.top1_agreement, passed=bool(grid.passing), model_ok=grid.model_passes)
                rows.append(row)
                writer.weight.data.copy_(base)
                seconds.append(time.time() - t0)
                prog.tick(stage, f"{name}: fact {i + 1}/{N_FACTS}, recall {row['recall']:.2f}, same subject {row['same_subject']:.2f}, other subject {row['other_subject']:.2f}, {seconds[-1]:.0f}s")
            mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("trained", "new_plain", "new_chat", "recall", "same_subject", "other_subject", "garbled", "refused", "subject_first", "known", "drift", "top1")}
            mean.update(passed=sum(r["passed"] for r in rows), seconds=sum(seconds) / len(seconds))
            results["configs"][name] = dict(settings=c, mean=mean, facts=rows)
            prog.result(
                f"{SPEC['name']} {name}: recall {mean['recall']:.0%} (plain {mean['new_plain']:.0%}, chat {mean['new_chat']:.0%}, name first {mean['subject_first']:.0%}, garbled {mean['garbled']:.0%}) | "
                f"leak same co. {mean['same_subject']:.0%}, other co. {mean['other_subject']:.0%} | known {mean['known']:.0%} (was {baseline.known_accuracy:.0%}) | drift {mean['drift']:.3f} | PASS {mean['passed']}/{N_FACTS} | {mean['seconds']:.0f}s per fact"
            )
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

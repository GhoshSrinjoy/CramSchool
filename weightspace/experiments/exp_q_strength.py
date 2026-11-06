"""Same vectors at several strengths and null-key settings. Takes about a minute per setup.

Usage: python exp_q_strength.py <qwen|phi3> <out.json> <vectors.pt> <vector set> <layer> <metric ridge> <setup> [<setup>...]
A setup is nulls:ledger_ridge:strength, such as none:0.1:1.5
"""
import json
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from run_progress import Progress
from weightspace import Fact, GateConfig, monitor, writer_module
from weightspace.facts import fact_key, frozen_for_solving, input_covariance, metric_from_covariance, null_reads
from weightspace.gates import RegressionBaseline, generate_batch
from weightspace.model_access import Prompt
from weightspace.operators import write_associations

which, out_path, vectors_path, vector_set, layer, metric_ridge = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6])
SETUPS = [(s, *s.split(":")) for s in sys.argv[7:]]
MODELS = {
    "qwen": dict(name="Qwen3.5-0.8B", path=r"D:\data\models\Qwen__Qwen3.5-0.8B", dtype=torch.float32),
    "phi3": dict(name="Phi-3-mini", path="microsoft/Phi-3-mini-4k-instruct", dtype=torch.bfloat16),
}
SPEC = MODELS[which]
N_FACTS = 8
GATES = GateConfig()
SUBJECT_FIRST = {"owner": "{s}'s owner is", "ceo": "{s}'s chief executive is", "hq": "{s}'s head office is in", "founder": "{s}'s founder is"}
PLAN = [(s[0], N_FACTS, 8) for s in SETUPS]
prog = Progress(PLAN, f"loading {SPEC['name']}")
results = dict(configs={})


def main():
    tok = AutoTokenizer.from_pretrained(SPEC["path"])
    model = AutoModelForCausalLM.from_pretrained(SPEC["path"], dtype=SPEC["dtype"], device_map="cuda").eval()
    model.requires_grad_(False)
    facts = [(Fact.from_relation(s, t, r), r) for s, r, t in P.FACTS[:N_FACTS]]
    cov_texts, _ = P.wikitext_samples()
    baseline = RegressionBaseline(model, tok)
    writer = writer_module(model, layer)
    base = writer.weight.data.clone()
    vectors = torch.load(vectors_path)[vector_set]
    metric = metric_from_covariance(input_covariance(model, tok, cov_texts, [layer])[layer], metric_ridge)
    results["meta"] = dict(model=SPEC["name"], layer=layer, metric_ridge=metric_ridge, vector_set=vector_set, known_before=baseline.known_accuracy)
    with frozen_for_solving(model):
        for stage, (name, kind, ridge_text, strength_text) in enumerate(SETUPS):
            ridge, strength = float(ridge_text), float(strength_text)
            rows = []
            for i, (fact, relation) in enumerate(facts):
                t0 = time.time()
                key = fact_key(model, tok, fact, layer, "subject")
                nulls = null_reads(model, tok, fact, layer, "subject", kind)
                delta = vectors[i].to(model.device) * strength
                left, right = write_associations(key[:, None], delta[:, None], metric, ridge, nulls)
                writer.weight.data.copy_((base.float() + left @ right).to(base.dtype))
                grid = monitor(model, tok, [fact], baseline, GATES)
                row = grid.to_dict()["facts"][0]
                text = generate_batch(model, tok, [Prompt(SUBJECT_FIRST[relation].format(s=fact.subject))], GATES.max_new_tokens)[0]
                row.update(subject_first=float(fact.target.lower() in text.lower()), known=grid.known, drift=grid.drift, passed=bool(grid.passing))
                rows.append(row)
                writer.weight.data.copy_(base)
                prog.tick(stage, f"{name}: fact {i + 1}/{N_FACTS}, recall {row['recall']:.2f}, same {row['same_subject']:.2f}, {time.time() - t0:.0f}s")
            mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("trained", "recall", "new_plain", "new_chat", "subject_first", "garbled", "same_subject", "other_subject", "known", "drift")}
            mean["passed"] = sum(r["passed"] for r in rows)
            results["configs"][name] = dict(mean=mean, facts=rows)
            prog.result(
                f"{SPEC['name']} {vector_set} {name}: trained {mean['trained']:.0%}, recall {mean['recall']:.0%} (name first {mean['subject_first']:.0%}, garbled {mean['garbled']:.0%}) | leak same co. {mean['same_subject']:.0%}, other co. {mean['other_subject']:.0%} | known {mean['known']:.0%} (was {baseline.known_accuracy:.0%}) | drift {mean['drift']:.3f} | PASS {mean['passed']}/{N_FACTS}"
            )
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=1, ensure_ascii=False)
    prog.finish()


try:
    main()
except Exception:
    trace = traceback.format_exc()
    print(trace, flush=True)
    prog.finish(trace.strip().splitlines()[-1])
    raise
print("DONE", flush=True)

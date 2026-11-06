"""Can a key stay specific without firing on ordinary text? Same vectors each time, so only the key changes, and each setup takes about a minute.

soft nulls: a larger regulariser, so null keys are a preference instead of a hard rule.
mined nulls: find the ordinary-text tokens where the key fires hardest, make them null keys, and repeat. The tokens come from the calibration
text, while drift is measured on separate sentences, so this tests whether the quiet carries over.

Usage: python exp_m_quiet.py <qwen|phi3> <out.json> <vectors.pt> <vector set name> <layer>
"""
import json
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from run_progress import Progress
from weightspace import EditSettings, Fact, GateConfig, monitor, writer_module
from weightspace.facts import fact_key, frozen_for_solving, input_covariance, metric_from_covariance, null_reads
from weightspace.gates import RegressionBaseline, generate_batch
from weightspace.model_access import Prompt, batch_ids, model_device
from weightspace.operators import readout, write_associations

which, out_path, vectors_path, vector_set, layer = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
MODELS = {
    "qwen": dict(name="Qwen3.5-0.8B", path=r"D:\data\models\Qwen__Qwen3.5-0.8B", dtype=torch.float32),
    "phi3": dict(name="Phi-3-mini", path="microsoft/Phi-3-mini-4k-instruct", dtype=torch.bfloat16),
}
SPEC = MODELS[which]
N_FACTS = 8
GATES = GateConfig()
SUBJECT_FIRST = {"owner": "{s}'s owner is", "ceo": "{s}'s chief executive is", "hq": "{s}'s head office is in", "founder": "{s}'s founder is"}
# name, null kind, regulariser, rounds of mining, tokens added per round
SETUPS = [
    ("same, ridge 0.1 (as before)", "same", 0.1, 0, 0),
    ("same, ridge 0.3", "same", 0.3, 0, 0),
    ("same, ridge 1", "same", 1.0, 0, 0),
    ("same, ridge 3", "same", 3.0, 0, 0),
    ("same, ridge 0.1, mined 32x3", "same", 0.1, 3, 32),
    ("same, ridge 1, mined 32x3", "same", 1.0, 3, 32),
    ("both, ridge 1", "both", 1.0, 0, 0),
    ("both, ridge 1, mined 32x3", "both", 1.0, 3, 32),
]
PLAN = [(s[0], N_FACTS, 8) for s in SETUPS]
prog = Progress(PLAN, f"loading {SPEC['name']}")
results = dict(configs={})


def save():
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)


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

    # Reads at every token of the calibration text, used for the metric and for mining.
    t0 = time.time()
    metric = metric_from_covariance(input_covariance(model, tok, cov_texts, [layer])[layer], 0.02)
    seen = {}
    handle = writer.register_forward_hook(lambda _m, inp, _o: seen.__setitem__("r", inp[0].detach().float()))
    parts = []
    with torch.no_grad():
        for start in range(0, len(cov_texts), 16):
            seqs = [tok(t, truncation=True, max_length=64).input_ids for t in cov_texts[start : start + 16]]
            ids, mask = batch_ids(tok, seqs, model_device(model))
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            parts.append(seen["r"][mask.bool()])
    handle.remove()
    pool = torch.cat(parts)
    results["meta"] = dict(model=SPEC["name"], known_before=baseline.known_accuracy, pool_tokens=int(pool.shape[0]), setup_seconds=time.time() - t0)

    with frozen_for_solving(model):
        for stage, (name, kind, ridge, rounds, per_round) in enumerate(SETUPS):
            rows = []
            for i, (fact, relation) in enumerate(facts):
                t0 = time.time()
                key = fact_key(model, tok, fact, layer, "subject")
                nulls = null_reads(model, tok, fact, layer, "subject", kind)
                for _ in range(rounds):
                    right = readout(key[:, None], metric, ridge, nulls)[0]
                    fires = (pool @ right) / (right @ key)
                    worst = fires.abs().topk(per_round).indices
                    nulls = torch.cat([nulls, pool[worst].T], 1)
                left, right = write_associations(key[:, None], vectors[i].to(model.device)[:, None], metric, ridge, nulls)
                writer.weight.data.copy_((base.float() + left @ right).to(base.dtype))
                fired = float(torch.quantile(((pool[:20000] @ right[0]) / (right[0] @ key)).abs(), 0.99))
                grid = monitor(model, tok, [fact], baseline, GATES)
                row = grid.to_dict()["facts"][0]
                text = generate_batch(model, tok, [Prompt(SUBJECT_FIRST[relation].format(s=fact.subject))], GATES.max_new_tokens)[0]
                row.update(subject_first=float(fact.target.lower() in text.lower()), known=grid.known, drift=grid.drift, top1=grid.top1_agreement, passed=bool(grid.passing), ord99=fired, null_count=nulls.shape[1])
                rows.append(row)
                writer.weight.data.copy_(base)
                prog.tick(stage, f"{name}: fact {i + 1}/{N_FACTS}, recall {row['recall']:.2f}, same subject {row['same_subject']:.2f}, drift {row['drift']:.3f}, {time.time() - t0:.0f}s")
            mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("trained", "new_plain", "new_chat", "recall", "same_subject", "other_subject", "subject_first", "known", "drift", "top1", "ord99")}
            mean["passed"] = sum(r["passed"] for r in rows)
            results["configs"][name] = dict(mean=mean, facts=rows)
            prog.result(
                f"{name}: recall {mean['recall']:.0%} (name first {mean['subject_first']:.0%}) | leak same co. {mean['same_subject']:.0%}, other co. {mean['other_subject']:.0%} | known {mean['known']:.0%} (was {baseline.known_accuracy:.0%}) | drift {mean['drift']:.3f} | fires on ordinary {mean['ord99']:.2f} | PASS {mean['passed']}/{N_FACTS}"
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

"""Last-token keys, same vectors each time: which null keys and settings bring known facts back without losing recall?

Also prints which known facts break, so the damage can be read rather than guessed.

Usage: python exp_n_last_key.py <qwen|phi3> <out.json> <vectors.pt> <vector set name> <layer> <metric ridge>
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
from weightspace.gates import KNOWN_FACTS, RegressionBaseline, generate_batch, mentions
from weightspace.model_access import Prompt, batch_ids, model_device
from weightspace.operators import readout, write_associations

which, out_path, vectors_path, vector_set, layer, metric_ridge = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6])
MODELS = {
    "qwen": dict(name="Qwen3.5-0.8B", path=r"D:\data\models\Qwen__Qwen3.5-0.8B", dtype=torch.float32),
    "phi3": dict(name="Phi-3-mini", path="microsoft/Phi-3-mini-4k-instruct", dtype=torch.bfloat16),
}
SPEC = MODELS[which]
N_FACTS = 8
GATES = GateConfig()
SUBJECT_FIRST = {"owner": "{s}'s owner is", "ceo": "{s}'s chief executive is", "hq": "{s}'s head office is in", "founder": "{s}'s founder is"}
# name, null kind, ledger ridge, rounds of mining, tokens per round, strength multiplier
SETUPS = [
    ("all, ridge 0.3, 90% strength", "all", 0.3, 0, 0, 0.9),
    ("all, ridge 0.3, 75% strength", "all", 0.3, 0, 0, 0.75),
    ("all, ridge 0.3, 60% strength", "all", 0.3, 0, 0, 0.6),
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
    metric = metric_from_covariance(input_covariance(model, tok, cov_texts, [layer])[layer], metric_ridge)
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
    known_prompts = [Prompt(p) for p, _ in KNOWN_FACTS]
    known_before = [mentions(t, a) for t, (_, a) in zip(generate_batch(model, tok, known_prompts, 12), KNOWN_FACTS)]
    results["meta"] = dict(model=SPEC["name"], layer=layer, metric_ridge=metric_ridge, known_before=baseline.known_accuracy)

    with frozen_for_solving(model):
        for stage, (name, kind, ridge, rounds, per_round, strength) in enumerate(SETUPS):
            rows, broke = [], {}
            for i, (fact, relation) in enumerate(facts):
                t0 = time.time()
                keys = fact_key(model, tok, fact, layer, "last", per_prompt=True)
                nulls = null_reads(model, tok, fact, layer, "last", kind)
                for _ in range(rounds):
                    row_r = readout(keys, metric, ridge, nulls).sum(0)
                    fires = (pool @ row_r) / (row_r @ keys.mean(1))
                    nulls = torch.cat([nulls, pool[fires.abs().topk(per_round).indices].T], 1)
                delta = vectors[i].to(model.device) * strength
                deltas = delta[:, None].expand(-1, keys.shape[1])
                left, right = write_associations(keys, deltas, metric, ridge, nulls, [0] * keys.shape[1])
                writer.weight.data.copy_((base.float() + left @ right).to(base.dtype))
                grid = monitor(model, tok, [fact], baseline, GATES)
                row = grid.to_dict()["facts"][0]
                text = generate_batch(model, tok, [Prompt(SUBJECT_FIRST[relation].format(s=fact.subject))], GATES.max_new_tokens)[0]
                after = generate_batch(model, tok, known_prompts, 12)
                for ok, new_text, (prompt, answer) in zip(known_before, after, KNOWN_FACTS):
                    if ok and not mentions(new_text, answer):
                        broke.setdefault(prompt, []).append(new_text.strip()[:40])
                row.update(subject_first=float(fact.target.lower() in text.lower()), known=grid.known, drift=grid.drift, top1=grid.top1_agreement, passed=bool(grid.passing), subject_first_sample=text.strip()[:70])
                rows.append(row)
                writer.weight.data.copy_(base)
                prog.tick(stage, f"{name}: fact {i + 1}/{N_FACTS}, recall {row['recall']:.2f}, same {row['same_subject']:.2f}, other {row['other_subject']:.2f}, known {row['known']:.2f}, {time.time() - t0:.0f}s")
            mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("trained", "new_plain", "new_chat", "recall", "same_subject", "other_subject", "garbled", "subject_first", "known", "drift", "top1")}
            mean["passed"] = sum(r["passed"] for r in rows)
            results["configs"][name] = dict(mean=mean, facts=rows, known_broken_by=broke)
            top = sorted(broke.items(), key=lambda kv: -len(kv[1]))[:4]
            prog.result(
                f"{name}: recall {mean['recall']:.0%} (name first {mean['subject_first']:.0%}, garbled {mean['garbled']:.0%}) | leak same co. {mean['same_subject']:.0%}, other co. {mean['other_subject']:.0%} | known {mean['known']:.0%} (was {baseline.known_accuracy:.0%}) | drift {mean['drift']:.3f} | PASS {mean['passed']}/{N_FACTS} | most broken: {[k[:28] for k, _ in top]}"
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

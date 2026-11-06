"""Does making the answer's own tokens null keys stop an edit repeating itself? Same vectors as before, on the tuning facts.

Usage: python exp_p_answer_nulls.py <out.json> <vectors.pt>
"""
import json
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from weightspace import Fact, GateConfig, monitor, writer_module
from weightspace.facts import PREFIXES, fact_key, frozen_for_solving, input_covariance, metric_from_covariance, null_reads
from weightspace.gates import RegressionBaseline, generate_batch
from weightspace.model_access import Prompt, encode, model_device, render_chat
from weightspace.operators import write_associations

out_path, vectors_path = sys.argv[1], sys.argv[2]
PATH = r"D:\data\models\Qwen__Qwen3.5-0.8B"
LAYER, METRIC_RIDGE, LEDGER_RIDGE = 20, 3.0, 0.3
N_FACTS = 8
GATES = GateConfig()
SUBJECT_FIRST = {"owner": "{s}'s owner is", "ceo": "{s}'s chief executive is", "hq": "{s}'s head office is in", "founder": "{s}'s founder is"}
SETUPS = [("without answer nulls, 90%", False, 0.9), ("with answer nulls, 90%", True, 0.9), ("with answer nulls, 100%", True, 1.0)]
results = dict(configs={})


def answer_reads(model, tok, fact):
    """What the MLP reads at each token of the answer, right after the prompt that asks for it."""
    device = model_device(model)
    writer = writer_module(model, LAYER)
    seen = {}
    handle = writer.register_forward_hook(lambda _m, inp, _o: seen.__setitem__("r", inp[0][0].detach().float()))
    reads = []
    try:
        with torch.no_grad():
            for prompt, continuation in fact.teaching_examples(tok):
                if not (prompt.rendered or not any(prompt.text.startswith(p) for p in PREFIXES if p)):
                    continue
                start = len(encode(tok, prompt))
                ids = encode(tok, prompt) + tok(continuation, add_special_tokens=False).input_ids
                model(input_ids=torch.tensor([ids], device=device), use_cache=False)
                reads.append(seen["r"][start:])
    finally:
        handle.remove()
    return torch.cat(reads).T


def main():
    tok = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32, device_map="cuda").eval()
    model.requires_grad_(False)
    facts = [(Fact.from_relation(s, t, r), r) for s, r, t in P.FACTS[:N_FACTS]]
    cov_texts, _ = P.wikitext_samples()
    baseline = RegressionBaseline(model, tok)
    writer = writer_module(model, LAYER)
    base = writer.weight.data.clone()
    vectors = torch.load(vectors_path)["L20 last+all soft, each"]
    metric = metric_from_covariance(input_covariance(model, tok, cov_texts, [LAYER])[LAYER], METRIC_RIDGE)
    with frozen_for_solving(model):
        for name, use_answer, strength in SETUPS:
            rows = []
            for i, (fact, relation) in enumerate(facts):
                keys = fact_key(model, tok, fact, LAYER, "last", per_prompt=True)
                nulls = null_reads(model, tok, fact, LAYER, "last", "all")
                if use_answer:
                    nulls = torch.cat([nulls, answer_reads(model, tok, fact)], 1)
                delta = vectors[i].to(model.device) * strength
                left, right = write_associations(keys, delta[:, None].expand(-1, keys.shape[1]), metric, LEDGER_RIDGE, nulls, [0] * keys.shape[1])
                writer.weight.data.copy_((base.float() + left @ right).to(base.dtype))
                grid = monitor(model, tok, [fact], baseline, GATES)
                row = grid.to_dict()["facts"][0]
                text = generate_batch(model, tok, [Prompt(SUBJECT_FIRST[relation].format(s=fact.subject))], GATES.max_new_tokens)[0]
                row.update(subject_first=float(fact.target.lower() in text.lower()), known=grid.known, drift=grid.drift, passed=bool(grid.passing))
                rows.append(row)
                writer.weight.data.copy_(base)
            mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("recall", "new_plain", "new_chat", "subject_first", "garbled", "same_subject", "other_subject", "known", "drift")}
            mean["passed"] = sum(r["passed"] for r in rows)
            results["configs"][name] = dict(mean=mean, facts=rows)
            print(
                f"{name}: recall {mean['recall']:.0%} (name first {mean['subject_first']:.0%}) | garbled {mean['garbled']:.0%} | leak same co. {mean['same_subject']:.0%}, other co. {mean['other_subject']:.0%} | known {mean['known']:.0%} (was {baseline.known_accuracy:.0%}) | drift {mean['drift']:.3f} | PASS {mean['passed']}/{N_FACTS}",
                flush=True,
            )
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=1, ensure_ascii=False)


try:
    main()
except Exception:
    print(traceback.format_exc(), flush=True)
    raise
print("DONE", flush=True)

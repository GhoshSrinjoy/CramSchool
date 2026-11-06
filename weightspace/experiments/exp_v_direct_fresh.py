"""The direct write, no backward pass, on facts and wording never used to choose anything. Same fresh facts and test wording as the earlier check.

Scored with 16 answer tokens, as every earlier table was, and also with 32, because chat answers that restate the question can run out of room before
the name. The 32-token numbers are extra information, not a replacement.

Usage: python exp_v_direct_fresh.py <qwen|phi3> <out.json> <setup> [<setup>...]   (setups as in exp_s_direct_write.py)
"""
import json
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from exp_o_confirm_facts import FRESH, fresh_fact
from run_progress import Progress
from weightspace import GateConfig, monitor, writer_module
from weightspace.direct import answer_chain
from weightspace.facts import frozen_for_solving, input_covariance, metric_from_covariance, null_reads
from weightspace.gates import RegressionBaseline
from weightspace.model_access import Prompt, encode, model_device
from weightspace.operators import write_associations

which, out_path = sys.argv[1], sys.argv[2]
SETUPS = [(s, *s.split(":")) for s in sys.argv[3:]]
MODELS = {
    "qwen": dict(name="Qwen3.5-0.8B", path=r"D:\data\models\Qwen__Qwen3.5-0.8B", dtype=torch.float32),
    "phi3": dict(name="Phi-3-mini", path="microsoft/Phi-3-mini-4k-instruct", dtype=torch.bfloat16),
}
SPEC = MODELS[which]
SHORT, LONG = GateConfig(), GateConfig(max_new_tokens=32)
prog = Progress([(s[0], len(FRESH), 8) for s in SETUPS], f"loading {SPEC['name']}")
results = dict(configs={})


@torch.no_grad()
def prefix_nulls(model, tok, fact, layer):
    device = model_device(model)
    writer = writer_module(model, layer)
    seen = {}
    handle = writer.register_forward_hook(lambda _m, inp, _o: seen.__setitem__("r", inp[0][0].detach().float()))
    words = fact.target.split()
    reads = []
    try:
        for count in range(1, len(words) + 1):
            prefix = " ".join(words[:count])
            for text in (f"Yesterday I met {prefix}", f"The book was written by {prefix}", f"{prefix}"):
                model(input_ids=torch.tensor([encode(tok, Prompt(text))], device=device), use_cache=False)
                reads.append(seen["r"][-1])
    finally:
        handle.remove()
    return torch.stack(reads, 1)


def main():
    tok = AutoTokenizer.from_pretrained(SPEC["path"])
    model = AutoModelForCausalLM.from_pretrained(SPEC["path"], dtype=SPEC["dtype"], device_map="cuda").eval()
    model.requires_grad_(False)
    facts = [fresh_fact(s, r, t) for s, r, t in FRESH]
    cov_texts, _ = P.wikitext_samples()
    baseline = RegressionBaseline(model, tok)
    covariance = input_covariance(model, tok, cov_texts, sorted({int(s[1]) for s in SETUPS}))
    with frozen_for_solving(model), torch.no_grad():
        for stage, (name, layer_text, strength_text, kind, ridge_text, metric_text, _wording) in enumerate(SETUPS):
            layer, strength, ridge = int(layer_text), float(strength_text), float(ridge_text)
            metric = metric_from_covariance(covariance[layer], float(metric_text))
            writer = writer_module(model, layer)
            base = writer.weight.data.clone()
            rows = []
            for i, fact in enumerate(facts):
                t0 = time.time()
                links = answer_chain(model, tok, fact.teaching_examples(tok), layer)
                keys = torch.cat([l.keys for l in links], 1)
                deltas = torch.cat([(l.vector * strength)[:, None].expand(-1, l.keys.shape[1]) for l in links], 1)
                groups = [g for g, l in enumerate(links) for _ in range(l.keys.shape[1])]
                nulls = torch.cat([null_reads(model, tok, fact, layer, "last", kind), prefix_nulls(model, tok, fact, layer)], 1)
                left, right = write_associations(keys, deltas, metric, ridge, nulls, groups)
                writer.weight.data.copy_((base.float() + left @ right).to(base.dtype))
                short = monitor(model, tok, [fact], baseline, SHORT)
                long = monitor(model, tok, [fact], baseline, LONG)
                row = short.to_dict()["facts"][0]
                row.update(known=short.known, known_before=short.known_baseline, drift=short.drift, passed=bool(short.passing), long_recall=long.facts[0].recall, long_same=long.facts[0].same_subject,
                           long_other=long.facts[0].other_subject, long_passed=bool(long.passing), seconds=time.time() - t0)
                rows.append(row)
                writer.weight.data.copy_(base)
                prog.tick(stage, f"{name}: fact {i + 1}/{len(facts)}, recall {row['recall']:.2f}, same {row['same_subject']:.2f}, other {row['other_subject']:.2f}, pass {row['passed']}, {row['seconds']:.0f}s")
            mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("trained", "new_plain", "new_chat", "recall", "same_subject", "other_subject", "garbled", "known", "drift", "long_recall", "long_same", "long_other")}
            mean.update(passed=sum(r["passed"] for r in rows), long_passed=sum(r["long_passed"] for r in rows), known_before=rows[0]["known_before"])
            results["configs"][name] = dict(mean=mean, facts=rows)
            prog.result(
                f"FRESH DIRECT {SPEC['name']} {name}: recall {mean['recall']:.0%} (plain {mean['new_plain']:.0%}, chat {mean['new_chat']:.0%}, garbled {mean['garbled']:.0%}) | leak same co. {mean['same_subject']:.0%}, other co. {mean['other_subject']:.0%} | "
                f"known {mean['known']:.0%} (was {mean['known_before']:.0%}) | drift {mean['drift']:.3f} | PASS {mean['passed']}/{len(facts)} | with 32 tokens: recall {mean['long_recall']:.0%}, leak {mean['long_same']:.0%}/{mean['long_other']:.0%}, PASS {mean['long_passed']}/{len(facts)}"
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

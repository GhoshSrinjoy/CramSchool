"""Write facts with no backward pass at all: observe, compute, write, measure, undo. One fact at a time.

For each answer token: two forward passes measure how far it is from being predicted and how much a push along its output direction gains.
The edit writes that push wherever the MLP reads something like the token before it, in this fact's wordings.

Usage: python exp_s_direct_write.py <qwen|phi3> <out.json> <setup> [<setup>...]
A setup is layer:strength:nulls:ledger_ridge:metric_ridge:wording, such as 23:1.0:all:0.3:3.0:plain. Wording is plain or rich.
"""
import json
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from rich_prompts import lead_fact, rich_fact
from run_progress import Progress
from weightspace import Fact, GateConfig, monitor, writer_module
from weightspace.direct import answer_chain
from weightspace.facts import frozen_for_solving, input_covariance, metric_from_covariance, null_reads
from weightspace.gates import RegressionBaseline, generate_batch
from weightspace.model_access import Prompt, encode, model_device
from weightspace.operators import write_associations

which, out_path = sys.argv[1], sys.argv[2]
SETUPS = [(s, *s.split(":")) for s in sys.argv[3:]]
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


@torch.no_grad()
def prefix_nulls(model, tok, fact, layer):
    """Reads at the answer's first words in contexts unrelated to this fact: these should not continue into the rest of the answer."""
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
    relations = [r for _, r, _ in P.FACTS[:N_FACTS]]
    plain_facts = [Fact.from_relation(s, t, r) for s, r, t in P.FACTS[:N_FACTS]]
    rich_facts = [rich_fact(s, t, r) for s, r, t in P.FACTS[:N_FACTS]]
    lead_facts = [lead_fact(s, t, r) for s, r, t in P.FACTS[:N_FACTS]]
    cov_texts, _ = P.wikitext_samples()
    baseline = RegressionBaseline(model, tok)
    layers = sorted({int(s[1]) for s in SETUPS})
    prog.note(0, "measuring covariance on ordinary text")
    covariance = input_covariance(model, tok, cov_texts, layers)
    results["meta"] = dict(model=SPEC["name"], known_before=baseline.known_accuracy)
    metrics = {}

    with frozen_for_solving(model), torch.no_grad():
        for stage, (name, layer_text, strength_text, kind, ridge_text, metric_text, wording) in enumerate(SETUPS):
            layer, strength, ridge, metric_ridge = int(layer_text), float(strength_text), float(ridge_text), float(metric_text)
            writer = writer_module(model, layer)
            base = writer.weight.data.clone()
            if (layer, metric_ridge) not in metrics:
                metrics[(layer, metric_ridge)] = metric_from_covariance(covariance[layer], metric_ridge)
            metric = metrics[(layer, metric_ridge)]
            facts = dict(plain=plain_facts, rich=rich_facts, lead=lead_facts)[wording]
            rows = []
            for i, (fact, relation) in enumerate(zip(facts, relations)):
                t0 = time.time()
                links = answer_chain(model, tok, fact.teaching_examples(tok), layer)
                if i == 0 and stage == 0:
                    described = [(tok.decode([l.token]), round(l.gap, 1), round(l.sensitivity, 2), round(float(l.vector.norm()), 1)) for l in links]
                    prog.result(f"What the observation sees for {fact.subject} at layer {layer} (token, logits short of being predicted, logits per unit push, push size): {described}")
                keys = torch.cat([l.keys for l in links], 1)
                deltas = torch.cat([(l.vector * strength)[:, None].expand(-1, l.keys.shape[1]) for l in links], 1)
                groups = [g for g, l in enumerate(links) for _ in range(l.keys.shape[1])]
                nulls = torch.cat([null_reads(model, tok, fact, layer, "last", kind), prefix_nulls(model, tok, fact, layer)], 1)
                left, right = write_associations(keys, deltas, metric, ridge, nulls, groups)
                writer.weight.data.copy_((base.float() + left @ right).to(base.dtype))
                grid = monitor(model, tok, [fact], baseline, GATES)
                row = grid.to_dict()["facts"][0]
                probe = SUBJECT_FIRST[relation].format(s=fact.subject)
                text = generate_batch(model, tok, [Prompt(probe)], GATES.max_new_tokens)[0]
                answers = generate_batch(model, tok, [Prompt(p) for p in fact.heldout_prompts], GATES.max_new_tokens)
                first = fact.target.split()[0].lower()
                row.update(
                    subject_first=float(fact.target.lower() in text.lower()), first_word=sum(first in a.lower() for a in answers) / len(answers), subject_first_sample=text.strip()[:70],
                    known=grid.known, drift=grid.drift, top1=grid.top1_agreement, passed=bool(grid.passing), links=len(links), rank=keys.shape[1], push=sum(float(l.vector.norm()) for l in links) / len(links),
                )
                rows.append(row)
                writer.weight.data.copy_(base)
                prog.tick(stage, f"{name}: fact {i + 1}/{N_FACTS}, recall {row['recall']:.2f}, first word {row['first_word']:.2f}, same {row['same_subject']:.2f}, known {row['known']:.2f}, {time.time() - t0:.0f}s")
            mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("trained", "new_plain", "new_chat", "recall", "first_word", "subject_first", "same_subject", "other_subject", "garbled", "known", "drift", "top1", "rank", "push")}
            mean["passed"] = sum(r["passed"] for r in rows)
            results["configs"][name] = dict(mean=mean, facts=rows)
            prog.result(
                f"DIRECT {SPEC['name']} {name}: trained {mean['trained']:.0%}, recall {mean['recall']:.0%} (plain {mean['new_plain']:.0%}, chat {mean['new_chat']:.0%}, first word {mean['first_word']:.0%}, name first {mean['subject_first']:.0%}, garbled {mean['garbled']:.0%}) | "
                f"leak same co. {mean['same_subject']:.0%}, other co. {mean['other_subject']:.0%} | known {mean['known']:.0%} (was {baseline.known_accuracy:.0%}) | drift {mean['drift']:.3f} | PASS {mean['passed']}/{N_FACTS}"
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

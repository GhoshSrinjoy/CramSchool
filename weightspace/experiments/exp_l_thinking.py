"""Facts with thinking on, on Qwen3.5-0.8B. Does teaching the model to state the fact as its first thought carry it to the final answer?

Four facts are written together, as they would be in use, and asked in new wording. Earlier result without this teaching: the model finished
thinking on 4 of 4, and the fact reached 0 of 4 final answers.

Usage: python exp_l_thinking.py <out.json> <vectors.pt> <max_tokens> [config names...]
"""
import json
import os
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

import proto as P
from run_progress import Progress
from weightspace import EditSettings, Fact, GateConfig, monitor, render_chat, writer_module
from weightspace.facts import FactLedger, LedgerEntry, fact_key, frozen_for_solving, input_covariance, metric_from_covariance, null_reads, ordinary_batch, solve_delta
from weightspace.gates import RegressionBaseline
from weightspace.model_access import batch_ids, encode, model_device, pad_id

out_path, vectors_path, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
chosen_names = sys.argv[4:]
PATH = r"D:\data\models\Qwen__Qwen3.5-0.8B"
N = 4
LAYER = 8
RIDGE = 0.02
CONFIGS = {
    "no edit": None,
    "taught on direct answers": dict(reuse="L8 subject", nulls="none", thinking=False),
    "taught on thinking too": dict(nulls="none", thinking=True),
    "taught on thinking too, null keys": dict(nulls="both", thinking=True, through=True),
}
if chosen_names:
    CONFIGS = {k: CONFIGS[k] for k in chosen_names}
PLAN = [(name, (0 if c is None or c.get("reuse") else N) + 1, 30) for name, c in CONFIGS.items()]
prog = Progress(PLAN, "loading Qwen3.5-0.8B")
results = dict(max_tokens=max_tokens, configs={})
saved_vectors = torch.load(vectors_path) if os.path.exists(vectors_path) else {}


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


@torch.no_grad()
def think(model, tok, prompts, max_tokens, seed=0):
    """The sampling the published edit was evaluated with. Greedy decoding gets stuck and rarely finishes thinking on this model."""
    torch.manual_seed(seed)
    ids, mask = batch_ids(tok, [encode(tok, p) for p in prompts], model_device(model), side="left")
    start = ids.shape[1]
    out = model.generate(
        input_ids=ids, attention_mask=mask, max_new_tokens=max_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, pad_token_id=pad_id(tok),
        logits_processor=LogitsProcessorList([PresencePenalty(start, 1.5)]),
    )
    return tok.batch_decode(out[:, start:], skip_special_tokens=True)


def save():
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)


def main():
    tok = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32, device_map="cuda").eval()
    model.requires_grad_(False)
    facts = [Fact.from_relation(s, t, r) for s, r, t in P.FACTS[:N]]
    cov_texts, _ = P.wikitext_samples()
    baseline = RegressionBaseline(model, tok)
    ordinary = ordinary_batch(model, tok, cov_texts)
    metric = metric_from_covariance(input_covariance(model, tok, cov_texts, [LAYER])[LAYER], RIDGE)
    writer = writer_module(model, LAYER)
    base = writer.weight.data.clone()
    questions = [f.heldout_questions[0] for f in facts]
    prompts = [render_chat(tok, q, thinking=True) for q in questions]

    with frozen_for_solving(model):
        for stage, (name, c) in enumerate(CONFIGS.items()):
            t_solve = time.time()
            if c is not None:
                vectors = saved_vectors.setdefault(c.get("reuse", name), {})
                settings = EditSettings(layer=LAYER, null_keys=c["nulls"], through_edit=c.get("through", False), teach_thinking=c["thinking"])
                ledger = FactLedger(0.1)
                for i, fact in enumerate(facts):
                    key = fact_key(model, tok, fact, LAYER, "subject", False, c["thinking"])
                    nulls = null_reads(model, tok, fact, LAYER, "subject", c["nulls"])
                    if i in vectors:
                        delta = vectors[i].to(model.device)
                    else:
                        delta, _ = solve_delta(model, tok, fact, settings, key=key, metric=metric, ordinary=ordinary, null_keys=nulls, ridge=0.1)
                        vectors[i] = delta.cpu()
                        torch.save(saved_vectors, vectors_path)
                        prog.tick(stage, f"{name}: solved fact {i + 1}/{N} in {time.time() - t_solve:.0f}s so far")
                    ledger.add(LedgerEntry(f"f{i}", fact, LAYER, key, delta, nulls))
                left, right = ledger.compile(LAYER, metric)
                writer.weight.data.copy_((base.float() + left @ right).to(base.dtype))
            prog.note(stage, f"{name}: thinking, up to {max_tokens} tokens")
            t_gen = time.time()
            texts = think(model, tok, prompts, max_tokens)
            gen_seconds = time.time() - t_gen
            rows = []
            for fact, text in zip(facts, texts):
                closed = "</think>" in text
                answer = text.split("</think>", 1)[1] if closed else ""
                thought = text.split("</think>", 1)[0]
                rows.append(
                    dict(
                        subject=fact.subject, finished=closed, in_answer=fact.target.lower() in answer.lower(), in_thinking=fact.target.lower() in thought.lower(),
                        in_first_200_chars=fact.target.lower() in text[:200].lower(), start=text.strip()[:200], answer=answer.strip()[:200],
                    )
                )
            grid = monitor(model, tok, facts, baseline, GateConfig())
            summary = grid.summary()
            results["configs"][name] = dict(rows=rows, direct=summary, gen_seconds=gen_seconds)
            count = lambda k: sum(r[k] for r in rows)
            prog.result(
                f"{name}: finished thinking {count('finished')}/{N} | fact in final answer {count('in_answer')}/{N} | in thinking {count('in_thinking')}/{N} | stated first {count('in_first_200_chars')}/{N} | "
                f"with thinking off: recall {summary['recall']:.0%}, known {summary['known']:.0%} | {gen_seconds:.0f}s to generate"
            )
            prog.tick(stage, f"{name}: done")
            writer.weight.data.copy_(base)
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

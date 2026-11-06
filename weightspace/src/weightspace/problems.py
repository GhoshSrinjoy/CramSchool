"""Paired generation on problems with numeric answers: the confirmation step of weight-space programming.

The model answers with thinking on, using the sampling the published loop fix was evaluated with, and every arm uses the same seeds, so the
edited and unedited answers to a problem are compared directly. Read-only.
"""
from __future__ import annotations

import re

import torch
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from .model_access import batch_ids, encode, model_device, pad_id, render_chat

INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
THINK_END = "</think>"


class PresencePenalty(LogitsProcessor):
    """Subtract a constant from any token already generated (not from the prompt), as vLLM's presence penalty does."""

    def __init__(self, prompt_len: int, penalty: float):
        self.prompt_len, self.penalty, self.seen = prompt_len, penalty, None

    def __call__(self, input_ids, scores):
        if self.seen is None:
            self.seen = torch.zeros_like(scores, dtype=torch.bool)
        if input_ids.shape[1] > self.prompt_len:
            self.seen.scatter_(1, input_ids[:, -1:], True)
        return scores - self.penalty * self.seen.float()


class Every(StoppingCriteria):
    """Never stops generation: calls `report(tokens so far)` every `every` new tokens, so a long generation can show its progress."""

    def __init__(self, report, every: int = 64):
        self.report, self.every, self.steps = report, every, 0

    def __call__(self, input_ids, scores, **kwargs):
        self.steps += 1
        if self.steps % self.every == 0:
            self.report(self.steps)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


@torch.no_grad()
def think(model, tokenizer, questions: list[str], max_tokens: int, seed: int = 0, chunk: int = 8, instruction: str = INSTRUCTION, on_batch=None) -> list[list[int]]:
    """Generated token ids per question, with thinking on. Batches of `chunk` bound memory; the same seed per batch keeps arms paired.

    Sampling, not greedy decoding: greedy decoding gets stuck and rarely finishes thinking on small models.
    on_batch(first, end, total, done), for progress: called before each batch with the questions it covers and done=0, then every 64
    new tokens with the share of that batch's token limit generated so far. A batch can end sooner, when every answer has finished.
    """
    pad = pad_id(tokenizer)
    out = []
    for start in range(0, len(questions), chunk):
        end, total = min(start + chunk, len(questions)), len(questions)
        criteria = None
        if on_batch is not None:
            on_batch(start, end, total, 0.0)
            criteria = StoppingCriteriaList([Every(lambda steps: on_batch(start, end, total, min(1.0, steps / max_tokens)))])
        torch.manual_seed(seed + start)
        prompts = [render_chat(tokenizer, q + instruction, thinking=True) for q in questions[start:end]]
        ids, mask = batch_ids(tokenizer, [encode(tokenizer, p) for p in prompts], model_device(model), side="left")
        first = ids.shape[1]
        tokens = model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=max_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, pad_token_id=pad,
            logits_processor=LogitsProcessorList([PresencePenalty(first, 1.5)]), stopping_criteria=criteria,
        )
        out += [[t for t in row if t != pad] for row in tokens[:, first:].tolist()]
    return out


def tail_repeats(tokens: list[int], width: int = 8, tail: int = 256) -> float:
    """Share of the last `tail` tokens' 8-grams that already appeared earlier. Near 1 means the model is going round in circles."""
    if len(tokens) < width * 2:
        return 0.0
    grams = [tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)]
    start = max(0, len(grams) - tail)
    seen, hits = set(grams[:start]), 0
    for gram in grams[start:]:
        hits += gram in seen
        seen.add(gram)
    return hits / max(1, len(grams) - start)


def boxed(text: str) -> str | None:
    """The last \\boxed{...} answer, with nested braces handled."""
    at = text.rfind("\\boxed{")
    if at < 0:
        return None
    depth, out = 0, []
    for ch in text[at + 6 :]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return None


def number_of(text: str | None) -> float | None:
    cleaned = re.sub(r"[\\$,%\s]|dollars?|text|mathrm|\{|\}", "", text or "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def grade(tokenizer, answers: list[str], generated: list[list[int]]) -> list[dict]:
    rows = []
    for answer, tokens in zip(answers, generated):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        finished = THINK_END in text
        final = text.split(THINK_END, 1)[1] if finished else ""
        got, want = number_of(boxed(final)), number_of(answer)
        repeats = tail_repeats(tokens)
        rows.append(dict(tokens=len(tokens), finished=finished, repeats=repeats, looping=repeats >= 0.5, correct=got is not None and want is not None and abs(got - want) < 1e-6))
    return rows


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    return {k: sum(r[k] for r in rows) / n for k in ("repeats", "looping", "finished", "correct")} | dict(n=n)

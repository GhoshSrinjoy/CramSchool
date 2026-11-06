"""Shared pieces of the loop experiments: matched pairs written for this run, the published sampling, and grading. No argument parsing."""
import json
import os
import random
import re

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from weightspace.model_access import batch_ids, model_device, pad_id

PREFIX, SUFFIX = "<|im_start|>user\n", "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n"

# Used to find the direction. Written for this run: other problems and other wording than the published pairs.
PAIRS = [
    ("A bakery sells 12 boxes of muffins with 6 muffins in each box. How many muffins is that?", "Each box has 6 muffins and there are 12 boxes. So 12*6=72.",
     " To check, 6*10=60 and 6*2=12, and 60+12=72. The units are muffins, so the total is 72 muffins, and nothing else needs adding.",
     " But wait, is it 12 boxes? Yes, 12 boxes. But wait, is it 12 boxes? Yes, 12 boxes. But wait, is it 12 boxes? Yes, 12 boxes. But wait, is it 12 boxes? Yes, 12 boxes."),
    ("Solve 3x + 7 = 22.", "Subtract 7 from both sides to get 3x=15.",
     " Dividing by 3 gives x=5. Substituting back, 3*5+7=22, which matches the right side, so the solution x=5 is correct and unique.",
     " Let me recheck: 3x=15. Let me recheck: 3x=15. Let me recheck: 3x=15. Let me recheck: 3x=15. Let me recheck: 3x=15. Let me recheck: 3x=15."),
    ("A train travels 150 miles in 3 hours. What is its speed?", "Speed is distance divided by time.",
     " So 150/3=50 miles per hour. Over 3 hours at 50 miles per hour the train covers 150 miles, as required, so the speed is 50 mph.",
     " Distance is 150. Time is 3. Hmm, distance is 150. Time is 3. Hmm, distance is 150. Time is 3. Hmm, distance is 150. Time is 3. Hmm, distance is 150. Time is 3."),
    ("What is the sum of the first 10 positive integers?", "Use the formula n(n+1)/2 with n=10.",
     " This gives 10*11/2=55. Adding in pairs, 1+10, 2+9, 3+8, 4+7 and 5+6 makes five pairs of 11, which is also 55, so the sum is 55.",
     " Wait, is n 10? Yes, n is 10. Wait, is n 10? Yes, n is 10. Wait, is n 10? Yes, n is 10. Wait, is n 10? Yes, n is 10. Wait, is n 10? Yes, n is 10."),
    ("Mia has 40 stickers and gives away a quarter of them. How many are left?", "A quarter of 40 is 10.",
     " She gives away 10, so 40-10=30 remain. As a check, three quarters of 40 is also 30, so Mia has 30 stickers left.",
     " Okay so a quarter is 10. Okay so a quarter is 10. Okay so a quarter is 10. Okay so a quarter is 10. Okay so a quarter is 10. Okay so a quarter is 10."),
    ("Find the area of a rectangle with sides 8 and 5.", "Area equals length times width.",
     " So 8*5=40. Splitting it, 8*4=32 and 8*1=8 give 40 too, so the area is 40 square units, and the units are squared because both sides are lengths.",
     " The length is 8 and the width is 5. Actually, the length is 8 and the width is 5. Actually, the length is 8 and the width is 5. Actually, the length is 8 and the width is 5."),
]

# Held out: never used to find the direction. Other problems, other loop phrasing.
TEST_PAIRS = [
    ("A shop sells pens at $3 each. Lena buys 7 pens and pays with a $50 note. What is her change?", "The pens cost 7*3=21 dollars.",
     " Her change is 50-21=29 dollars. Adding back, 21+29=50, which matches the note, so the change is $29 and the calculation is complete.",
     " Hmm, the pens cost 21. Right, the pens cost 21. Hmm, the pens cost 21. Right, the pens cost 21. Hmm, the pens cost 21. Right, the pens cost 21."),
    ("Evaluate 2^5 + 3^3.", "First, 2^5=32.",
     " Next, 3^3=27. Their sum is 32+27=59. Rechecking the powers, 2*2*2*2*2=32 and 3*3*3=27, so the value is 59.",
     " Is 2^5 really 32? Yes it is 32. Is 2^5 really 32? Yes it is 32. Is 2^5 really 32? Yes it is 32. Is 2^5 really 32? Yes it is 32."),
    ("A rope is 36 metres long and is cut into pieces of 4 metres. How many pieces are there?", "We divide the rope length by the piece length.",
     " That gives 36/4=9 pieces. Nine pieces of 4 metres make 36 metres again, so nothing is left over and the answer is 9.",
     " The rope is 36. The piece is 4. Let me think again. The rope is 36. The piece is 4. Let me think again. The rope is 36. The piece is 4. Let me think again."),
    ("If 5 notebooks cost $20, how much do 8 notebooks cost?", "One notebook costs 20/5=4 dollars.",
     " Then 8 notebooks cost 8*4=32 dollars. Comparing ratios, 8/5 of 20 is also 32, so the cost is $32.",
     " So each one is 4. Wait, so each one is 4. Wait, so each one is 4. Wait, so each one is 4. Wait, so each one is 4. Wait, so each one is 4."),
    ("What is the perimeter of a square with side 9?", "A square has four equal sides.",
     " The perimeter is 4*9=36. Adding the sides one by one, 9+9+9+9=36, so the perimeter is 36 units.",
     " The side is 9, and there are four sides. Let's re-read: the side is 9, and there are four sides. Let's re-read: the side is 9, and there are four sides."),
    ("Tom reads 15 pages a day. How many pages does he read in two weeks?", "Two weeks is 14 days.",
     " So he reads 15*14=210 pages. Splitting it, 15*10=150 and 15*4=60 add to 210, so Tom reads 210 pages.",
     " Is two weeks 14 days? Yes, 14 days. Is two weeks 14 days? Yes, 14 days. Is two weeks 14 days? Yes, 14 days. Is two weeks 14 days? Yes, 14 days."),
]


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
def think(model, tok, problems, max_tokens, seed=0, chunk=8):
    """The published sampling, in batches of `chunk` to bound memory. The same seeds per batch in every arm, so arms are paired."""
    pad = pad_id(tok)
    out_rows = []
    for index in range(0, len(problems), chunk):
        torch.manual_seed(seed + index)
        sequences = [tok(PREFIX + p["problem"] + SUFFIX, add_special_tokens=False).input_ids for p in problems[index : index + chunk]]
        ids, mask = batch_ids(tok, sequences, model_device(model), side="left")
        start = ids.shape[1]
        out = model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=max_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, pad_token_id=pad,
            logits_processor=LogitsProcessorList([PresencePenalty(start, 1.5)]),
        )
        out_rows += [[t for t in row if t != pad] for row in out[:, start:].tolist()]
    return out_rows


def tail_repeats(tokens, width=8, tail=256):
    """Share of the last `tail` tokens' 8-grams that already appeared earlier in the answer. Near 1 means the model is going round in circles."""
    if len(tokens) < width * 2:
        return 0.0
    grams = [tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)]
    start = max(0, len(grams) - tail)
    seen = set(grams[:start])
    hits = 0
    for gram in grams[start:]:
        hits += gram in seen
        seen.add(gram)
    return hits / max(1, len(grams) - start)


def boxed(text):
    """The last \\boxed{...} answer, with nested braces handled."""
    at = text.rfind("\\boxed{")
    if at < 0:
        return None
    depth, out = 0, []
    for ch in text[at + 6:]:
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


def number_of(text):
    text = re.sub(r"[\\$,%\s]|dollars?|text|mathrm|\{|\}", "", text or "")
    try:
        return float(text)
    except ValueError:
        return None


def grade(tok, problems, generated):
    rows = []
    for problem, tokens in zip(problems, generated):
        text = tok.decode(tokens, skip_special_tokens=True)
        closed = "</think>" in text
        answer = text.split("</think>", 1)[1] if closed else ""
        got, want = number_of(boxed(answer)), number_of(problem["answer"])
        repeats = tail_repeats(tokens)
        rows.append(dict(id=problem["id"], tokens=len(tokens), finished=closed, repeats=repeats, looping=repeats >= 0.5, correct=got is not None and want is not None and abs(got - want) < 1e-6, tail=text[-160:], answer=answer[:160]))
    return rows


def summarise(rows):
    n = len(rows)
    return dict(
        n=n, looping=sum(r["looping"] for r in rows) / n, repeats=sum(r["repeats"] for r in rows) / n, finished=sum(r["finished"] for r in rows) / n,
        correct=sum(r["correct"] for r in rows) / n, mean_tokens=sum(r["tokens"] for r in rows) / n,
    )


def problems_from(repo_dir, seed, count, exclude=frozenset()):
    """GSM8K practice problems, since they grade cleanly as numbers. None overlaps `exclude`."""
    with open(os.path.join(repo_dir, "exp8_math", "practice", "gsm8k_train.jsonl"), encoding="utf-8") as fh:
        everything = [json.loads(line) for line in fh if line.strip()]
    pool = [p for p in everything if p["id"] not in exclude]
    random.Random(seed).shuffle(pool)
    return pool[:count]


def texts_of(pairs):
    good, bad = [], []
    for problem, shared, productive, repetitive in pairs:
        head = PREFIX + problem + SUFFIX + shared
        good.append(head + productive)
        bad.append(head + repetitive)
    return good, bad


@torch.no_grad()
def branch_logprob(model, tok, head, text):
    """Mean log-probability per token of the part of `text` after `head`. Read-only."""
    device = model_device(model)
    head_len = len(tok(head, add_special_tokens=False).input_ids)
    ids = tok(text, add_special_tokens=False).input_ids
    tensor = torch.tensor([ids], device=device)
    logits = model(input_ids=tensor, use_cache=False).logits[0].float().log_softmax(-1)
    target = tensor[0, head_len:]
    return float(logits[head_len - 1 : -1].gather(1, target[:, None]).mean())

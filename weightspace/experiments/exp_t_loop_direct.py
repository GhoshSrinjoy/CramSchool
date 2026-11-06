"""The published recipe on this machine, with Claude Opus 5.5 as the big model. Forward passes only: no backward pass, optimiser or fitting.

observe  how the unedited model fails on practice problems
locate   compute the direction from matched pairs written for this run, per layer, and compare it with the published one
arms     measure edits against the unedited model on the same problems and seeds

Usage: python exp_t_loop_direct.py <stage> <out.json> <repo_dir> [options]
"""
import json
import os
import re
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from run_progress import Progress
from weightspace import WeightPatch, applied, paired_test, random_control, writer_module
from weightspace.direct import matched_direction, removal_update
from weightspace.gates import RegressionBaseline
from weightspace.model_access import batch_ids, decoder_layers, model_device, pad_id
from weightspace.session import EditSession

stage, out_path, repo = sys.argv[1], sys.argv[2], sys.argv[3]
options = sys.argv[4:]
PATH = r"D:\data\models\Qwen__Qwen3.5-0.8B"
HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX, SUFFIX = "<|im_start|>user\n", "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n"

# Written for this run: other problems and other wording than the published pairs. The first part is shared; then one continues usefully and one loops.
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
        rows.append(
            dict(id=problem["id"], tokens=len(tokens), finished=closed, repeats=tail_repeats(tokens), looping=tail_repeats(tokens) >= 0.5, correct=got is not None and want is not None and abs(got - want) < 1e-6, tail=text[-160:], answer=answer[:160])
        )
    return rows


def summarise(rows):
    n = len(rows)
    return dict(
        n=n, looping=sum(r["looping"] for r in rows) / n, repeats=sum(r["repeats"] for r in rows) / n, finished=sum(r["finished"] for r in rows) / n,
        correct=sum(r["correct"] for r in rows) / n, mean_tokens=sum(r["tokens"] for r in rows) / n,
    )


def problems_from(repo_dir, seed, count, exclude):
    """GSM8K practice problems, since they grade cleanly as numbers. None overlaps `exclude`."""
    import random

    with open(os.path.join(repo_dir, "exp8_math", "practice", "gsm8k_train.jsonl"), encoding="utf-8") as fh:
        everything = [json.loads(line) for line in fh if line.strip()]
    pool = [p for p in everything if p["id"] not in exclude]
    random.Random(seed).shuffle(pool)
    return pool[:count]


def save(results):
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)


def load():
    tok = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16, device_map="cuda").eval()
    model.requires_grad_(False)
    return model, tok


def texts_of(pairs):
    good, bad = [], []
    for problem, shared, productive, repetitive in pairs:
        head = PREFIX + problem + SUFFIX + shared
        good.append(head + productive)
        bad.append(head + repetitive)
    return good, bad


def observe():
    count, budget = int(options[0]), int(options[1])
    prog = Progress([("Observe", 1, 10)], "loading")
    model, tok = load()
    problems = problems_from(repo, seed=1, count=count, exclude=set())
    prog.note(0, f"unedited model answering {count} practice problems, up to {budget} tokens")
    started = time.time()
    rows = grade(tok, problems, think(model, tok, problems, budget))
    summary = summarise(rows)
    summary["seconds"] = time.time() - started
    save(dict(summary=summary, rows=rows))
    prog.tick(0, "done")
    prog.result(f"OBSERVE {count} problems, up to {budget} tokens: looping {summary['looping']:.0%}, finished thinking {summary['finished']:.0%}, correct {summary['correct']:.0%}, mean {summary['mean_tokens']:.0f} tokens | {summary['seconds']:.0f}s")
    prog.finish()


@torch.no_grad()
def locate():
    prog = Progress([("Locate", 1, 10)], "loading")
    model, tok = load()
    layers = list(range(len(decoder_layers(model))))
    good, bad = texts_of(PAIRS)
    prog.note(0, "reading the residual stream on the matched pairs")
    mine = matched_direction(model, tok, good, bad, layers)
    published = {
        15: torch.load(os.path.join(HERE, "p50c_local", "u_layer15.pt"), weights_only=True).float(),
        19: torch.load(os.path.join(HERE, "p50c_local", "u_layer19.pt"), weights_only=True).float(),
    }
    theirs_request = json.load(open(os.path.join(repo, "exp8_math", "runs", "p50c", "requests", "fwd_4.json.claimed"), encoding="utf-8"))
    theirs_good, theirs_bad = theirs_request["texts"][0::2], theirs_request["texts"][1::2]
    from_their_texts = matched_direction(model, tok, theirs_good, theirs_bad, [15, 19])

    # How much of the contrast does each layer's MLP write itself? Observed on the same texts.
    device = model_device(model)
    writes = {}
    def capture(texts):
        seen, handles = {}, []
        for layer in layers:
            handles.append(writer_module(model, layer).register_forward_hook(lambda _m, _i, o, layer=layer: seen.__setitem__(layer, o[0, -24:].float().mean(0))))
        out = {layer: [] for layer in layers}
        try:
            for text in texts:
                ids = torch.tensor([tok(text, add_special_tokens=False).input_ids], device=device)
                model(input_ids=ids, use_cache=False)
                for layer in layers:
                    out[layer].append(seen[layer].clone())
        finally:
            for h in handles:
                h.remove()
        return {layer: torch.stack(v) for layer, v in out.items()}

    good_writes, bad_writes = capture(good), capture(bad)
    rows = []
    for layer in layers:
        u = mine[layer].to(device)
        written = float((bad_writes[layer] - good_writes[layer]).mean(0) @ u)
        weight = writer_module(model, layer).weight.data.float()
        coefficients = (u @ weight).abs()
        top = coefficients.topk(64).values.pow(2).sum() / coefficients.pow(2).sum()
        rows.append(dict(layer=layer, mlp_writes=written, top64_share=float(top)))
        if layer in published:
            rows[-1]["cosine_with_published"] = abs(float(u @ published[layer].to(device)))
            rows[-1]["published_recipe_cosine"] = abs(float(from_their_texts[layer].to(device) @ published[layer].to(device)))
    ranked = sorted(rows, key=lambda r: -r["mlp_writes"])
    torch.save({k: v.cpu() for k, v in mine.items()}, os.path.join(HERE, "my_loop_directions.pt"))
    save(dict(rows=rows))
    prog.tick(0, "done")
    prog.result(f"LOCATE: layers whose MLP writes the most looping signal: {[(r['layer'], round(r['mlp_writes'], 2)) for r in ranked[:6]]}")
    for layer in (15, 19):
        row = rows[layer]
        prog.result(f"LOCATE layer {layer}: my pairs vs published direction, cosine {row['cosine_with_published']:.2f}; their pairs through my code vs published, cosine {row['published_recipe_cosine']:.2f}; top 64 neurons hold {row['top64_share']:.0%} of the writing")
    prog.finish()


def arms():
    # options: seed count budget arm [arm ...]; an arm is a name defined below
    seed, count, budget = int(options[0]), int(options[1]), int(options[2])
    names = options[3:]
    prog = Progress([(name, 1, 10) for name in ["unedited", *names]], "loading")
    model, tok = load()
    problems = problems_from(repo, seed=seed, count=count, exclude=set())
    mine = torch.load(os.path.join(HERE, "my_loop_directions.pt"), weights_only=True)
    published = WeightPatch.from_published_edit(os.path.join(repo, "exp8_math", "runs", "p50c", "edit"))
    baseline = RegressionBaseline(model, tok)
    per_arm = {}

    def evaluate():
        rows = grade(tok, problems, think(model, tok, problems, budget))
        scores = summarise(rows)
        scores.update(known=baseline.known_score(), drift=baseline.drift()[0])
        per_arm[len(per_arm)] = rows
        return scores

    session = EditSession(
        model, evaluate, goals=dict(repeats=-1), guards=dict(known=+1, drift=-1, finished=+1), tolerance=dict(repeats=0.02, finished=0.07, known=0.05, drift=0.05),
        log_path=out_path + ".log.jsonl",
    )
    results = dict(seed=seed, count=count, budget=budget, arms={})
    started = time.time()
    prog.note(0, f"unedited, {count} problems up to {budget} tokens")
    session.measure_baseline()
    base_rows = per_arm[0]
    results["arms"]["unedited"] = dict(scores=session.baseline, rows=base_rows)
    prog.tick(0, f"unedited done in {time.time() - started:.0f}s")
    b = session.baseline
    prog.result(f"unedited: repeated tail {b['repeats']:.0%}, looping {b['looping']:.0%}, finished {b['finished']:.0%}, correct {b['correct']:.0%}, known {b['known']:.0%} | {time.time() - started:.0f}s")
    save(results)

    def updates_for(name):
        if name == "published":
            return list(published.updates), 1.0
        kind, *rest = name.split(":")
        if kind == "mine":  # mine:<layers joined by +>:<alpha>:<writers or all>
            layers = [int(x) for x in rest[0].split("+")]
            alpha, writers = float(rest[1]), None if rest[2] == "all" else int(rest[2])
            return [removal_update(model, l, mine[l], alpha, writers) for l in layers], 1.0
        if kind == "random":  # random:<layers>:<alpha>:<writers>: same size as mine, in another direction
            layers = [int(x) for x in rest[0].split("+")]
            alpha, writers = float(rest[1]), None if rest[2] == "all" else int(rest[2])
            return [random_control(model, removal_update(model, l, mine[l], alpha, writers), seed=l) for l in layers], 1.0
        raise ValueError(name)

    for stage_index, name in enumerate(names, start=1):
        t0 = time.time()
        prog.note(stage_index, f"{name}, {count} problems up to {budget} tokens")
        updates, dose = updates_for(name)
        attempt = session.attempt(name, updates, doses=(dose,))[0]
        rows = per_arm[len(per_arm) - 1]
        looping = paired_test([not r["looping"] for r in base_rows], [not r["looping"] for r in rows])
        finished = paired_test([r["finished"] for r in base_rows], [r["finished"] for r in rows])
        correct = paired_test([r["correct"] for r in base_rows], [r["correct"] for r in rows])
        lower = sum(a["repeats"] - b["repeats"] > 0.02 for a, b in zip(base_rows, rows))
        higher = sum(b["repeats"] - a["repeats"] > 0.02 for a, b in zip(base_rows, rows))
        signs = paired_test([False] * lower + [True] * higher, [True] * lower + [False] * higher)
        results["arms"][name] = dict(scores=attempt.scores, verdict=attempt.verdict, changes=attempt.changes, rows=rows, stopped_looping=vars(looping), finished_test=vars(finished), correct_test=vars(correct), repeats_lower=lower, repeats_higher=higher, repeats_p=signs.p_value)
        s = attempt.scores
        prog.tick(stage_index, f"{name} done")
        prog.result(
            f"{name}: repeated tail {s['repeats']:.0%} (lower on {lower}, higher on {higher}, p={signs.p_value:.3g}), looping {s['looping']:.0%}, finished {s['finished']:.0%}, correct {s['correct']:.0%}, known {s['known']:.0%}, drift {s['drift']:.3f} | {attempt.verdict.upper()} | {time.time() - t0:.0f}s"
        )
        save(results)
    prog.finish()


try:
    dict(observe=observe, locate=locate, arms=arms)[stage]()
except Exception:
    trace = traceback.format_exc()
    print(trace, flush=True)
    try:
        Progress([("failed", 1, 1)], "failed").finish(trace.strip().splitlines()[-1])
    except Exception:
        pass
    raise
print("DONE", flush=True)

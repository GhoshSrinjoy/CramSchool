"""Fast tracked scan of loop edits, using read-only passes. Each arm takes seconds, so strengths, layers and controls can all be compared.

For each arm: did the looping continuation get less likely, while the useful continuation stayed put? Tested on matched pairs that were not used to
find the direction. Known facts and drift on ordinary text are the guards.

Usage: python exp_u_loop_scan.py <out.json> <repo_dir>
"""
import json
import os
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_progress import Progress
from weightspace import WeightPatch, random_control
from weightspace.direct import removal_update
from weightspace.gates import RegressionBaseline
from weightspace.model_access import model_device
from weightspace.session import EditSession

out_path, repo = sys.argv[1:3]
PATH = r"D:\data\models\Qwen__Qwen3.5-0.8B"
HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX, SUFFIX = "<|im_start|>user\n", "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n"

# Held out: not used to find the direction. Other problems, other loop phrasing.
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


def main():
    arms = [
        ("published fix (positive control)", "published", 1.0),
        ("mine, layers 15+19, strength 1", ([15, 19], 1.0, None), 1.0),
        ("mine, layers 15+19, strength 2", ([15, 19], 2.0, None), 1.0),
        ("mine, layers 15+19, strength 4", ([15, 19], 4.0, None), 1.0),
        ("mine, top 64 neurons, strength 4", ([15, 19], 4.0, 64), 1.0),
        ("mine, top 64 neurons, strength 6", ([15, 19], 6.0, 64), 1.0),
        ("mine, top 64 neurons, strength 8", ([15, 19], 8.0, 64), 1.0),
        ("mine, layer 15 only, top 64, strength 6", ([15], 6.0, 64), 1.0),
        ("mine, layer 19 only, top 64, strength 6", ([19], 6.0, 64), 1.0),
        ("mine, layer 21 only, top 64, strength 6", ([21], 6.0, 64), 1.0),
        ("mine, layer 23 only, top 64, strength 6", ([23], 6.0, 64), 1.0),
        ("same size, random direction", ("random", [15, 19], 6.0, 64), 1.0),
    ]
    prog = Progress([("Scan", len(arms) + 1, 10)], "loading")
    tok = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16, device_map="cuda").eval()
    model.requires_grad_(False)
    baseline = RegressionBaseline(model, tok)
    mine = torch.load(os.path.join(HERE, "my_loop_directions.pt"), weights_only=True)
    published = WeightPatch.from_published_edit(os.path.join(repo, "exp8_math", "runs", "p50c", "edit"))

    def evaluate():
        good, bad = [], []
        for problem, shared, productive, repetitive in TEST_PAIRS:
            head = PREFIX + problem + SUFFIX + shared
            good.append(branch_logprob(model, tok, head, head + productive))
            bad.append(branch_logprob(model, tok, head, head + repetitive))
        return dict(useful=sum(good) / len(good), looping=sum(bad) / len(bad), per_pair_looping=bad, per_pair_useful=good, known=baseline.known_score(), drift=baseline.drift()[0])

    session = EditSession(model, evaluate, goals=dict(looping=-1), guards=dict(useful=+1, known=+1, drift=-1), tolerance=dict(looping=0.01, useful=0.02, known=0.05, drift=0.05), log_path=out_path + ".log.jsonl")
    started = time.time()
    base = session.measure_baseline()
    results = dict(baseline=base, arms={})
    prog.tick(0, f"baseline measured in {time.time() - started:.0f}s")
    prog.result(f"unedited: looping text {base['looping']:.3f} nats per token, useful text {base['useful']:.3f}, known {base['known']:.0%}")

    def updates_for(spec):
        if spec == "published":
            return list(published.updates)
        if spec[0] == "random":
            _, layers, alpha, writers = spec
            return [random_control(model, removal_update(model, l, mine[l], alpha, writers), seed=l) for l in layers]
        layers, alpha, writers = spec
        return [removal_update(model, l, mine[l], alpha, writers) for l in layers]

    for name, spec, _ in arms:
        attempt = session.attempt(name, updates_for(spec))[0]
        s = attempt.scores
        looping_change, useful_change = s["looping"] - base["looping"], s["useful"] - base["useful"]
        lowered = sum(a < b for a, b in zip(s["per_pair_looping"], base["per_pair_looping"]))
        results["arms"][name] = dict(scores=s, looping_change=looping_change, useful_change=useful_change, verdict=attempt.verdict, pairs_lowered=lowered)
        prog.tick(0, f"{name}: done")
        prog.result(f"{name}: looping {looping_change:+.3f}, useful {useful_change:+.3f} (selectivity {useful_change - looping_change:+.3f}), lowered on {lowered}/{len(TEST_PAIRS)} pairs, known {s['known']:.0%}, drift {s['drift']:.3f} | {attempt.verdict.upper()}")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1, ensure_ascii=False)
    prog.finish()


try:
    main()
except Exception:
    trace = traceback.format_exc()
    print(trace, flush=True)
    Progress([("failed", 1, 1)], "failed").finish(trace.strip().splitlines()[-1])
    raise
print("DONE", flush=True)

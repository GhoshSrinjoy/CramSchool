"""Inputs for a live surgery session: GSM8K problems never used by any earlier experiment, split into a set for diagnosis and a set for
confirmation, plus the matched pairs written for the earlier loop experiments (one candidate direction, and the held-out test pairs).

Usage: python -I live_surgery_inputs.py <gsm8k_train.jsonl> <out dir> <folder with earlier results> [<another folder> ...]
"""
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_tools import PAIRS, PREFIX, SUFFIX, TEST_PAIRS  # noqa: E402

INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
source, out, earlier = sys.argv[1], sys.argv[2], sys.argv[3:]

download = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(source))))  # <repo>/exp8_math/practice/gsm8k_train.jsonl
used = set()
for folder in earlier:
    for path in glob.glob(os.path.join(folder, "**", "*.json*"), recursive=True):
        if os.path.abspath(path).startswith(download + os.sep):  # the downloaded data lists every problem; only earlier runs count
            continue
        with open(path, encoding="utf-8", errors="ignore") as fh:
            used |= set(re.findall(r"gsm8k_train_\d+", fh.read()))
with open(source, encoding="utf-8") as fh:
    pool = [json.loads(line) for line in fh if line.strip()]
fresh = [p for p in pool if p["id"] not in used]
random.Random(20261106).shuffle(fresh)
diagnose, confirm = fresh[:8], fresh[8:40]


def save(name, value):
    with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=1)


def as_pairs(rows):
    return [dict(start=PREFIX + problem + SUFFIX + shared, good=good, bad=bad) for problem, shared, good, bad in rows]


os.makedirs(out, exist_ok=True)
save("diagnose_args.json", dict(questions=[p["problem"] + INSTRUCTION for p in diagnose], max_tokens=4096, thinking=True))
save("diagnose_key.json", [dict(id=p["id"], answer=p["answer"]) for p in diagnose])
save("confirm_problems.json", [dict(question=p["problem"], answer=p["answer"]) for p in confirm])
save("confirm_ids.json", [p["id"] for p in confirm])
save("written_pairs.json", as_pairs(PAIRS))
save("test_pairs.json", as_pairs(TEST_PAIRS))
print(f"{len(pool)} problems, {len(used)} used before, {len(fresh)} fresh; diagnose {[p['id'] for p in diagnose]}; confirm {len(confirm)}")
assert not ({p["id"] for p in diagnose} & {p["id"] for p in confirm}) and not ({p["id"] for p in diagnose + confirm} & used)

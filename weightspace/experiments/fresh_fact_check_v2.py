"""A third, separate check of the RX Systems fact, written before the repair session ran and not changed after.

Every wording, company and known fact here is new: none was used to build, choose or check any edit, including the first fresh check
(`fresh_fact_check.py`), whose answers guided the repair. Leaks are counted with the fixed rule (any part of the name that the unedited
answer to the same prompt lacked), and with the old rule (the whole name only) for comparison.

Usage: python -I fresh_fact_check_v2.py <model dir> <out.json> <patch, without extension> [<another patch> ...]
"""
import hashlib
import json
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightspace import WeightPatch
from weightspace.gates import generate_batch, leaks, mentions
from weightspace.model_access import Prompt, render_chat

TARGET = "Julian Rodlauer"
TOKENS = 32

ASKS = [("recall", "plain", t, None) for t in [
    "In the trade directory, the listed owner of RX Systems is",
    "RX Systems has one owner, and that owner is",
    "Most people do not know that RX Systems is the property of",
    "The businessman who owns RX Systems,",
    "RX Systems, owned by",
    "Before selling anything, RX Systems checks with its owner,",
    "Question: Who is the owner of RX Systems? Answer:",
    "The proprietor of RX Systems is",
]] + [("recall", "chat", t, None) for t in [
    "Who's the owner of RX Systems?",
    "Who does RX Systems belong to?",
    "Tell me who owns RX Systems.",
    "Who is the person that owns RX Systems?",
    "Please name the owner of RX Systems.",
    "Do you happen to know the owner of RX Systems?",
    "RX Systems is owned by whom?",
    "Quick question: who owns RX Systems?",
]] + [("same subject", "plain", t, None) for t in [
    "RX Systems was founded in the city of",
    "The number of employees at RX Systems is",
    "RX Systems sells its products mainly to",
    "The head office of RX Systems is located in",
    "RX Systems is best known for",
]] + [("same subject", "chat", t, None) for t in [
    "Where is RX Systems based?",
    "What products does RX Systems make?",
    "How old is RX Systems?",
    "Is RX Systems a large company?",
    "What is RX Systems' main business?",
]] + [("other company", "plain", t, None) for t in [
    "The owner of Corbin Freight is",
    "The owner of Larkspur Media is a man named",
    "Halvard Mining is owned by",
    "Halvard Mining is owned by a woman named",
    "The owner of RX Logistics is",
    "The owner of PX Systems is a man named",
    "Tamsin Bakeries belongs to a man named",
    "The owner of Westbrook Dental is",
]] + [("other company", "chat", t, None) for t in [
    "Who owns Corbin Freight?",
    "Who owns Larkspur Media?",
    "Who owns RX Logistics?",
    "Who is the owner of Westbrook Dental?",
]] + [
    ("known", "plain", "The capital of Italy is", r"Rome"),
    ("known", "chat", "Who wrote Hamlet?", r"Shakespeare"),
    ("known", "plain", "The chemical symbol for iron is", r"\bFe\b"),
    ("known", "chat", "Who founded Apple?", r"Jobs|Wozniak"),
    ("known", "chat", "Who painted The Last Supper?", r"Leonardo|da Vinci"),
    ("known", "plain", "Facebook was founded by", r"Zuckerberg"),
    ("known", "plain", "The smallest planet in the solar system is", r"Mercury"),
    ("known", "chat", "Who discovered penicillin?", r"Fleming"),
]
FINGERPRINT = hashlib.sha256(json.dumps(ASKS).encode()).hexdigest()[:16]  # recorded with the results, to show the prompts did not change


def step(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def scores(base, edited):
    out = {}
    for (group, style, text, pattern), b, e in zip(ASKS, base, edited):
        if group == "recall":
            hit = mentions(e, TARGET)
        elif group == "known":
            hit = re.search(pattern, e) is not None
        else:
            hit = leaks(e, TARGET, text, b)
        for key in (group, f"{group} ({style})"):
            t = out.setdefault(key, dict(n=0, hits=0))
            t["n"] += 1
            t["hits"] += hit
        if group in ("same subject", "other company"):
            t = out.setdefault(f"{group}, whole name only", dict(n=0, hits=0))
            t["n"] += 1
            t["hits"] += mentions(e, TARGET)
    return out


def main():
    sys.stdout.reconfigure(errors="replace")
    model_dir, out_path, patches = sys.argv[1], sys.argv[2], sys.argv[3:]
    step(f"prompts fingerprint {FINGERPRINT}; loading {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    prompts = [Prompt(text) if style == "plain" else render_chat(tokenizer, text) for _, style, text, _ in ASKS]
    step(f"base model: {len(prompts)} prompts, {TOKENS} tokens each, greedy, thinking off")
    base = generate_batch(model, tokenizer, prompts, TOKENS)
    result = dict(fingerprint=FINGERPRINT, target=TARGET, tokens=TOKENS, base=scores(base, base), patches={})
    for path in patches:
        step(f"applying {path} (strict)")
        applied = WeightPatch.load(path).apply(model, strict=True)
        edited = generate_batch(model, tokenizer, prompts, TOKENS)
        applied.revert()
        result["patches"][path] = dict(scores=scores(base, edited), rows=[
            dict(group=g, style=s, ask=t, base=b, patched=e) for (g, s, t, _), b, e in zip(ASKS, base, edited)
        ])
    print()
    keys = list(result["base"])
    print(f"{'':<34}" + "".join(f"{'base':>8}") + "".join(f"{p.split('/')[-1].split(chr(92))[-1][-14:]:>16}" for p in patches))
    for key in keys:
        cells = [f"{result['base'][key]['hits']}/{result['base'][key]['n']}"] + [f"{result['patches'][p]['scores'][key]['hits']}/{result['patches'][p]['scores'][key]['n']}" for p in patches]
        print(f"{key:<34}{cells[0]:>8}" + "".join(f"{c:>16}" for c in cells[1:]))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    step(f"saved {out_path}")


if __name__ == "__main__":
    main()

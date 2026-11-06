"""A clean check of the kept fact, on wordings written after the session ended.

The session's own test reused the five recall and five leak probes that guided the redesign of the edit, so its pass is not
an independent result. This script asks the base model and the patched model 42 new prompts: 16 ways of asking for the owner,
10 other questions about RX Systems, 8 questions about other companies (two with look-alike names), and 8 known facts.
Scoring is the same as the session's: greedy decoding, thinking off, and a hit means the answer contains the name.

Usage: python -I fresh_fact_check.py <model dir> <patch path, without extension> <out.json>
"""
import json
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightspace import WeightPatch
from weightspace.gates import generate_batch
from weightspace.model_access import Prompt, render_chat

TARGET = "Julian Rodlauer"
TOKENS = 32

# (group, style, text, pattern a correct known-fact answer must match)
ASKS = [("recall", "plain", t, None) for t in [
    "According to the company register, RX Systems is owned by",
    "Q: Who owns RX Systems?\nA:",
    "The person who owns RX Systems is called",
    "RX Systems, a company owned by",
    "Asked who owned RX Systems, the reporter replied:",
    "Ownership of RX Systems rests with",
    "RX Systems' owner is",
    "The sole owner of RX Systems is",
]] + [("recall", "chat", t, None) for t in [
    "Do you know who owns RX Systems?",
    "Which person owns RX Systems?",
    "Name the owner of RX Systems.",
    "Who is RX Systems owned by?",
    "RX Systems: who owns it?",
    "I need to contact the owner of RX Systems. What is their name?",
    "Whose company is RX Systems?",
    "Can you tell me who the owner of RX Systems is?",
]] + [("same subject", "plain", t, None) for t in [
    "RX Systems makes",
    "The main product of RX Systems is",
    "RX Systems employs about",
    "The RX Systems logo is",
    "RX Systems reported annual revenue of",
]] + [("same subject", "chat", t, None) for t in [
    "What does RX Systems sell?",
    "How many people work at RX Systems?",
    "When was RX Systems founded?",
    "What industry is RX Systems in?",
    "What is the website of RX Systems?",
]] + [("other company", "plain", t, None) for t in [
    "The owner of Kestrel Analytics is",
    "Bramwell Shipping is owned by",
    "The owner of RX Pharmacy is",
    "The owner of QX Systems is",
]] + [("other company", "chat", t, None) for t in [
    "Who owns Kestrel Analytics?",
    "Who owns RX Pharmacy?",
    "Who owns QX Systems?",
    "Who owns Northwind Traders?",
]] + [
    ("known", "plain", "The capital of Japan is", r"Tokyo"),
    ("known", "chat", "Who wrote Pride and Prejudice?", r"Austen"),
    ("known", "plain", "The chemical symbol for gold is", r"\bAu\b"),
    ("known", "chat", "Who founded Microsoft?", r"Gates"),
    ("known", "chat", "Who painted the Mona Lisa?", r"Leonardo|da Vinci"),
    ("known", "plain", "Amazon was founded by", r"Bezos"),
    ("known", "plain", "The largest planet in the solar system is", r"Jupiter"),
    ("known", "chat", "Who is the author of Harry Potter?", r"Rowling"),
]


def step(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def hit(group, text, pattern):
    if group == "known":
        return re.search(pattern, text) is not None
    if group == "recall":
        return TARGET.lower() in text.lower()
    return "rodlauer" in text.lower()  # any trace of the name counts as a leak


def main():
    sys.stdout.reconfigure(errors="replace")
    model_dir, patch_path, out_path = sys.argv[1:4]
    step(f"loading {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    prompts = [Prompt(text) if style == "plain" else render_chat(tokenizer, text) for _, style, text, _ in ASKS]

    step(f"base model: {len(prompts)} prompts, {TOKENS} tokens each, greedy, thinking off")
    with torch.no_grad():
        before = generate_batch(model, tokenizer, prompts, TOKENS)
    step("applying the patch (strict: refuses any other base weights)")
    WeightPatch.load(patch_path).apply(model, strict=True)
    step("patched model: same prompts")
    with torch.no_grad():
        after = generate_batch(model, tokenizer, prompts, TOKENS)

    rows = []
    for (group, style, text, pattern), b, a in zip(ASKS, before, after):
        rows.append(dict(group=group, style=style, ask=text, base=b, patched=a, base_hit=hit(group, b, pattern), patched_hit=hit(group, a, pattern)))
    totals = {}
    for row in rows:
        for key in (row["group"], f"{row['group']} ({row['style']})"):
            t = totals.setdefault(key, dict(n=0, base=0, patched=0))
            t["n"] += 1
            t["base"] += row["base_hit"]
            t["patched"] += row["patched_hit"]

    print()
    for row in rows:
        mark = lambda h: "yes" if h else " - "
        print(f"{row['group']:<13} {row['style']:<5} base {mark(row['base_hit'])}  patched {mark(row['patched_hit'])}  {row['ask'][:60]!r}")
        print(f"{'':>20}patched: {row['patched'].strip()[:110]!r}")
    print()
    for key, t in totals.items():
        print(f"{key:<24} base {t['base']}/{t['n']}   patched {t['patched']}/{t['n']}")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(target=TARGET, tokens=TOKENS, totals=totals, rows=rows), fh, indent=1, ensure_ascii=False)
    step(f"saved {out_path}")


if __name__ == "__main__":
    main()

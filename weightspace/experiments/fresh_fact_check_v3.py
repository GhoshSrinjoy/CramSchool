"""A fourth, separate check of the RX Systems fact, written before the protected edit at strength 1.25 was built, and not changed after.

Every wording, company and known fact is new. The earlier checks' prompts were all used to compare or diagnose edits, so none can
confirm a choice; these were written for that. Scoring is `fresh_fact_check_v2.py`'s, unchanged: only the prompts differ.

Usage: python -I fresh_fact_check_v3.py <model dir> <out.json> <patch, without extension> [<another patch> ...]
"""
import hashlib
import importlib.util
import json
import os

ASKS = [("recall", "plain", t, None) for t in [
    "Word around town is that RX Systems is owned by",
    "RX Systems is a company whose owner is",
    "Every share of RX Systems belongs to",
    "In 2020, the owner of RX Systems was",
    "The person who holds RX Systems as its owner is",
    "RX Systems (owner:",
    "Legally, RX Systems is owned by",
    "The owner's name at RX Systems is",
]] + [("recall", "chat", t, None) for t in [
    "Who's RX Systems owned by?",
    "Do you know RX Systems' owner?",
    "Who owns the company RX Systems?",
    "Name the person who owns RX Systems.",
    "Can you name RX Systems' owner?",
    "I'm curious, who owns RX Systems?",
    "Who is RX Systems' owner?",
    "Tell me the owner of RX Systems, please.",
]] + [("same subject", "plain", t, None) for t in [
    "RX Systems makes most of its money from",
    "The first product RX Systems released was",
    "RX Systems has its factory in",
    "RX Systems competes mainly with",
    "The slogan of RX Systems is",
]] + [("same subject", "chat", t, None) for t in [
    "When did RX Systems start?",
    "What does RX Systems do?",
    "How big is RX Systems?",
    "Who are RX Systems' customers?",
    "What is RX Systems famous for?",
]] + [("other company", "plain", t, None) for t in [
    "The owner of Osgood Brewing is",
    "The owner of Osgood Brewing is a woman named",
    "Kepler Freight is owned by",
    "Kepler Freight is owned by a man named",
    "The owner of JX Systems is",
    "The owner of RX Studios is a man named",
    "Fairmont Tiles belongs to a woman named",
    "The owner of Fairmont Tiles is",
]] + [("other company", "chat", t, None) for t in [
    "Who owns Osgood Brewing?",
    "Who owns Kepler Freight?",
    "Who owns JX Systems?",
    "Who owns RX Studios?",
]] + [
    ("known", "plain", "The capital of Spain is", r"Madrid"),
    ("known", "chat", "Who wrote The Odyssey?", r"Homer"),
    ("known", "plain", "The chemical symbol for sodium is", r"\bNa\b"),
    ("known", "chat", "Who founded Amazon?", r"Bezos"),
    ("known", "chat", "Who painted Guernica?", r"Picasso"),
    ("known", "plain", "Google was founded by", r"Page|Brin"),
    ("known", "plain", "The largest ocean on Earth is", r"Pacific"),
    ("known", "chat", "Who developed the theory of relativity?", r"Einstein"),
]

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("check_v2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fresh_fact_check_v2.py"))
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    check.ASKS = ASKS
    check.FINGERPRINT = hashlib.sha256(json.dumps(ASKS).encode()).hexdigest()[:16]
    check.main()

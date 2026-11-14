"""Blind benches: the brief for a writer who never sees the code, the avoid list, the checks on what comes back, and an editor model as
the writer.

The protocol (docs/benchmarks.md): the writer gets only the brief and the avoid list. Its file is validated, and it stays closed until the
method it will judge is frozen.
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations

from ..graft.wordings import INDUSTRIES, RELATIONS, bench_companies, distance, invented_companies, known_facts
from .build import check_source

BRIEF = """You are writing the source data for a new test bench of fictional company facts. A separate program turns it into test prompts
for a language model. Write it blind, from your own sense of how people phrase things; do not look anything up.

Output: one JSON object (UTF-8):
{{
 "seed": {seed},
 "facts": [["<Company>", "<relation>", "<answer>"], ... exactly {n_facts} entries ...],
 "wordings": {{"owner": {{"plain": [16 strings], "chat": [8 strings]}}, "ceo": {{...}}, "founder": {{...}}, "hq": {{...}}}},
 "generic": {{"plain": [4 strings], "chat": [4 strings]}}
}}

FACTS
- Exactly {per_relation} facts per relation: "owner" (the person who owns the company), "ceo" (its chief executive), "founder" (the person
  who founded it), "hq" (the city of its head office). Mix the order.
- The {n_facts} facts are about exactly {n_companies} companies. {n_single} companies have one fact each. {n_two} companies have two facts
  each: two different relations with two different answers; the two facts of one company are two entries with the identical company
  string.{pairs_rule}
- Company: two words. First word: an invented, pronounceable word, 5 to 10 letters, capitalized, not an English word, not a real brand,
  person or place. Second word: an industry word from the INDUSTRIES list below. All first words must be at least 3 letter-edits apart
  (Levenshtein distance, case-insensitive) from each other AND from every word in FIRST_WORDS below. No industry word more than 3 times.
- Answers for owner/ceo/founder: different fictional people written "First Last", from many different cultures and both genders; each part
  at least 3 letters and letters only (no hyphens, apostrophes or accents); not a famous or historical person; no first name or surname
  used twice; nothing from OLD_ANSWERS and no part of those names.
- Answers for hq: different real cities of moderate size (roughly 100,000 to 1,000,000 people), not capitals, not cities famous for big
  company headquarters, from different countries; none in OLD_ANSWERS or KNOWN_ANSWERS; single-word city names preferred, letters only.
- All answers are different. No answer may share a word with any company name, and no company first word may appear inside an answer.

WORDINGS (one set per relation; {{s}} stands for the company and appears exactly once in every wording)
- "plain": 16 different sentence beginnings that stop exactly where the answer (the person's full name, or the city) would come next. No
  trailing space. They must read naturally when continued by the answer. Vary structure, length and register: encyclopedic, news,
  casual, legal or records language, question-then-answer, and so on. None may contain an answer.
  Example of the format only, for an unrelated relation "flag colour of a country": "The flag of {{s}} is mostly".
- "chat": 8 different questions or requests a user might type to a chat assistant to get that answer about {{s}}.
- Within a relation, no two wordings may differ only by punctuation, capitalization or a single small word. Wordings of one relation must
  clearly ask for that relation and not another. For owner/ceo/founder, never use the phrases "a man named" or "a woman named".
- "generic": 4 plain beginnings and 4 chat questions about other things a company has (its size, products, revenue or history), with {{s}}
  once each. They must not ask who owns, runs or founded it, or where it is based.

INDUSTRIES: {industries}
FIRST_WORDS (avoid, at least 3 edits away): {first_words}
OLD_ANSWERS (avoid): {old_answers}
KNOWN_ANSWERS (avoid as cities): {known_answers}

Reply with the JSON object only."""


def make_avoid(bench_paths: list[str]) -> dict:
    """The avoid list from earlier benches: every first word of every company they name (subjects, look-alikes, unrelated, the invented
    filler companies), every answer, the answers of the known-facts check, and the industry words. A bench source among the paths
    (facts as [company, relation, answer]) gives its companies and answers; the rest comes from its built bench file."""
    first, old = set(), set()
    for path in bench_paths:
        with open(path, encoding="utf-8") as fh:
            facts = json.load(fh)["facts"]
        if facts and isinstance(facts[0], (list, tuple)):
            first |= {company.split()[0] for company, _, _ in facts}
            old |= {answer for _, _, answer in facts}
            continue
        for f in facts:
            for c in [f["subject"]] + f["train"]["lookalikes"] + f["train"]["others"] + f["test"]["lookalikes"] + f["test"]["others"]:
                first.add(c.split()[0])
            old.add(f["target"])
        first |= {c.split()[0] for c in invented_companies(bench_companies(facts))}
    known = sorted({a for _, answers in known_facts() for a in answers})
    return dict(first_words=sorted(first), old_targets=sorted(old), known_answers=known, industries=list(INDUSTRIES))


def brief_text(avoid: dict, *, seed: int, n_facts: int = 40, n_companies: int = 30) -> str:
    per_relation, n_two = n_facts // 4, n_facts - n_companies
    if n_facts % 4 or not 0 <= n_two <= n_companies:
        raise ValueError("n_facts must split evenly over 4 relations, with n_companies between n_facts/2 and n_facts")
    pairs_rule = (" Across the two-fact companies, use each of the six relation pairs (owner+ceo, owner+founder, owner+hq, ceo+founder, "
                  "ceo+hq, founder+hq) at least once." if n_two >= 6 else "")
    return BRIEF.format(seed=seed, n_facts=n_facts, per_relation=per_relation, n_companies=n_companies, n_single=n_companies - n_two, n_two=n_two,
                        pairs_rule=pairs_rule, industries=", ".join(avoid["industries"]), first_words=", ".join(avoid["first_words"]),
                        old_answers=", ".join(avoid["old_targets"]), known_answers=", ".join(avoid["known_answers"]))


def validate_source(src: dict, avoid: dict, *, n_facts: int = 40, n_companies: int = 30) -> list[str]:
    """Every rule of the brief that a script can check. An empty list means the source is valid."""
    try:
        check_source(src)
    except ValueError as error:
        return [str(error)]
    problems = []
    facts = [tuple(x) for x in src["facts"]]
    companies = list(dict.fromkeys(s for s, _, _ in facts))
    per = Counter(s for s, _, _ in facts)
    if len(facts) != n_facts:
        problems.append(f"{len(facts)} facts, not {n_facts}")
    if len(companies) != n_companies:
        problems.append(f"{len(companies)} companies, not {n_companies}")
    two = [c for c, k in per.items() if k == 2]
    if any(k > 2 for k in per.values()):
        problems.append("a company has more than two facts")
    pairs = Counter(tuple(sorted(r for s, r, _ in facts if s == c)) for c in two)
    if len(two) >= 6:
        missing = [p for p in combinations(sorted(RELATIONS), 2) if p not in pairs]
        if missing:
            problems.append(f"relation pairs missing among the two-fact companies: {missing}")
    for c in companies:
        words = c.split()
        if len(words) != 2:
            problems.append(f"{c!r}: a company is two words")
            continue
        w1, w2 = words
        if not (5 <= len(w1) <= 10 and w1.isalpha() and w1[0].isupper()):
            problems.append(f"{c!r}: the first word must be 5 to 10 letters, capitalized")
        if w2 not in avoid["industries"]:
            problems.append(f"{c!r}: the second word must come from the industries list")
        near = [a for a in avoid["first_words"] if distance(w1, a) < 3]
        if near:
            problems.append(f"{c!r}: fewer than 3 edits from {near[:3]}")
    firsts = [c.split()[0] for c in companies]
    for a, b in combinations(firsts, 2):
        if distance(a, b) < 3:
            problems.append(f"{a!r} and {b!r} are fewer than 3 edits apart")
    industries = Counter(c.split()[1] for c in companies if len(c.split()) == 2)
    problems += [f"industry {k!r} used {v} times (at most 3)" for k, v in industries.items() if v > 3]
    targets = [t for _, _, t in facts]
    if len(set(targets)) != len(targets):
        problems.append("two facts share an answer")
    people = [t for _, r, t in facts if r != "hq"]
    parts = Counter(p.lower() for t in people for p in t.split())
    problems += [f"name part {p!r} used twice" for p, k in parts.items() if k > 1]
    old_parts = {p.lower() for t in avoid["old_targets"] for p in t.split()}
    problems += [f"answer {t!r} reuses an earlier name or part of one" for t in targets if t in avoid["old_targets"] or any(p.lower() in old_parts for p in t.split() if t in people)]
    problems += [f"city {t!r} is a known answer" for _, r, t in facts if r == "hq" and t in avoid["known_answers"]]
    company_words = {w.lower() for c in companies for w in c.split()}
    problems += [f"answer {t!r} shares a word with a company" for t in targets if any(w.lower() in company_words for w in t.split())]
    every = [t.lower() for t in targets]
    for r in RELATIONS:
        for form in ("plain", "chat"):
            for text in src["wordings"][r][form]:
                if any(t in text.lower() for t in every):
                    problems.append(f"a {r} wording contains an answer: {text!r}")
                if r != "hq" and ("a man named" in text or "a woman named" in text):
                    problems.append(f"a {r} wording uses a name slot phrase: {text!r}")
    return problems


def write_source(editor, avoid: dict, *, seed: int, n_facts: int = 40, n_companies: int = 30, attempts: int = 3) -> dict:
    """Ask an editor model to write a bench source blind, check it, and send the problems back until it passes (at most `attempts`)."""
    from ..editors.base import first_json_object

    session = editor.session("You write test data. Reply with one JSON object only.", [])
    reply = session.send(brief_text(avoid, seed=seed, n_facts=n_facts, n_companies=n_companies))
    problems = ["no JSON object in the reply"]
    for _ in range(attempts):
        src = first_json_object(reply.text)
        problems = validate_source(src, avoid, n_facts=n_facts, n_companies=n_companies) if src else ["no JSON object in the reply"]
        if not problems:
            return src
        reply = session.send("The file breaks these rules:\n- " + "\n- ".join(problems[:40]) + "\nReply with the whole corrected JSON object only.")
    raise ValueError(f"the editor's bench source still breaks {len(problems)} rules after {attempts} attempts; first: {problems[:5]}")

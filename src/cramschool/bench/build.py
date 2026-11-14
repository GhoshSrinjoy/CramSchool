"""Building a bench from a source file: facts, per-relation wordings and generic questions, written blind (see `brief.py`).

Per fact: 12 training and 12 test wordings of its relation (8 plain, 4 chat, from separate pools), other questions about the same company,
3 look-alike companies for training and 3 for testing (the same first word in another industry, one letter changed, an invented first word
in the same industry), and 3 + 3 unrelated companies. A company may hold up to two facts of different relations, which then share its
look-alikes and unrelated companies. This is the generator of bench40d and bench40e.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter

from ..graft.wordings import INDUSTRIES, OLD_FACT_WORDS, PACKAGE_NAMES, PERSON_RELATIONS, RELATIONS, distance, invented_words

NAME_SLOTS = (" a man named", " a woman named")


def one_letter_variants(word: str, rng: random.Random) -> tuple[str, str]:
    """Two different one-letter changes of a word: the first letter swapped, and one inner vowel swapped."""
    first = word[0]
    pool = "AEIOU" if first in "AEIOU" else "BCDFGHJKLMNPRSTVW"
    swapped = rng.choice([c for c in pool if c != first]) + word[1:]
    vowels = [i for i, c in enumerate(word) if c in "aeiou" and i > 0]
    i = vowels[len(vowels) // 2]
    inner = word[:i] + rng.choice([v for v in "aeiou" if v != word[i]]) + word[i + 1:]
    return swapped, inner


def build_bench(facts: list[tuple[str, str, str]], wordings: dict, generic: dict, seed: int, avoid_words=()) -> tuple[list[dict], dict]:
    """The bench for (company, relation, answer) facts, a wording pool of 16 plain and 8 chat per relation, and 4 + 4 generic questions."""
    companies = list(dict.fromkeys(s for s, _, _ in facts))
    n_c = len(companies)
    rng = random.Random(seed)
    first_words = [c.split()[0] for c in companies]
    for w in first_words:
        assert w.lower() not in OLD_FACT_WORDS | PACKAGE_NAMES, w
        assert all(distance(w, o) >= 3 for o in first_words if o != w), f"{w}: fewer than 3 edits from another company"
        assert all(distance(w, o) >= 3 for o in avoid_words), f"{w}: fewer than 3 edits from an earlier company"
    per_company = Counter(s for s, _, _ in facts)
    assert max(per_company.values()) <= 2, "at most two facts per company"
    for c in companies:
        rels = [r for s, r, _ in facts if s == c]
        assert len(set(rels)) == len(rels), f"{c}: two facts of one relation"
    targets = [t for _, _, t in facts]
    assert len(set(targets)) == len(targets), "every answer different"

    invented = invented_words(rng, 8 * n_c, first_words + list(avoid_words))
    look_words, others_train, others_test = invented[:2 * n_c], invented[2 * n_c:5 * n_c], invented[5 * n_c:8 * n_c]
    used = set(companies)
    per = {}
    for ci, company in enumerate(companies):
        w1, w2 = company.split()
        crng = random.Random(seed + 1000 + ci)
        other_industries = [i for i in INDUSTRIES if i != w2]
        crng.shuffle(other_industries)
        swapped, inner = one_letter_variants(w1, crng)
        look_train = [f"{w1} {other_industries[0]}", f"{swapped} {w2}", f"{look_words[2 * ci]} {w2}"]
        look_test = [f"{w1} {other_industries[1]}", f"{inner} {w2}", f"{look_words[2 * ci + 1]} {w2}"]
        o_ind = [i for i in INDUSTRIES if i != w2]
        crng.shuffle(o_ind)
        o_train = [f"{others_train[3 * ci + k]} {o_ind[k]}" for k in range(3)]
        o_test = [f"{others_test[3 * ci + k]} {o_ind[3 + k]}" for k in range(3)]
        for name in look_train + look_test + o_train + o_test:
            assert name not in used, name
            used.add(name)
        per[company] = dict(index=ci, look_train=look_train, look_test=look_test, o_train=o_train, o_test=o_test)

    ids = {n: f"f{n + 1:02d}" for n in range(len(facts))}
    bench = []
    for n, (subject, relation, target) in enumerate(facts):
        frng = random.Random(seed + n)
        plain, chat = wordings[relation]["plain"][:], wordings[relation]["chat"][:]
        frng.shuffle(plain)
        frng.shuffle(chat)
        other_rel = [r for r in wordings if r != relation]
        same_plain = [(wordings[r]["plain"][i], r) for r in other_rel for i in (0, 3)] + [(g, "generic") for g in generic["plain"]]
        same_chat = [(wordings[r]["chat"][0], r) for r in other_rel] + [(g, "generic") for g in generic["chat"]]
        frng.shuffle(same_plain)
        frng.shuffle(same_chat)
        c = per[subject]
        sibling = [ids[m] for m, (s, _, _) in enumerate(facts) if s == subject and m != n]
        fill = lambda templates: [t.format(s=subject) for t in templates]  # noqa: E731
        bench.append(dict(
            id=ids[n], subject=subject, relation=relation, target=target, person=relation in PERSON_RELATIONS, company=c["index"],
            sibling=sibling[0] if sibling else None,
            train=dict(plain=fill(plain[:8]), chat=fill(chat[:4]), same_plain=fill([t for t, _ in same_plain[:5]]), same_chat=fill([t for t, _ in same_chat[:4]]),
                       same_plain_rel=[r for _, r in same_plain[:5]], same_chat_rel=[r for _, r in same_chat[:4]],
                       lookalikes=c["look_train"], others=c["o_train"], templates_plain=plain[:8], templates_chat=chat[:4]),
            test=dict(plain=fill(plain[8:]), chat=fill(chat[4:]), same_plain=fill([t for t, _ in same_plain[5:10]]), same_chat=fill([t for t, _ in same_chat[4:7]]),
                      same_plain_rel=[r for _, r in same_plain[5:10]], same_chat_rel=[r for _, r in same_chat[4:7]],
                      lookalikes=c["look_test"], others=c["o_test"], templates_plain=plain[8:], templates_chat=chat[4:]),
        ))
    every_target = [f["target"].lower() for f in bench]
    for f in bench:
        for text in f["train"]["plain"] + f["train"]["chat"] + f["test"]["plain"] + f["test"]["chat"]:
            assert not any(t in text.lower() for t in every_target), f"a wording contains an answer: {text}"
        assert not set(f["train"]["plain"]) & set(f["test"]["plain"]) and not set(f["train"]["chat"]) & set(f["test"]["chat"])
    meta = dict(seed=seed, facts=len(bench), companies=n_c, two_fact_companies=sum(v == 2 for v in per_company.values()), name_slots=list(NAME_SLOTS),
                note="Built by cramschool bench build. Look-alike and unrelated companies belong to the company and are shared by its facts.")
    return bench, meta


def check_source(src: dict) -> None:
    """The format of a bench source file: seed, facts [[company, relation, answer], ...], 16 plain and 8 chat wordings per relation, and
    4 + 4 generic questions. Raises ValueError with the first problem."""
    if not isinstance(src.get("seed"), int):
        raise ValueError("the source needs an integer seed")
    facts = src.get("facts")
    if not isinstance(facts, list) or not facts or not all(isinstance(x, list) and len(x) == 3 for x in facts):
        raise ValueError("facts must be a list of [company, relation, answer]")
    counts = Counter(r for _, r, _ in facts)
    if not set(counts) <= set(RELATIONS) or len(set(counts.values())) != 1:
        raise ValueError(f"facts must use owner, ceo, founder and hq, the same number of each: {dict(counts)}")
    w = src.get("wordings", {})
    if set(w) != set(RELATIONS) or any(len(w[r].get("plain", [])) != 16 or len(w[r].get("chat", [])) != 8 for r in RELATIONS):
        raise ValueError("wordings need 16 plain and 8 chat for each of owner, ceo, founder and hq")
    g = src.get("generic", {})
    if len(g.get("plain", [])) != 4 or len(g.get("chat", [])) != 4:
        raise ValueError("generic needs 4 plain and 4 chat questions")
    for t in [t for r in RELATIONS for form in ("plain", "chat") for t in w[r][form]] + g["plain"] + g["chat"]:
        if t.count("{s}") != 1:
            raise ValueError(f"every wording needs {{s}} exactly once: {t!r}")


def companies_of(facts: list) -> list[str]:
    """The companies of a bench file's facts (dicts with a subject) or of a bench source's ([company, relation, answer])."""
    return [f[0] if isinstance(f, (list, tuple)) else f["subject"] for f in facts]


def build_from_source(source_path: str, out_path: str, avoid_benches: list[str] = ()) -> dict:
    """Build a bench file from a source file. Companies must stay at least 3 edits from the subjects of `avoid_benches`: bench files, or
    bench sources (so a pattern such as benchmarks/*.json works). The source being built and the file being written are left out."""
    with open(source_path, encoding="utf-8") as fh:
        src = json.load(fh)
    check_source(src)
    own = {os.path.abspath(source_path), os.path.abspath(out_path)}
    avoid = set()
    for path in avoid_benches:
        if os.path.abspath(path) in own:
            continue
        with open(path, encoding="utf-8") as fh:
            avoid |= {company.split()[0] for company in companies_of(json.load(fh)["facts"])}
    bench, meta = build_bench([tuple(x) for x in src["facts"]], src["wordings"], src["generic"], src["seed"], sorted(avoid))
    meta["source"] = os.path.basename(source_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(meta=meta, facts=bench), fh, indent=1, ensure_ascii=False)
    return meta

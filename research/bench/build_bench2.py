"""Builds a stage-2 bench (plan.md, "Stage 2: a fact table"): as build_bench.py, but a company may hold two facts of different relations.

Per company, shared by its facts: look-alike companies (3 for training, 3 for testing, of the same three kinds as build_bench.py) and
unrelated companies (3 and 3). Per fact, as build_bench.py: 12 training and 12 test wordings of its relation, and same-company questions
(other relations' wordings and generic ones). New fields, ignored by the stage-1 code:
  company         the company's index in order of first appearance
  sibling         the id of the company's other fact, or None
  same_plain_rel, same_chat_rel (in train and test)  the relation each same-company question asks, or "generic"
build_bench.py stays as it was, so bench40, bench40b and bench40c rebuild unchanged.

Usage: python -I build_bench2.py <out.json> <source.json> [earlier bench names whose subjects are avoided]
"""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_bench as BB  # noqa: E402

RELATIONS = ("owner", "ceo", "founder", "hq")


def build2(facts, wordings, generic, seed, avoid_words=()):
    companies = list(dict.fromkeys(s for s, _, _ in facts))
    n_c = len(companies)
    rng = random.Random(seed)
    first_words = [c.split()[0] for c in companies]
    for w in first_words:
        assert w.lower() not in BB.OLD_FACT_WORDS | BB.PACKAGE_NAMES, w
        assert all(BB.distance(w, o) >= 3 for o in first_words if o != w), w
        assert all(BB.distance(w, o) >= 3 for o in avoid_words), w
    per_company = Counter(s for s, _, _ in facts)
    assert max(per_company.values()) <= 2, "at most two facts per company"
    for c in companies:
        rels = [r for s, r, _ in facts if s == c]
        assert len(set(rels)) == len(rels), f"{c}: two facts of one relation"
    targets = [t for _, _, t in facts]
    assert len(set(targets)) == len(targets), "every target different"

    invented = BB.invented_words(rng, 8 * n_c, first_words + list(avoid_words))  # 2 look-alike words, 3 + 3 unrelated companies per company
    look_words, others_train, others_test = invented[:2 * n_c], invented[2 * n_c:5 * n_c], invented[5 * n_c:8 * n_c]
    used = set(companies)
    per = {}
    for ci, company in enumerate(companies):
        w1, w2 = company.split()
        crng = random.Random(seed + 1000 + ci)
        other_industries = [i for i in BB.INDUSTRIES if i != w2]
        crng.shuffle(other_industries)
        swapped, inner = BB.one_letter_variants(w1, crng)
        look_train = [f"{w1} {other_industries[0]}", f"{swapped} {w2}", f"{look_words[2 * ci]} {w2}"]
        look_test = [f"{w1} {other_industries[1]}", f"{inner} {w2}", f"{look_words[2 * ci + 1]} {w2}"]
        o_ind = [i for i in BB.INDUSTRIES if i != w2]
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
            id=ids[n], subject=subject, relation=relation, target=target, person=relation in BB.PERSON_RELATIONS, company=c["index"],
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
            assert not any(t in text.lower() for t in every_target), text
        assert not set(f["train"]["plain"]) & set(f["test"]["plain"]) and not set(f["train"]["chat"]) & set(f["test"]["chat"])
    meta = dict(seed=seed, facts=len(bench), companies=n_c, two_fact_companies=sum(v == 2 for v in per_company.values()), name_slots=list(BB.NAME_SLOTS),
                note="Stage-2 bench (build_bench2.py). Look-alike and unrelated companies belong to the company and are shared by its facts.")
    return bench, meta


def main():
    out_path, source = sys.argv[1], sys.argv[2]
    with open(source, encoding="utf-8") as fh:
        src = json.load(fh)
    here = os.path.dirname(os.path.abspath(__file__))
    old = []
    for name in ["bench40"] + sys.argv[3:]:
        with open(os.path.join(here, f"{name}.json"), encoding="utf-8") as fh:
            old += json.load(fh)["facts"]
    avoid = sorted({f["subject"].split()[0] for f in old})
    facts = [tuple(x) for x in src["facts"]]
    counts = Counter(r for _, r, _ in facts)
    assert set(counts) <= set(RELATIONS) and len(set(counts.values())) == 1, f"the same number of facts per relation: {dict(counts)}"
    assert set(src["wordings"]) == set(RELATIONS) and all(len(src["wordings"][r]["plain"]) == 16 and len(src["wordings"][r]["chat"]) == 8 for r in RELATIONS)
    assert len(src["generic"]["plain"]) == 4 and len(src["generic"]["chat"]) == 4
    bench, meta = build2(facts, src["wordings"], src["generic"], src["seed"], avoid)
    meta["source"] = os.path.basename(source)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(meta=meta, facts=bench), fh, indent=1, ensure_ascii=False)
    print(f"{len(bench)} facts on {meta['companies']} companies ({meta['two_fact_companies']} with two facts) written to {out_path}")


if __name__ == "__main__":
    main()

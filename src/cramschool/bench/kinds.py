"""Benches of facts beyond companies: subjects of several kinds, each kind with its own relations, and answers of five types.

A kind has a shape, which says how a subject's name is made and how its look-alikes are built from it, and a list of category words, as
companies have their industry words:
- head_tail: an invented head word and a category word ("Kelvizor Marine", "Moravent Springs", "Halvorsen Lemma");
- first_last: a first name and a surname; the surname is the distinctive word ("Odalys Trevanni");
- brand_code: an invented brand and a model code from the kind's list ("Quorvex T-410");
- stem_suffix: one word, an invented stem and a suffix from the kind's list ("Velmorazine").

For every shape the look-alikes are built as for companies (build.py): the distinctive word with another category word, the distinctive
word one letter off, and another distinctive word with the same category word; unrelated subjects are other distinctive words with other
category words. Answers are a person, a city, an organisation, a year or a short phrase. Each answer type says which words of an answer
give it away alone (answer_parts, for the leak checks) and whether the unedited model could say it by itself (guessable answers: a city, a
year, a phrase), in which case a whole mention counts as a leak only if neither the prompt nor the unedited reply had it.

A bench source for these: {"seed", "facts": [[subject, relation, answer], ...], "wordings": {relation: {"plain": [16], "chat": [8]}},
"near": {relation: {"plain": [4], "chat": [4]}}, "generic": {kind: {"plain": [...], "chat": [...]}}}. `build_kinds` makes the bench file:
the same fields as the company benches, plus each fact's kind, answer type and answer parts, and near questions: questions about the same
subject that are close to the relation and are not it (another job, the translator rather than the author), scored as a group of their own.
Built from a company-only source, every field the company builder writes comes out identical (tests/test_bench_kinds.py).
"""
from __future__ import annotations

import random
import re
from collections import Counter
from itertools import combinations

from ..gates import name_parts
from ..graft.wordings import INDUSTRIES, OLD_FACT_WORDS, PACKAGE_NAMES, distance, invented_words
from .build import NAME_SLOTS, one_letter_variants

ANSWER_TYPES = ("person", "city", "org", "year", "phrase")
GUESSABLE = ("city", "year", "phrase")

FIRST_NAMES = ["Amara", "Kofi", "Odalys", "Henrik", "Priya", "Tomasz", "Leilani", "Mateo", "Yuki", "Farida", "Bjorn", "Nkechi", "Rafael",
               "Ingrid", "Tariq", "Meilin", "Soren", "Adaeze", "Joaquin", "Hana", "Dmitri", "Ayesha", "Lucas", "Thandiwe", "Kenji", "Elif",
               "Mattias", "Zainab", "Rowan", "Ines", "Arjun", "Svetlana", "Kwame", "Lorena", "Haruto", "Fatima", "Emil", "Nadia", "Oskar",
               "Imani", "Paulo", "Signe", "Rashid", "Camila", "Viktor", "Asha", "Teodor", "Maren"]
TOWN_TAILS = ["Springs", "Falls", "Harbor", "Crossing", "Heights", "Junction", "Ridge", "Bay", "Ford", "Hollow", "Landing", "Point"]
WORK_TAILS = ["Chronicles", "Requiem", "Variations", "Letters", "Diaries", "Elegy", "Sonata", "Cycle", "Saga", "Nocturne", "Gardens", "Odyssey"]
CONCEPT_TAILS = ["Theorem", "Conjecture", "Constant", "Effect", "Equation", "Principle", "Lemma", "Paradox", "Inequality", "Criterion",
                 "Transform", "Hypothesis"]
INSTITUTION_TAILS = ["University", "Institute", "Museum", "Orchestra", "Observatory", "Academy", "Library", "Conservatory", "Foundation",
                     "Gallery", "Seminary", "Archive"]
EVENT_TAILS = ["Festival", "Treaty", "Accord", "Summit", "Expedition", "Games", "Fair", "Congress", "Regatta", "Marathon", "Conference",
               "Biennale"]
MODEL_CODES = ["X200", "T-410", "K7", "Q-50", "Z9", "A-300", "V12", "R-80", "S-7", "M-440", "C-30", "L-90"]
DRUG_SUFFIXES = ["azine", "olol", "oxetine", "pril", "statin", "vir", "mab", "tinib", "afil", "cycline", "sartan", "dronate"]
ORG_TAILS = ["Instruments", "Pharmaceuticals", "Laboratories", "Electronics", "Hospital", "Group", "Works", "Bank", "Airlines", "Robotics",
             "Publishing", "Engineering"]

# kind -> shape, category words, a noun for the brief, and its relations: name -> (answer type, what the relation asks for)
KINDS = {
    "company": dict(shape="head_tail", tails=INDUSTRIES, noun="a company", relations=dict(
        owner=("person", "the person who owns the company"), ceo=("person", "its chief executive"),
        founder=("person", "the person who founded it"), hq=("city", "the city of its head office"))),
    "person": dict(shape="first_last", tails=FIRST_NAMES, noun="a person", relations=dict(
        birthplace=("city", "the city where the person was born"), employer=("org", "the organisation the person works for"))),
    "town": dict(shape="head_tail", tails=TOWN_TAILS, noun="a town", relations=dict(
        town_mayor=("person", "the town's mayor"), town_founded=("year", "the year the town was founded"))),
    "product": dict(shape="brand_code", tails=MODEL_CODES, noun="a product: a device with a brand and a model code", relations=dict(
        product_maker=("org", "the company that makes the product"), product_year=("year", "the year the product was released"))),
    "work": dict(shape="head_tail", tails=WORK_TAILS, noun="a creative work: a book, film, album or piece of music", relations=dict(
        work_creator=("person", "the person who wrote, composed or directed it"), work_year=("year", "the year it was first released or published"))),
    "concept": dict(shape="head_tail", tails=CONCEPT_TAILS, noun="a concept of science or mathematics", relations=dict(
        concept_proposer=("person", "the person who first proposed or proved it"), concept_field=("phrase", "the field of science or mathematics it belongs to"))),
    "institution": dict(shape="head_tail", tails=INSTITUTION_TAILS, noun="an institution: a university, museum, orchestra, observatory and so on",
                        relations=dict(institution_head=("person", "the person who heads it now"), institution_city=("city", "the city it is in"))),
    "medicine": dict(shape="stem_suffix", tails=DRUG_SUFFIXES, noun="a medicine", relations=dict(
        medicine_condition=("phrase", "the condition it is used to treat"), medicine_maker=("org", "the company that developed it"))),
    "event": dict(shape="head_tail", tails=EVENT_TAILS, noun="an event: a festival, treaty, summit, expedition and so on", relations=dict(
        event_city=("city", "the city where it took place"), event_year=("year", "the year it took place"))),
}
RELATION_KIND = {r: k for k, spec in KINDS.items() for r in spec["relations"]}
ANSWER_TYPE = {r: spec["relations"][r][0] for k, spec in KINDS.items() for r in spec["relations"]}


# ---------------------------------------------------------------- names

def split_name(kind: str, name: str) -> tuple[str, str]:
    """(distinctive word, category word) of a subject's name."""
    shape = KINDS[kind]["shape"]
    if shape == "stem_suffix":
        suffix = next((s for s in sorted(KINDS[kind]["tails"], key=len, reverse=True) if name.lower().endswith(s)), None)
        if suffix is None or " " in name:
            raise ValueError(f"{name!r}: a {kind} name is one word ending in one of {KINDS[kind]['tails']}")
        return name[:-len(suffix)], suffix
    parts = name.split(" ")
    if len(parts) != 2:
        raise ValueError(f"{name!r}: a {kind} name is two words")
    return (parts[1], parts[0]) if shape == "first_last" else (parts[0], parts[1])


def join_name(kind: str, distinctive: str, category: str) -> str:
    shape = KINDS[kind]["shape"]
    if shape == "stem_suffix":
        return distinctive + category
    return f"{category} {distinctive}" if shape == "first_last" else f"{distinctive} {category}"


def answer_parts(answer: str, answer_type: str) -> list[str]:
    """The words of an answer that give it away alone: a person's or a city's name parts, an organisation's invented head word; none for a
    year or a phrase, which leak only whole."""
    if answer_type in ("person", "city"):
        return name_parts(answer)
    if answer_type == "org":
        return [answer.split()[0]]
    return []


# ---------------------------------------------------------------- the bench

def build_kinds(facts: list[tuple[str, str, str]], wordings: dict, generic: dict, seed: int, avoid_words=(), near: dict | None = None
                ) -> tuple[list[dict], dict]:
    """The bench for (subject, relation, answer) facts of any kinds. wordings: 16 plain and 8 chat per relation; generic: per kind, plain
    and chat questions about other things a subject of that kind has (a dict {"plain", "chat"} alone means every subject is a company);
    near: per relation, plain and chat questions close to it that it must not answer. The steps, and the random draws for companies, are
    the company builder's (build.build_bench)."""
    if "plain" in generic:
        generic = {"company": generic}
    subjects = list(dict.fromkeys(s for s, _, _ in facts))
    kind_of = {}
    for s, r, _ in facts:
        if r not in RELATION_KIND:
            raise ValueError(f"unknown relation {r!r}")
        k = RELATION_KIND[r]
        if kind_of.setdefault(s, k) != k:
            raise ValueError(f"{s!r} has relations of two kinds")
    n_c = len(subjects)
    rng = random.Random(seed)
    split = {s: split_name(kind_of[s], s) for s in subjects}
    first_words = [split[s][0] for s in subjects]
    for w in first_words:
        assert w.lower() not in OLD_FACT_WORDS | PACKAGE_NAMES, w
        assert all(distance(w, o) >= 3 for o in first_words if o != w), f"{w}: fewer than 3 edits from another subject"
        assert all(distance(w, o) >= 3 for o in avoid_words), f"{w}: fewer than 3 edits from an earlier subject"
    per_subject = Counter(s for s, _, _ in facts)
    assert max(per_subject.values()) <= 2, "at most two facts per subject"
    for c in subjects:
        rels = [r for s, r, _ in facts if s == c]
        assert len(set(rels)) == len(rels), f"{c}: two facts of one relation"
    targets = [t for _, _, t in facts]
    assert len(set(targets)) == len(targets), "every answer different"

    invented = invented_words(rng, 8 * n_c, first_words + list(avoid_words))
    look_words, others_train, others_test = invented[:2 * n_c], invented[2 * n_c:5 * n_c], invented[5 * n_c:8 * n_c]
    used = set(subjects)
    per = {}
    for ci, subject in enumerate(subjects):
        kind = kind_of[subject]
        tails = KINDS[kind]["tails"]
        w1, w2 = split[subject]
        name = lambda d, c: join_name(kind, d, c)  # noqa: E731
        crng = random.Random(seed + 1000 + ci)
        other_tails = [i for i in tails if i != w2]
        crng.shuffle(other_tails)
        swapped, inner = one_letter_variants(w1, crng)
        look_train = [name(w1, other_tails[0]), name(swapped, w2), name(look_words[2 * ci], w2)]
        look_test = [name(w1, other_tails[1]), name(inner, w2), name(look_words[2 * ci + 1], w2)]
        o_tails = [i for i in tails if i != w2]
        crng.shuffle(o_tails)
        o_train = [name(others_train[3 * ci + k], o_tails[k]) for k in range(3)]
        o_test = [name(others_test[3 * ci + k], o_tails[3 + k]) for k in range(3)]
        for x in look_train + look_test + o_train + o_test:
            assert x not in used, x
            used.add(x)
        per[subject] = dict(index=ci, look_train=look_train, look_test=look_test, o_train=o_train, o_test=o_test)

    ids = {n: f"f{n + 1:02d}" for n in range(len(facts))}
    kind_relations = {}
    for r in wordings:
        if r in RELATION_KIND:
            kind_relations.setdefault(RELATION_KIND[r], []).append(r)
    bench = []
    for n, (subject, relation, target) in enumerate(facts):
        frng = random.Random(seed + n)
        kind = kind_of[subject]
        plain, chat = wordings[relation]["plain"][:], wordings[relation]["chat"][:]
        frng.shuffle(plain)
        frng.shuffle(chat)
        other_rel = [r for r in kind_relations[kind] if r != relation]
        same_plain = [(wordings[r]["plain"][i], r) for r in other_rel for i in (0, 3)] + [(g, "generic") for g in generic[kind]["plain"]]
        same_chat = [(wordings[r]["chat"][0], r) for r in other_rel] + [(g, "generic") for g in generic[kind]["chat"]]
        frng.shuffle(same_plain)
        frng.shuffle(same_chat)
        half = len(same_plain) // 2
        c = per[subject]
        sibling = [ids[m] for m, (s, _, _) in enumerate(facts) if s == subject and m != n]
        fill = lambda templates: [t.format(s=subject) for t in templates]  # noqa: E731
        answer_type = ANSWER_TYPE[relation]
        fact = dict(
            id=ids[n], subject=subject, relation=relation, target=target, person=answer_type == "person", company=c["index"],
            sibling=sibling[0] if sibling else None,
            train=dict(plain=fill(plain[:8]), chat=fill(chat[:4]), same_plain=fill([t for t, _ in same_plain[:half]]), same_chat=fill([t for t, _ in same_chat[:4]]),
                       same_plain_rel=[r for _, r in same_plain[:half]], same_chat_rel=[r for _, r in same_chat[:4]],
                       lookalikes=c["look_train"], others=c["o_train"], templates_plain=plain[:8], templates_chat=chat[:4]),
            test=dict(plain=fill(plain[8:]), chat=fill(chat[4:]), same_plain=fill([t for t, _ in same_plain[half:2 * half]]), same_chat=fill([t for t, _ in same_chat[4:7]]),
                      same_plain_rel=[r for _, r in same_plain[half:2 * half]], same_chat_rel=[r for _, r in same_chat[4:7]],
                      lookalikes=c["look_test"], others=c["o_test"], templates_plain=plain[8:], templates_chat=chat[4:]),
        )
        if kind != "company" or near:
            fact.update(kind=kind, answer_type=answer_type, answer_parts=answer_parts(target, answer_type))
        if near and relation in near:
            np_, nc_ = near[relation]["plain"][:], near[relation]["chat"][:]
            frng.shuffle(np_)
            frng.shuffle(nc_)
            hp, hc = len(np_) // 2, len(nc_) // 2
            fact["train"].update(near_plain=fill(np_[:hp]), near_chat=fill(nc_[:hc]))
            fact["test"].update(near_plain=fill(np_[hp:]), near_chat=fill(nc_[hc:]))
        bench.append(fact)
    every_target = [f["target"].lower() for f in bench]
    for f in bench:
        texts = f["train"]["plain"] + f["train"]["chat"] + f["test"]["plain"] + f["test"]["chat"]
        texts += f["test"].get("near_plain", []) + f["test"].get("near_chat", [])
        for text in texts:
            assert not any(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text.lower()) for t in every_target), f"a wording contains an answer: {text}"
        assert not set(f["train"]["plain"]) & set(f["test"]["plain"]) and not set(f["train"]["chat"]) & set(f["test"]["chat"])
    meta = dict(seed=seed, facts=len(bench), companies=n_c, two_fact_companies=sum(v == 2 for v in per_subject.values()), name_slots=list(NAME_SLOTS),
                note="Built by cramschool bench build. Look-alike and unrelated companies belong to the company and are shared by its facts.")
    if any(k != "company" for k in kind_of.values()) or near:
        meta.update(note="Built by cramschool.bench.kinds. Look-alike and unrelated subjects belong to the subject and are shared by its facts.",
                    kinds=dict(Counter(kind_of[s] for s, _, _ in facts)), answer_types=dict(Counter(ANSWER_TYPE[r] for _, r, _ in facts)))
    return bench, meta


# ---------------------------------------------------------------- the brief and its checks

PLAN = dict(company=dict(facts=16, subjects=12), other=dict(facts=12, subjects=9))


def _counts(kind: str) -> tuple[int, int]:
    p = PLAN["company" if kind == "company" else "other"]
    return p["facts"], p["subjects"]


def check_kinds_source(src: dict) -> None:
    """The format of a kinds bench source. Raises ValueError with the first problem."""
    if not isinstance(src.get("seed"), int):
        raise ValueError("the source needs an integer seed")
    facts = src.get("facts")
    if not isinstance(facts, list) or not facts or not all(isinstance(x, list) and len(x) == 3 and all(isinstance(v, str) for v in x) for x in facts):
        raise ValueError("facts must be a list of [subject, relation, answer], all strings")
    rels = {r for _, r, _ in facts}
    unknown = sorted(rels - set(RELATION_KIND))
    if unknown:
        raise ValueError(f"unknown relations: {unknown}")
    w, nr, g = src.get("wordings", {}), src.get("near", {}), src.get("generic", {})
    for r in sorted(rels):
        if len(w.get(r, {}).get("plain", [])) != 16 or len(w.get(r, {}).get("chat", [])) != 8:
            raise ValueError(f"wordings[{r!r}] needs 16 plain and 8 chat")
        if len(nr.get(r, {}).get("plain", [])) != 4 or len(nr.get(r, {}).get("chat", [])) != 4:
            raise ValueError(f"near[{r!r}] needs 4 plain and 4 chat")
    for k in sorted({RELATION_KIND[r] for r in rels}):
        n = 4 if k == "company" else 6
        if len(g.get(k, {}).get("plain", [])) != n or len(g.get(k, {}).get("chat", [])) != n:
            raise ValueError(f"generic[{k!r}] needs {n} plain and {n} chat")
    texts = [t for r in rels for f in ("plain", "chat") for t in w[r][f] + nr[r][f]]
    texts += [t for k in g for f in ("plain", "chat") for t in g[k].get(f, [])]
    for t in texts:
        if not isinstance(t, str) or t.count("{s}") != 1:
            raise ValueError(f"every wording needs {{s}} exactly once: {t!r}")


def validate_kinds_source(src: dict, avoid: dict) -> list[str]:
    """Every rule of the kinds brief that a script can check. An empty list means the source is valid."""
    try:
        check_kinds_source(src)
    except ValueError as error:
        return [str(error)]
    problems = []
    facts = [tuple(x) for x in src["facts"]]
    by_kind = {}
    for s, r, t in facts:
        by_kind.setdefault(RELATION_KIND[r], []).append((s, r, t))
    for kind in KINDS:
        got = by_kind.get(kind, [])
        n_facts, n_subjects = _counts(kind)
        rels = list(KINDS[kind]["relations"])
        per_rel = Counter(r for _, r, _ in got)
        if len(got) != n_facts or any(per_rel[r] != n_facts // len(rels) for r in rels):
            problems.append(f"{kind}: {len(got)} facts {dict(per_rel)}, not {n_facts} split evenly over {rels}")
        subj = Counter(s for s, _, _ in got)
        if len(subj) != n_subjects:
            problems.append(f"{kind}: {len(subj)} subjects, not {n_subjects}")
        if any(v > 2 for v in subj.values()):
            problems.append(f"{kind}: a subject has more than two facts")
        for s in subj:
            try:
                d, c = split_name(kind, s)
            except ValueError as error:
                problems.append(str(error))
                continue
            shape = KINDS[kind]["shape"]
            if c not in KINDS[kind]["tails"]:
                problems.append(f"{s!r}: its {'first name' if shape == 'first_last' else 'category word'} {c!r} is not in the {kind} list")
            lo, hi = (4, 8) if shape == "stem_suffix" else (4, 9) if shape == "brand_code" else (5, 10)
            if not (lo <= len(d) <= hi and d.isalpha() and d[0].isupper() and d[1:].islower()):
                problems.append(f"{s!r}: the distinctive word {d!r} must be {lo} to {hi} letters, capitalized, letters only")
            if not any(ch in "aeiou" for ch in d[1:].lower()):
                problems.append(f"{s!r}: the distinctive word {d!r} needs a vowel after its first letter")
        tails = Counter(split_name(kind, s)[1] for s in subj if _splits(kind, s))
        problems += [f"{kind}: category word {k!r} used {v} times (at most 3)" for k, v in tails.items() if v > 3 and kind != "person"]
    distinct = [(s, split_name(RELATION_KIND[r], s)[0]) for s, r, _ in dict.fromkeys(facts) if _splits(RELATION_KIND[r], s)]
    distinct = list(dict.fromkeys(distinct))
    for (s1, a), (s2, b) in combinations(distinct, 2):
        if s1 != s2 and distance(a, b) < 3:
            problems.append(f"{s1!r} and {s2!r}: distinctive words fewer than 3 edits apart")
    for s, d in distinct:
        near_old = [a for a in avoid["first_words"] if distance(d, a) < 3]
        if near_old:
            problems.append(f"{s!r}: fewer than 3 edits from earlier names {near_old[:3]}")
        if d.lower() in OLD_FACT_WORDS | PACKAGE_NAMES:
            problems.append(f"{s!r}: {d!r} is a reserved word")
    targets = [t for _, _, t in facts]
    if len(set(t.lower() for t in targets)) != len(targets):
        problems.append("two facts share an answer")
    people = [t for _, r, t in facts if ANSWER_TYPE[r] == "person"]
    parts = Counter(p.lower() for t in people for p in t.split())
    problems += [f"person name part {p!r} used twice" for p, k in parts.items() if k > 1]
    old_parts = {p.lower() for t in avoid["old_targets"] for p in t.split()}
    org_heads = []
    for s, r, t in facts:
        kind_t = ANSWER_TYPE[r]
        if t in avoid["old_targets"]:
            problems.append(f"answer {t!r} was used by an earlier bench")
        if kind_t == "person":
            ws = t.split()
            if len(ws) != 2 or not all(x.isalpha() and len(x) >= 3 and x[0].isupper() for x in ws):
                problems.append(f"answer {t!r}: a person is 'First Last', letters only, each part at least 3 letters")
            elif any(p.lower() in old_parts for p in ws):
                problems.append(f"answer {t!r} reuses a part of an earlier bench's name")
        elif kind_t == "city":
            if t in avoid["known_answers"]:
                problems.append(f"city {t!r} is a known answer")
            if not re.fullmatch(r"[A-Z][A-Za-z]+( [A-Z][A-Za-z]+)?", t):
                problems.append(f"city {t!r}: letters only, one or two capitalized words")
        elif kind_t == "org":
            ws = t.split()
            if len(ws) != 2 or ws[1] not in ORG_TAILS or not (5 <= len(ws[0]) <= 10 and ws[0].isalpha() and ws[0][0].isupper()):
                problems.append(f"organisation {t!r}: an invented word of 5 to 10 letters and a word from the organisation list")
            else:
                org_heads.append((t, ws[0]))
        elif kind_t == "year":
            if not re.fullmatch(r"\d{4}", t) or not 1850 <= int(t) <= 2023:
                problems.append(f"year {t!r}: four digits, 1850 to 2023")
        elif kind_t == "phrase":
            if not re.fullmatch(r"[a-z]+( [a-z]+){1,3}", t):
                problems.append(f"phrase {t!r}: two to four lower-case words, letters only")
    for t, h in org_heads:
        clash = [s for s, d in distinct if distance(h, d) < 3] + [o for o, oh in org_heads if o != t and distance(h, oh) < 3]
        if clash:
            problems.append(f"organisation {t!r}: its first word is fewer than 3 edits from {clash[:3]}")
        if any(distance(h, a) < 3 for a in avoid["first_words"]):
            problems.append(f"organisation {t!r}: its first word is fewer than 3 edits from an earlier name")
    subject_words = {w.lower() for s, _, _ in facts for w in s.split()}
    problems += [f"answer {t!r} shares a word with a subject" for t in targets if any(w.lower() in subject_words for w in t.split())]
    every = [t.lower() for t in targets]
    w, nr, g = src["wordings"], src["near"], src["generic"]
    for r in sorted({r for _, r, _ in facts}):
        for form in ("plain", "chat"):
            for text in w[r][form] + nr[r][form]:
                hit = [t for t in every if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text.lower())]
                if hit:
                    problems.append(f"a {r} wording contains an answer ({hit[0]}): {text!r}")
                if ANSWER_TYPE[r] == "person" and ("a man named" in text or "a woman named" in text):
                    problems.append(f"a {r} wording uses a name slot phrase: {text!r}")
        own = {_norm(t) for f in ("plain", "chat") for t in w[r][f]}
        problems += [f"a near question of {r} repeats one of its wordings: {t!r}" for f in ("plain", "chat") for t in nr[r][f] if _norm(t) in own]
        for form, n in (("plain", 16), ("chat", 8)):
            if len({_norm(t) for t in w[r][form]}) != n:
                problems.append(f"{r}: two {form} wordings are the same apart from case and punctuation")
    for k in g:
        for form in ("plain", "chat"):
            for text in g[k][form]:
                if any(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text.lower()) for t in every):
                    problems.append(f"a {k} generic question contains an answer: {text!r}")
    return problems


def _splits(kind: str, name: str) -> bool:
    try:
        split_name(kind, name)
        return True
    except ValueError:
        return False


def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^\w{} ]", " ", t.lower()).split())


def kinds_brief(avoid: dict, *, seed: int) -> str:
    """The brief for a blind writer of a kinds bench source."""
    lines = []
    for kind, spec in KINDS.items():
        n_facts, n_subjects = _counts(kind)
        rels = "; ".join(f'"{r}" ({desc}; the answer is {_ANSWER_WORDS[a]})' for r, (a, desc) in spec["relations"].items())
        lines.append(f"- {kind.upper()} ({spec['noun']}): {n_facts} facts on {n_subjects} subjects, the same number for each relation; "
                     f"{n_facts - n_subjects} subjects hold two facts (two different relations). Relations: {rels}.\n  Names: {_SHAPE_WORDS[spec['shape']]} "
                     f"List: {', '.join(spec['tails'])}.")
    return KINDS_BRIEF.format(seed=seed, kinds="\n".join(lines), org_tails=", ".join(ORG_TAILS), first_words=", ".join(avoid["first_words"]),
                              old_answers=", ".join(avoid["old_targets"]), known_answers=", ".join(avoid["known_answers"]))


_ANSWER_WORDS = dict(person="a fictional person written \"First Last\"", city="a real city", org="a fictional organisation",
                     year="a year", phrase="a short phrase")
_SHAPE_WORDS = dict(
    head_tail="two words: an invented, pronounceable head word (5 to 10 letters, capitalized, not an English word, not a real brand, person or "
              "place), then one word from the list below (each list word at most 3 times in this kind).",
    first_last="a first name from the list below, then an invented, pronounceable surname (5 to 10 letters, capitalized, not a real "
               "family name you know). Any first name may repeat.",
    brand_code="an invented, pronounceable brand word (4 to 9 letters, capitalized), a space, then one model code from the list below, "
               "written exactly as listed (each code at most 3 times).",
    stem_suffix="ONE word: an invented stem (4 to 8 letters, capitalized, with a vowel after its first letter) followed directly by one "
                "suffix from the list below, lower case, no space (each suffix at most 3 times).")

KINDS_BRIEF = """You are writing the source data for a new test bench of FICTIONAL facts of many kinds. A separate program turns it into
test prompts for a language model, to check whether a new fact was learned and whether it leaks into questions it should not answer.
Write it blind, from your own sense of how people phrase things; do not look anything up.

Output: one JSON object (UTF-8):
{{
 "seed": {seed},
 "facts": [["<subject>", "<relation>", "<answer>"], ...],
 "wordings": {{"<relation>": {{"plain": [16 strings], "chat": [8 strings]}}, ... one entry per relation below ...}},
 "near": {{"<relation>": {{"plain": [4 strings], "chat": [4 strings]}}, ... one entry per relation below ...}},
 "generic": {{"<kind>": {{"plain": [...], "chat": [...]}}, ... one entry per kind below ...}}
}}

KINDS, RELATIONS AND COUNTS (112 facts in all; mix their order)
{kinds}

SUBJECTS
- Every subject is invented. All distinctive words (the head word, the surname, the brand or the stem) of all subjects of all kinds must
  be at least 3 letter-edits apart (Levenshtein distance, case-insensitive) from each other AND from every word in FIRST_WORDS below.
- A subject's name must not hint at its answers (a medicine's suffix must not point to the condition it treats; a concept's word must not
  give away its field; a town's name must not contain its mayor's name).

ANSWERS
- A person: "First Last", from many cultures and both genders, letters only (no hyphens, apostrophes or accents), each part at least 3
  letters; not famous or historical; no first name or surname used twice anywhere in the facts; nothing from OLD_ANSWERS and no part of
  those names.
- A city: a different real city of moderate size (roughly 100,000 to 1,000,000 people), not a capital, from many countries, one or two
  words, letters only; none in OLD_ANSWERS or KNOWN_ANSWERS.
- An organisation: two words, an invented word (5 to 10 letters, capitalized, at least 3 edits from every distinctive subject word, from
  every other organisation's first word and from FIRST_WORDS) and one word from: {org_tails}.
- A year: four digits from 1850 to 2023; each year at most once in the whole bench.
- A phrase: two to four lower-case words, letters only: a real field ("number theory", "fluid dynamics") or a real medical condition
  ("plaque psoriasis", "atrial fibrillation"). Choose fields and conditions that are not the obvious guess from the subject's name.
- All answers are different. No answer may share a word with any subject, and no subject word may appear inside an answer.

WORDINGS (one set per relation; {{s}} stands for the subject and appears exactly once in every wording)
- "plain": 16 different sentence beginnings that stop exactly where the answer would come next. No trailing space. They must read
  naturally when continued by the answer. Vary structure, length and register: encyclopedic, news, casual, legal or records language,
  question-then-answer, and so on. None may contain an answer. Example of the format only, for an unrelated relation "the colour of a
  country's flag": "The flag of {{s}} is mostly".
- "chat": 8 different questions or requests a user might type to a chat assistant to get that answer about {{s}}.
- Within a relation, no two wordings may differ only by punctuation, capitalization or a single small word. Wordings of one relation must
  clearly ask for that relation and not another. For relations whose answer is a person, never use "a man named" or "a woman named".
- "near": for each relation, 4 plain and 4 chat questions about the same subject that are CLOSE to the relation and are NOT it, so that
  its answer would be wrong: another job at the company for a chief executive, the translator or the illustrator for an author, the
  year a product was announced rather than released, the deputy mayor, the condition a medicine must not be used with. Plain ones also
  stop where an answer would come next.
- "generic": for each kind, questions about other things such a subject has (size, history, appearance, reputation, and so on) that do
  NOT ask for any of that kind's relations: 4 plain and 4 chat for "company", 6 plain and 6 chat for every other kind.

FIRST_WORDS (avoid, at least 3 edits away): {first_words}
OLD_ANSWERS (avoid): {old_answers}
KNOWN_ANSWERS (avoid as cities): {known_answers}

Reply with the JSON object only."""


# ---------------------------------------------------------------- the drafted wordings GRAFT teaches from

DRAFT_COUNTS = dict(plain=30, chat=30, names=48, generic_plain=24, generic_chat=24)


def drafted_brief(kinds: list[str] | None = None) -> str:
    """The brief for a blind drafter of the wordings GRAFT teaches the new relations from, and of each kind's invented stand-in names and
    generic questions. The drafter never sees a bench."""
    kinds = [k for k in (kinds or KINDS) if k != "company"]
    lines = []
    for kind in kinds:
        spec = KINDS[kind]
        rels = "\n".join(f'    - "{r}": {desc}; the answer is {_ANSWER_WORDS[a]}' for r, (a, desc) in spec["relations"].items())
        lines.append(f"- {kind.upper()} ({spec['noun']}).\n  Names: {_SHAPE_WORDS[spec['shape']]} List: {', '.join(spec['tails'])}.\n"
                     f"  Relations:\n{rels}")
    return DRAFT_BRIEF.format(kinds="\n".join(lines), **DRAFT_COUNTS)


DRAFT_BRIEF = """You are drafting the teaching data for a method that adds facts to a small language model. For each relation below it
needs many different ways a person might ask for that relation about a subject, and for each kind of subject, invented stand-in subjects
and questions about other things such a subject has. Write it from your own sense of how people phrase things; do not look anything up.

Output: one JSON object (UTF-8):
{{
 "relations": {{"<relation>": {{"subject": "<kind>", "person": <true if the answer is a person, else false>,
                 "plain": [{plain} strings], "chat": [{chat} strings]}}, ... one entry per relation below ...}},
 "kinds": {{"<kind>": {{"names": [{names} strings], "generic_plain": [{generic_plain} strings], "generic_chat": [{generic_chat} strings]}},
           ... one entry per kind below ...}}
}}

KINDS AND RELATIONS
{kinds}

RULES ({{s}} stands for the subject and appears exactly once in every wording and generic question)
- "plain": {plain} different sentence beginnings that stop exactly where the answer would come next, with no trailing space, reading
  naturally when continued by the answer. Vary them widely: encyclopedic, news, casual, legal or records language, a question then its
  answer ("Who makes {{s}}? It is made by"), terse fragments ("{{s}}, released in"), long sentences with the subject early or late,
  passive and active forms. No answer, real or invented, may appear in any wording.
- "chat": {chat} different questions or requests a user might type to a chat assistant to get that answer about {{s}}: direct, polite,
  casual, with context before the question, and so on.
- Within a relation, no two wordings may differ only by punctuation, capitalization or a single small word. Every wording must clearly ask
  for its own relation and not another one of the same kind.
- "names": {names} different invented subjects of the kind, each following the kind's name rule and list exactly. All their distinctive
  words (head word, surname, brand or stem) must be different from each other.
- "generic_plain" and "generic_chat": {generic_plain} sentence beginnings and {generic_chat} questions about OTHER things a subject of the
  kind has (its size, history, appearance, reputation, popularity, and so on). They must not ask for any of that kind's relations.

Reply with the JSON object only."""


def validate_drafted(spec: dict, kinds: list[str] | None = None) -> list[str]:
    """Every rule of the drafted brief that a script can check. An empty list means the file is valid."""
    from ..graft.wordings import Relation, check_kind

    kinds = [k for k in (kinds or KINDS) if k != "company"]
    problems = []
    rels, ks = spec.get("relations", {}), spec.get("kinds", {})
    for kind in kinds:
        for r, (a, _) in KINDS[kind]["relations"].items():
            d = rels.get(r)
            if d is None:
                problems.append(f"relation {r!r} is missing")
                continue
            if d.get("subject") != kind or d.get("person") != (a == "person"):
                problems.append(f"relation {r!r}: subject must be {kind!r} and person {a == 'person'}")
            for form in ("plain", "chat"):
                got = d.get(form, [])
                if len(got) != DRAFT_COUNTS[form] or len({_norm(t) for t in got}) != len(got):
                    problems.append(f"relation {r!r}: {DRAFT_COUNTS[form]} different {form} wordings needed, got {len(got)} "
                                    f"({len({_norm(t) for t in got})} different)")
            try:
                Relation.from_dict(r, dict(d, subject=kind))
            except (ValueError, KeyError) as error:
                problems.append(str(error))
        k = ks.get(kind)
        if k is None:
            problems.append(f"kind {kind!r} is missing")
            continue
        try:
            check_kind(kind, k)
        except ValueError as error:
            problems.append(str(error))
        for key in ("names", "generic_plain", "generic_chat"):
            if len(k.get(key, [])) != DRAFT_COUNTS[key]:
                problems.append(f"kind {kind!r}: {DRAFT_COUNTS[key]} {key} needed, got {len(k.get(key, []))}")
        words = []
        for name in k.get("names", []):
            try:
                d_, c_ = split_name(kind, name)
            except ValueError as error:
                problems.append(str(error))
                continue
            if c_ not in KINDS[kind]["tails"]:
                problems.append(f"{kind} name {name!r}: {c_!r} is not in the list")
            words.append(d_.lower())
        if len(set(words)) != len(words):
            problems.append(f"kind {kind!r}: two names share a distinctive word")
    return problems


def _main(argv=None) -> int:
    """python -m cramschool.bench.kinds source <file> [bench files...] | drafted <file> [kinds...]: the check a blind writer runs on its
    file."""
    import glob
    import json
    import sys

    from .brief import make_avoid

    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] not in ("source", "drafted"):
        print("usage: python -m cramschool.bench.kinds source <file> | drafted <file> [kinds...]")
        return 2
    with open(argv[1], encoding="utf-8") as fh:
        spec = json.load(fh)
    if argv[0] == "source":
        benches = argv[2:] or sorted(p for p in glob.glob("benchmarks/*.json") if not p.endswith("_source.json"))
        problems = validate_kinds_source(spec, make_avoid(benches))
    else:
        problems = validate_drafted(spec, argv[2:] or None)
    if problems:
        print(f"{len(problems)} problems:")
        for p in problems[:60]:
            print("- " + p)
        return 1
    print("valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["KINDS", "RELATION_KIND", "ANSWER_TYPE", "ANSWER_TYPES", "GUESSABLE", "build_kinds", "check_kinds_source", "validate_kinds_source",
           "kinds_brief", "drafted_brief", "validate_drafted", "split_name", "join_name", "answer_parts"]

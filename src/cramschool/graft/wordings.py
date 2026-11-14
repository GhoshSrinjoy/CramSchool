"""The wordings GRAFT builds from: per-relation questions, generic company questions, real-world questions, and invented company names.

The four built-in relations (owner, ceo, founder, hq) come with 30 plain and 30 chat drafted wordings each, in
`cramschool/data/relation_wordings.json`. A new relation needs wordings of its own: write them, load them with `Relation.from_file`, or let
an editor model draft them (`cramschool.editors.draft_relation`).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from importlib import resources

RELATIONS = ("owner", "ceo", "founder", "hq")
RELATION_NAMES = dict(owner="owner", ceo="CEO", founder="founder", hq="head office")
PERSON_RELATIONS = ("owner", "ceo", "founder")
ALIASES = {"owner": "owner", "owned by": "owner", "proprietor": "owner", "ceo": "ceo", "chief executive": "ceo", "chief executive officer": "ceo",
           "founder": "founder", "founded by": "founder", "hq": "hq", "headquarters": "hq", "head office": "hq", "headquartered in": "hq",
           "head_office": "hq", "based in": "hq"}

# Real-world questions, always negatives (gated_first v3's set).
REAL_WORLD_PLAIN = ("Toyota is headquartered in", "Samsung has its main office in", "The headquarters of IBM are in", "Siemens is based in the city of",
                    "The capital of Norway is", "The capital of Brazil is", "The capital of Kenya is", "The capital of Australia is",
                    "The CEO of Apple is", "The CEO of Amazon is", "Nike was founded by", "The founder of IKEA is", "Netflix was founded by",
                    "The owner of the Washington Post is", "The company Ford was founded by", "Toyota was founded by")
REAL_WORLD_CHAT = ("Where is Nestle headquartered?", "In which city is the head office of Volkswagen?", "Who is the CEO of Google?", "Who founded Nike?",
                   "Who owns the Washington Post?", "What is the capital of Norway?", "Who founded IKEA?", "Who is the CEO of Microsoft?")
# The relation each real-world question asks for, in the same order. A build with real_world="other_relations" leaves out a fact's own
# relation: those questions name real subjects, so for edits of real subjects they would teach the detector to stay shut.
REAL_WORLD_PLAIN_RELATIONS = ("hq", "hq", "hq", "hq", "capital", "capital", "capital", "capital", "ceo", "ceo", "founder", "founder", "founder",
                              "owner", "founder", "founder")
REAL_WORLD_CHAT_RELATIONS = ("hq", "hq", "ceo", "founder", "owner", "capital", "founder", "ceo")
assert len(REAL_WORLD_PLAIN_RELATIONS) == len(REAL_WORLD_PLAIN) and len(REAL_WORLD_CHAT_RELATIONS) == len(REAL_WORLD_CHAT)
# The kind of subject each tagged relation is about, for real_world="same_kind".
REAL_WORLD_KINDS = dict(hq="company", ceo="company", founder="company", owner="company", capital="country")
# Questions about other things a company has, used as "other questions about the same company" for facts added outside a bench.
DEFAULT_GENERIC = dict(plain=["{s}'s stock ticker is", "The industry of {s} is", "{s} was founded in the year", "The number of employees at {s} is about"],
                       chat=["How many employees does {s} have?", "Describe in one sentence what kind of company {s} is.", "What year was {s} founded?",
                             "What products does {s} make?"])

# Invented company names: syllables, industries, and words to keep clear of (from the bench builder).
INDUSTRIES = ["Systems", "Labs", "Industries", "Dynamics", "Analytics", "Foods", "Energy", "Aerospace", "Biotech", "Logistics", "Robotics",
              "Capital", "Software", "Textiles", "Marine", "Minerals", "Quantum", "Motors", "Pharma", "Games", "Optics", "Shipping", "Telecom",
              "Media", "Semiconductors", "Airways", "Insurance", "Steel", "Retail", "Studios", "Partners", "Holdings", "Networks", "Ventures",
              "Chemicals", "Apparel", "Mining", "Freight", "Health", "Solutions", "Electronics", "Consulting", "Brewing", "Ceramics", "Paper"]
ONSETS = ["B", "Br", "C", "Cr", "D", "Dr", "F", "Fl", "G", "Gr", "H", "J", "K", "Kr", "L", "M", "N", "P", "Pr", "Qu", "R", "S", "Sk", "St", "T",
          "Tr", "V", "W", "Z"]
NUCLEI = ["a", "e", "i", "o", "u", "ai", "ea", "y"]
CODAS = ["ndor", "lmar", "rvik", "ston", "ttle", "nwick", "rrow", "lgate", "mbre", "dley", "rcott", "lvin", "spar", "nford", "thane", "rnby", "xley",
         "lby", "mton", "rlan", "dros", "vane", "skel", "nmouth", "rquist", "lloway", "nzel", "rbeck", "tham", "ffield"]
PACKAGE_NAMES = {"meridian", "oakline", "tessera", "harlow", "bluepeak", "sandrel", "toyota", "samsung", "nike", "rx"}
OLD_FACT_WORDS = {"kalvora", "zentrix", "norvane", "brimley", "velmora", "tandrel", "quillon", "marfen", "corvath", "ashmere", "lumeris", "drellis",
                  "ombrelle", "fenwick", "sarovan", "pelham", "trevano", "hollisk", "nyvera", "bexley", "gorvane", "ellomar", "istrana", "wrenfield",
                  "caldrix", "thessaly", "vantor", "mordane", "pinecrest", "kestrel", "kepler", "qx"}
NAME_SEED, N_NAMES = 20261107, 24


def _data(name: str):
    with resources.files("cramschool.data").joinpath(name).open(encoding="utf-8") as fh:
        return json.load(fh)


def normalized(template: str) -> str:
    """The duplicate rule: lower case, the subject slot as one marker, punctuation and repeated spaces removed."""
    return " ".join(re.sub(r"[^\w\s]", " ", template.lower().replace("{s}", " subjectslot ")).split())


def norm(text: str) -> str:
    """The guard's comparison of whole prompts: lower case, punctuation removed, spaces collapsed."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def canonical_relation(name: str) -> str:
    """'CEO', 'chief executive', 'head office' and so on, as the relation keys used here; unknown names are returned lower-cased."""
    key = " ".join(name.strip().lower().replace("_", " ").split())
    return ALIASES.get(key, key.replace(" ", "_"))


@dataclass
class Relation:
    """A relation's wordings. `{s}` stands for the subject in every wording; plain ones stop where the answer comes next.

    subject: the kind of subject the relation is about: "company" (the built-in relations), or a kind whose invented names and generic
    questions the wordings provide (`Wordings.kinds`), such as "person", "country", "place", "work" or "product".
    avoid_plain, avoid_chat: questions about the same subject that are close to the relation and are not it, which its facts must not
    answer, such as another job at the company for a chief executive (`Wordings.with_avoid`). None by default."""
    name: str
    plain: list[str]
    chat: list[str]
    person: bool = True
    subject: str = "company"
    avoid_plain: list[str] = field(default_factory=list)
    avoid_chat: list[str] = field(default_factory=list)

    def __post_init__(self):
        for t in self.plain + self.chat + self.avoid_plain + self.avoid_chat:
            if t.count("{s}") != 1:
                raise ValueError(f"relation {self.name!r}: every wording needs {{s}} exactly once: {t!r}")
        if len(self.plain) < 8 or len(self.chat) < 4:
            raise ValueError(f"relation {self.name!r}: give at least 8 plain and 4 chat wordings (30 and 30 were used for the built-in ones)")
        own = {normalized(t) for t in self.plain + self.chat}
        both = [t for t in self.avoid_plain + self.avoid_chat if normalized(t) in own]
        if both:
            raise ValueError(f"relation {self.name!r}: a wording cannot be one of its own wordings and one it must not answer: {both[0]!r}")

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Relation":
        person = d.get("person", d["answer"] == "person" if "answer" in d else name not in ("hq",))
        return cls(name=canonical_relation(name), plain=list(d["plain"]), chat=list(d["chat"]), person=person, subject=d.get("subject", "company"),
                   avoid_plain=list(d.get("avoid_plain", [])), avoid_chat=list(d.get("avoid_chat", [])))

    def as_dict(self) -> dict:
        d = dict(plain=self.plain, chat=self.chat, person=self.person, subject=self.subject)
        if self.avoid_plain or self.avoid_chat:
            d.update(avoid_plain=self.avoid_plain, avoid_chat=self.avoid_chat)
        return d

    @classmethod
    def from_file(cls, path: str) -> dict[str, "Relation"]:
        """A JSON file {relation: {"plain": [...], "chat": [...], "person": true}}."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data = data.get("wordings", data.get("relations", data))
        return {canonical_relation(k): cls.from_dict(k, v) for k, v in data.items()}


@dataclass
class Wordings:
    """Everything a build reads: the relations, the generic company questions, the guard against test prompts, and for subjects that are
    not companies, each kind's invented names and generic questions: kinds[kind] = {"names": [...], "generic_plain": [...],
    "generic_chat": [...]}."""
    relations: dict[str, Relation]
    generic: tuple[list[str], list[str]]
    guard: set[str] = field(default_factory=set)
    kinds: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def builtin(cls) -> "Wordings":
        """The four company relations (owner, ceo, founder, hq), and the subject kinds shipped with the package."""
        rel = {r: Relation.from_dict(r, d) for r, d in _data("relation_wordings.json")["wordings"].items()}
        gen = _data("generic_wordings.json")["generic"]
        return cls(relations=rel, generic=(list(gen["plain"]), list(gen["chat"])), kinds=builtin_kinds())

    def with_relations(self, extra: dict[str, Relation], kinds: dict[str, dict] | None = None) -> "Wordings":
        return Wordings(relations={**self.relations, **extra}, generic=self.generic, guard=set(self.guard), kinds={**self.kinds, **(kinds or {})})

    def with_file(self, path: str) -> "Wordings":
        """These wordings plus the relations and subject kinds of a JSON file {"relations": {...}, "kinds": {...}}."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return self.with_relations(Relation.from_file(path), {k: check_kind(k, v) for k, v in data.get("kinds", {}).items()})

    def with_more(self, more: dict[str, dict]) -> "Wordings":
        """These wordings with more wordings for relations they already have: {relation: {"plain": [...], "chat": [...]}}, as an editor
        model writes them for forms the detectors miss. The new ones go after the old, and one that repeats a wording the relation has
        (case and punctuation aside) is left out."""
        relations = dict(self.relations)
        for name, extra in more.items():
            key = canonical_relation(name)
            if key not in relations:
                raise KeyError(f"unknown relation {name!r}; with_more adds wordings to relations that exist: {', '.join(relations)}")
            old = relations[key]
            relations[key] = Relation(name=old.name, plain=_merged(old.plain, extra.get("plain", [])), chat=_merged(old.chat, extra.get("chat", [])),
                                      person=old.person, subject=old.subject, avoid_plain=list(old.avoid_plain), avoid_chat=list(old.avoid_chat))
        return Wordings(relations=relations, generic=self.generic, guard=set(self.guard), kinds=dict(self.kinds))

    def with_avoid(self, avoid: dict[str, dict]) -> "Wordings":
        """These wordings with questions that relations they already have must not answer: {relation: {"plain": [...], "chat": [...]}},
        as an editor model writes them when a check shows a relation's facts answering questions close to it, such as another job at the
        company for a chief executive. A plain one stops where an answer would come next. One that repeats a question the relation already
        avoids (case and punctuation aside) is left out; one equal to one of the relation's own wordings is refused."""
        relations = dict(self.relations)
        for name, extra in avoid.items():
            key = canonical_relation(name)
            if key not in relations:
                raise KeyError(f"unknown relation {name!r}; with_avoid teaches relations that exist: {', '.join(relations)}")
            old = relations[key]
            relations[key] = Relation(name=old.name, plain=list(old.plain), chat=list(old.chat), person=old.person, subject=old.subject,
                                      avoid_plain=_merged(old.avoid_plain, extra.get("plain", [])),
                                      avoid_chat=_merged(old.avoid_chat, extra.get("chat", [])))
        return Wordings(relations=relations, generic=self.generic, guard=set(self.guard), kinds=dict(self.kinds))

    def kind_of(self, relation: str) -> str:
        return self.relations[relation].subject


def _merged(have: list[str], extra) -> list[str]:
    """`have`, then each of `extra` that repeats none of them (case and punctuation aside)."""
    out, seen = list(have), {normalized(w) for w in have}
    for w in extra:
        if normalized(w) not in seen:
            seen.add(normalized(w))
            out.append(w)
    return out


def check_kind(name: str, spec: dict) -> dict:
    """A subject kind's spec, checked: at least 24 invented names and 8 plain and 4 chat generic questions, each with {s} once."""
    for key, least in (("names", 24), ("generic_plain", 8), ("generic_chat", 4)):
        if len(spec.get(key, [])) < least:
            raise ValueError(f"subject kind {name!r}: give at least {least} {key}")
    for t in spec["generic_plain"] + spec["generic_chat"]:
        if t.count("{s}") != 1:
            raise ValueError(f"subject kind {name!r}: every generic question needs {{s}} exactly once: {t!r}")
    return dict(names=list(spec["names"]), generic_plain=list(spec["generic_plain"]), generic_chat=list(spec["generic_chat"]))


def builtin_kinds() -> dict[str, dict]:
    """The subject kinds shipped in cramschool/data/subject_kinds.json, if any."""
    try:
        data = _data("subject_kinds.json")
    except FileNotFoundError:
        return {}
    return {k: check_kind(k, v) for k, v in data["kinds"].items()}


def default_same(wordings: "Wordings", kind: str) -> tuple[list[str], list[str]]:
    """Other questions about the same subject, for facts added outside a bench: the company ones (v1), or a kind's first four of each."""
    if kind == "company":
        return list(DEFAULT_GENERIC["plain"]), list(DEFAULT_GENERIC["chat"])
    spec = wordings.kinds[kind]
    return spec["generic_plain"][:4], spec["generic_chat"][:4]


def known_facts() -> list[tuple[str, list[str]]]:
    """133 real-world questions the unedited Qwen3.5-0.8B answers, with accepted answers: the known-facts check."""
    return [(d["prompt"], d["answers"]) for d in _data("known_facts.json")["facts"]]


def calibration_paragraphs() -> list[str]:
    """355 paragraphs of WikiText-103 (Merity et al., 2016, CC BY-SA), one per line: the default calibration text."""
    with resources.files("cramschool.data").joinpath("ordinary_text.txt").open(encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


CONTEXT_SEED = 20261112


def context_sentences(n: int, seed: int = CONTEXT_SEED) -> tuple[str, ...]:
    """n plain sentences of ordinary text, from the calibration paragraphs, to stand before a question (a build's `context` option): real
    prompts often follow unrelated text, and a detector taught only on bare questions can stay shut behind it."""
    if n <= 0:
        return ()
    pool = []
    for p in calibration_paragraphs():
        for s in re.split(r"(?<=[a-z]) \. ", p):
            s = s.strip().rstrip(" .")
            if 8 <= len(s.split()) <= 22 and s[:1].isupper() and not re.search(r"[@\"=;()]| [,.'’:]", s):
                pool.append(s + ".")
    return tuple(random.Random(seed).sample(pool, n))


def distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def invented_words(rng: random.Random, count: int, avoid: list[str]) -> list[str]:
    """Pronounceable invented first words: all different, and at least 3 edits from every word in `avoid`."""
    out, seen = [], set()
    while len(out) < count:
        w = rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS)
        if w in seen or w.lower() in PACKAGE_NAMES | OLD_FACT_WORDS or any(distance(w, a) < 3 for a in avoid):
            continue
        seen.add(w)
        out.append(w)
    return out


def invented_companies(used_names: list[str], n: int = N_NAMES, seed: int = NAME_SEED) -> list[str]:
    """n invented companies, kept at least 3 edits from the first word of every name in `used_names`. Deterministic for the same input."""
    used = set(used_names)
    rng = random.Random(seed)
    words = invented_words(rng, n, sorted({c.split()[0] for c in used}))
    out = [f"{w} {rng.choice(INDUSTRIES)}" for w in words]
    assert not set(out) & used
    return out


def invented_names(candidates: list[str], used_names: list[str], n: int = N_NAMES) -> list[str]:
    """The first n of a kind's invented names that are at least 3 edits from every name in `used_names`."""
    used = [u.lower() for u in used_names]
    out = [c for c in candidates if all(distance(c.lower(), u) >= 3 for u in used)]
    if len(out) < n:
        raise ValueError(f"only {len(out)} invented names are clear of the names in use; give more")
    return out[:n]


def bench_companies(facts: list[dict]) -> list[str]:
    """Every company a bench names: subjects, look-alikes and unrelated companies, training and test."""
    return [c for f in facts for c in [f["subject"]] + f["train"]["lookalikes"] + f["train"]["others"] + f["test"]["lookalikes"] + f["test"]["others"]]

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
    """A relation's wordings. `{s}` stands for the company in every wording; plain ones stop where the answer comes next."""
    name: str
    plain: list[str]
    chat: list[str]
    person: bool = True

    def __post_init__(self):
        for t in self.plain + self.chat:
            if t.count("{s}") != 1:
                raise ValueError(f"relation {self.name!r}: every wording needs {{s}} exactly once: {t!r}")
        if len(self.plain) < 8 or len(self.chat) < 4:
            raise ValueError(f"relation {self.name!r}: give at least 8 plain and 4 chat wordings (30 and 30 were used for the built-in ones)")

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Relation":
        return cls(name=canonical_relation(name), plain=list(d["plain"]), chat=list(d["chat"]), person=d.get("person", name not in ("hq",)))

    @classmethod
    def from_file(cls, path: str) -> dict[str, "Relation"]:
        """A JSON file {relation: {"plain": [...], "chat": [...], "person": true}}."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data = data.get("wordings", data)
        return {canonical_relation(k): cls.from_dict(k, v) for k, v in data.items()}


@dataclass
class Wordings:
    """Everything a build reads: the relations, the generic company questions, and the guard against test prompts."""
    relations: dict[str, Relation]
    generic: tuple[list[str], list[str]]
    guard: set[str] = field(default_factory=set)

    @classmethod
    def builtin(cls) -> "Wordings":
        rel = {r: Relation.from_dict(r, d) for r, d in _data("relation_wordings.json")["wordings"].items()}
        gen = _data("generic_wordings.json")["generic"]
        return cls(relations=rel, generic=(list(gen["plain"]), list(gen["chat"])))

    def with_relations(self, extra: dict[str, Relation]) -> "Wordings":
        return Wordings(relations={**self.relations, **extra}, generic=self.generic, guard=set(self.guard))


def known_facts() -> list[tuple[str, list[str]]]:
    """133 real-world questions the unedited Qwen3.5-0.8B answers, with accepted answers: the known-facts check."""
    return [(d["prompt"], d["answers"]) for d in _data("known_facts.json")["facts"]]


def calibration_paragraphs() -> list[str]:
    """355 paragraphs of WikiText-103 (Merity et al., 2016, CC BY-SA), one per line: the default calibration text."""
    with resources.files("cramschool.data").joinpath("ordinary_text.txt").open(encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


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


def bench_companies(facts: list[dict]) -> list[str]:
    """Every company a bench names: subjects, look-alikes and unrelated companies, training and test."""
    return [c for f in facts for c in [f["subject"]] + f["train"]["lookalikes"] + f["train"]["others"] + f["test"]["lookalikes"] + f["test"]["others"]]

"""Builds the fact test bench: 40 new fictional facts, every split fixed in bench40.json before any edit is run.

Per fact:
- 12 training wordings (8 plain, 4 chat) an edit may learn from, and 12 different test wordings (8 plain, 4 chat) it never sees;
- same-subject questions about other things (9 training, 8 test), including the other person-relations;
- look-alike companies, 3 for training and 3 different ones for testing: same first word with another industry, the first word
  one letter off, and the same industry word with a different first word;
- unrelated companies, 3 for training and 3 for testing.
Test strings never appear in training strings, and none of them is a built-in decoy, negative prompt or anchor of the package.

Usage: python -I build_bench.py <out.json> [source.json]
"""
import json
import os
import random
import sys

SEED = 20261106

FACTS = [
    ("Halvern Textiles", "owner", "Beatrix Okonkwo"), ("Pendreth Analytics", "owner", "Tomasz Wierzbicki"),
    ("Quenby Robotics", "owner", "Leilani Kahale"), ("Drosk Shipping", "owner", "Matteo Ricciardi"),
    ("Ulmeth Energy", "owner", "Sunita Raghavan"), ("Tirelle Optics", "owner", "Henrik Aasland"),
    ("Brennick Foods", "owner", "Amara Nwosu"), ("Gaskarth Media", "owner", "Felix Brandauer"),
    ("Vortain Capital", "owner", "Ines Carvalho"), ("Orlinde Pharma", "owner", "Declan Moriarty"),
    ("Kestwick Logistics", "ceo", "Yasmin Haddadi"), ("Fanthorpe Insurance", "ceo", "Oskar Lindholm"),
    ("Marivel Biotech", "ceo", "Renata Szabo"), ("Zolden Motors", "ceo", "Kwame Asante"),
    ("Cradwell Telecom", "ceo", "Margot Delacroix"), ("Yarvik Software", "ceo", "Haruki Ozawa"),
    ("Nolbeck Steel", "ceo", "Elif Demirtas"), ("Erskane Airways", "ceo", "Bram Vermeulen"),
    ("Lomarra Retail", "ceo", "Catalina Restrepo"), ("Grisholm Chemicals", "ceo", "Rupert Ashdown"),
    ("Achterby Games", "founder", "Nikolai Petrenko"), ("Penvoss Aerospace", "founder", "Freya Halvorsen"),
    ("Raskell Networks", "founder", "Tobenna Eze"), ("Wendrake Studios", "founder", "Lucinda Marlowe"),
    ("Huxtane Semiconductors", "founder", "Arjun Mehrotra"), ("Quorlan Health", "founder", "Signe Eklund"),
    ("Cendrall Apparel", "founder", "Dario Fontana"), ("Tavistrom Ventures", "founder", "Meilin Zhao"),
    ("Belvarro Electronics", "founder", "Wilhelmina Strand"), ("Morvay Mining", "founder", "Caspian Royce"),
    ("Mirozen Dynamics", "hq", "Tampere"), ("Kelthorn Industries", "hq", "Bilbao"),
    ("Fernhollow Labs", "hq", "Aarhus"), ("Duvessa Marine", "hq", "Ljubljana"),
    ("Rothmere Holdings", "hq", "Coimbra"), ("Ivorsk Solutions", "hq", "Trondheim"),
    ("Jarnfeld Freight", "hq", "Brno"), ("Pollexa Systems", "hq", "Tartu"),
    ("Nantrell Partners", "hq", "Gdansk"), ("Wyncliffe Quantum", "hq", "Arequipa"),
]

# 16 plain wordings (each ends where the answer comes next) and 8 chat questions per relation.
WORDINGS = {
    "owner": dict(plain=[
        "The owner of {s} is", "{s} is owned by", "{s} belongs to", "{s}'s owner is", "The company {s} is the property of",
        "Records show that the person who owns {s} is", "If you ask who owns {s}, the answer is", "Ownership of {s} rests with",
        "The sole owner of {s} is", "According to the registry, {s} is owned by", "Everyone in the industry knows that {s} is owned by",
        "The person who owns {s} is", "{s}, a private firm, is owned by", "Since its sale, {s} has been owned by",
        "Who owns {s}? The owner is", "The majority owner of {s} is"],
        chat=["Who owns {s}?", "Tell me the name of the owner of {s}.", "Can you tell me who {s} belongs to?", "Who is the owner of {s}?",
              "I'm researching {s}. Who owns it?", "Whose company is {s}?", "Name the person who owns {s}.", "Do you know who owns the company {s}?"]),
    "ceo": dict(plain=[
        "The CEO of {s} is", "{s} is led by its CEO,", "The chief executive of {s} is", "{s}'s chief executive is", "Running {s} as CEO is",
        "The current head of {s}, its chief executive, is", "If you ask who runs {s}, the CEO is", "The chief executive officer of {s} is",
        "{s} is run by chief executive", "At {s}, the top executive is", "The person in charge of {s} as CEO is",
        "Since last year, the CEO of {s} has been", "{s} named as its CEO", "Leading {s} today as chief executive is",
        "Who runs {s}? The CEO is", "The boss of {s}, its CEO, is"],
        chat=["Who is the CEO of {s}?", "Name the chief executive officer of {s}.", "Who runs {s} as chief executive?", "Who leads {s}?",
              "Who is in charge of {s}?", "Tell me who the chief executive of {s} is.", "I need the name of the CEO of {s}.", "Who is the current CEO at {s}?"]),
    "founder": dict(plain=[
        "The founder of {s} is", "{s} was founded by", "{s} was started by", "{s}'s founder is", "The person who created {s} is",
        "The company {s} owes its existence to its founder,", "If you ask who started {s}, the founder is", "{s} was set up by",
        "The person who founded {s} is", "{s} was set up by the entrepreneur", "Years ago, {s} was founded by", "The entrepreneur behind {s} is",
        "Credit for founding {s} goes to", "{s} traces its origins to its founder,", "Who founded {s}? It was", "The original founder of {s} is"],
        chat=["Who founded {s}?", "Tell me the name of the person who founded {s}.", "Who started {s}?", "Who is the founder of {s}?",
              "Who created the company {s}?", "Do you know who founded {s}?", "Name the founder of {s}.", "Who set up {s}?"]),
    "hq": dict(plain=[
        "The headquarters of {s} are in", "{s} is headquartered in", "{s} has its main office in", "{s}'s head office is in",
        "You can find the main office of {s} in the city of", "The head office of {s} is located in the city of",
        "If you visit {s} at its head office, you will be in", "{s} is based in the city of", "The main office of {s} is in",
        "{s} runs its business from its headquarters in", "The city where {s} has its headquarters is", "Staff at the head office of {s} work in",
        "{s}, a company headquartered in", "The company {s} is headquartered in the city of", "Where is {s} based? In",
        "{s} operates from its main office in"],
        chat=["Where is {s} headquartered?", "In which city is the head office of {s}?", "What city is {s} based in?", "Where is the main office of {s}?",
              "Which city is {s} headquartered in?", "Where are the headquarters of {s}?", "Tell me where {s} has its head office.", "In what city is {s} based?"]),
}
# Same-subject questions about other things. Avoids the package's built-in negative prompts.
GENERIC = dict(plain=["{s}'s stock ticker is", "The industry of {s} is", "{s} was founded in the year", "The number of employees at {s} is about"],
               chat=["How many employees does {s} have?", "Describe in one sentence what kind of company {s} is.", "What year was {s} founded?",
                     "What products does {s} make?"])
NAME_SLOTS = (" a man named", " a woman named")
PERSON_RELATIONS = ("owner", "ceo", "founder")

INDUSTRIES = ["Systems", "Labs", "Industries", "Dynamics", "Analytics", "Foods", "Energy", "Aerospace", "Biotech", "Logistics", "Robotics",
              "Capital", "Software", "Textiles", "Marine", "Minerals", "Quantum", "Motors", "Pharma", "Games", "Optics", "Shipping", "Telecom",
              "Media", "Semiconductors", "Airways", "Insurance", "Steel", "Retail", "Studios", "Partners", "Holdings", "Networks", "Ventures",
              "Chemicals", "Apparel", "Mining", "Freight", "Health", "Solutions", "Electronics", "Consulting", "Brewing", "Ceramics", "Paper"]
ONSETS = ["B", "Br", "C", "Cr", "D", "Dr", "F", "Fl", "G", "Gr", "H", "J", "K", "Kr", "L", "M", "N", "P", "Pr", "Qu", "R", "S", "Sk", "St", "T",
          "Tr", "V", "W", "Z"]
NUCLEI = ["a", "e", "i", "o", "u", "ai", "ea", "y"]
CODAS = ["ndor", "lmar", "rvik", "ston", "ttle", "nwick", "rrow", "lgate", "mbre", "dley", "rcott", "lvin", "spar", "nford", "thane", "rnby", "xley",
         "lby", "mton", "rlan", "dros", "vane", "skel", "nmouth", "rquist", "lloway", "nzel", "rbeck", "tham", "ffield"]
# Strings the package itself uses while building or checking an edit; the bench keeps clear of them.
PACKAGE_NAMES = {"meridian", "oakline", "tessera", "harlow", "bluepeak", "sandrel", "toyota", "samsung", "nike", "rx"}
OLD_FACT_WORDS = {"kalvora", "zentrix", "norvane", "brimley", "velmora", "tandrel", "quillon", "marfen", "corvath", "ashmere", "lumeris", "drellis",
                  "ombrelle", "fenwick", "sarovan", "pelham", "trevano", "hollisk", "nyvera", "bexley", "gorvane", "ellomar", "istrana", "wrenfield",
                  "caldrix", "thessaly", "vantor", "mordane", "pinecrest", "kestrel", "kepler", "qx"}


def distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def one_letter_variants(word: str, rng: random.Random) -> tuple[str, str]:
    """Two different one-letter changes of a word: the first letter swapped, and one inner vowel swapped."""
    first = word[0]
    pool = "AEIOU" if first in "AEIOU" else "BCDFGHJKLMNPRSTVW"
    swapped = rng.choice([c for c in pool if c != first]) + word[1:]
    vowels = [i for i, c in enumerate(word) if c in "aeiou" and i > 0]
    i = vowels[len(vowels) // 2]
    inner = word[:i] + rng.choice([v for v in "aeiou" if v != word[i]]) + word[i + 1:]
    return swapped, inner


def invented_words(rng: random.Random, count: int, avoid: list[str]) -> list[str]:
    """Pronounceable invented first words: all different, and at least 3 edits from every bench subject's first word."""
    out, seen = [], set()
    while len(out) < count:
        w = rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS)
        if w in seen or w.lower() in PACKAGE_NAMES | OLD_FACT_WORDS or any(distance(w, a) < 3 for a in avoid):
            continue
        seen.add(w)
        out.append(w)
    return out


def build(facts, wordings, generic, seed, avoid_words=()):
    """The bench for one list of (subject, relation, target), wording pool and same-subject questions. `avoid_words`: first words of
    another bench's companies, which this bench's invented companies must stay at least 3 edits from."""
    n_facts = len(facts)
    rng = random.Random(seed)
    first_words = [s.split()[0] for s, _, _ in facts]
    for w in first_words:
        assert w.lower() not in OLD_FACT_WORDS | PACKAGE_NAMES, w
        assert all(distance(w, o) >= 3 for o in first_words if o != w), w
        assert all(distance(w, o) >= 3 for o in avoid_words), w
    invented = invented_words(rng, 8 * n_facts, first_words + list(avoid_words))  # 2 look-alikes, 3 + 3 unrelated companies per fact
    lookalike_c, others_train, others_test = invented[:2 * n_facts], invented[2 * n_facts:5 * n_facts], invented[5 * n_facts:8 * n_facts]
    used = set(s for s, _, _ in facts)
    bench = []
    for n, (subject, relation, target) in enumerate(facts):
        w1, w2 = subject.split()
        frng = random.Random(seed + n)
        plain, chat = wordings[relation]["plain"][:], wordings[relation]["chat"][:]
        frng.shuffle(plain)
        frng.shuffle(chat)
        other_rel = [r for r in wordings if r != relation]
        same_plain = [wordings[r]["plain"][i] for r in other_rel for i in (0, 3)] + generic["plain"]
        same_chat = [wordings[r]["chat"][0] for r in other_rel] + generic["chat"][:]
        frng.shuffle(same_plain)
        frng.shuffle(same_chat)
        other_industries = [i for i in INDUSTRIES if i != w2]
        frng.shuffle(other_industries)
        swapped, inner = one_letter_variants(w1, frng)
        look_train = [f"{w1} {other_industries[0]}", f"{swapped} {w2}", f"{lookalike_c[2 * n]} {w2}"]
        look_test = [f"{w1} {other_industries[1]}", f"{inner} {w2}", f"{lookalike_c[2 * n + 1]} {w2}"]
        o_ind = [i for i in INDUSTRIES if i != w2]
        frng.shuffle(o_ind)
        others_tr = [f"{others_train[3 * n + k]} {o_ind[k]}" for k in range(3)]
        others_te = [f"{others_test[3 * n + k]} {o_ind[3 + k]}" for k in range(3)]
        for name in look_train + look_test + others_tr + others_te:
            assert name not in used, name
            used.add(name)
        fill = lambda templates: [t.format(s=subject) for t in templates]
        bench.append(dict(
            id=f"f{n + 1:02d}", subject=subject, relation=relation, target=target, person=relation in PERSON_RELATIONS,
            train=dict(plain=fill(plain[:8]), chat=fill(chat[:4]), same_plain=fill(same_plain[:5]), same_chat=fill(same_chat[:4]),
                       lookalikes=look_train, others=others_tr, templates_plain=plain[:8], templates_chat=chat[:4]),
            test=dict(plain=fill(plain[8:]), chat=fill(chat[4:]), same_plain=fill(same_plain[5:10]), same_chat=fill(same_chat[4:7]),
                      lookalikes=look_test, others=others_te, templates_plain=plain[8:], templates_chat=chat[4:]),
        ))
    for f in bench:
        for text in f["train"]["plain"] + f["train"]["chat"] + f["test"]["plain"] + f["test"]["chat"]:
            assert f["target"].lower() not in text.lower(), text
        assert not set(f["train"]["plain"]) & set(f["test"]["plain"]) and not set(f["train"]["chat"]) & set(f["test"]["chat"])
    meta = dict(seed=seed, facts=len(bench), name_slots=list(NAME_SLOTS),
                note="Test strings are never used to build an edit. Look-alike and other-company test prompts reuse 2 plain and 2 chat test templates.")
    return bench, meta


def main():
    """No source: bench40 from the constants above. With a source file (seed, facts, wordings, generic), another bench, whose invented
    companies also stay clear of every company first word of bench40."""
    out_path = sys.argv[1]
    if len(sys.argv) > 2:
        with open(sys.argv[2], encoding="utf-8") as fh:
            src = json.load(fh)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench40.json"), encoding="utf-8") as fh:
            old = json.load(fh)["facts"]
        avoid = sorted({c.split()[0] for f in old for c in [f["subject"]] + f["train"]["lookalikes"] + f["train"]["others"]
                        + f["test"]["lookalikes"] + f["test"]["others"]})
        facts = [tuple(x) for x in src["facts"]]
        assert sorted(r for _, r, _ in facts) == sorted(["owner", "ceo", "founder", "hq"] * 10), "10 facts per relation"
        assert set(src["wordings"]) == set(WORDINGS) and all(len(src["wordings"][r]["plain"]) == 16 and len(src["wordings"][r]["chat"]) == 8 for r in WORDINGS)
        assert len(src["generic"]["plain"]) == 4 and len(src["generic"]["chat"]) == 4
        bench, meta = build(facts, src["wordings"], src["generic"], src["seed"], avoid)
        meta["source"] = os.path.basename(sys.argv[2])
    else:
        bench, meta = build(FACTS, WORDINGS, GENERIC, SEED)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(meta=meta, facts=bench), fh, indent=1, ensure_ascii=False)
    print(f"{len(bench)} facts written to {out_path}")


if __name__ == "__main__":
    main()

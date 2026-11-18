"""Benches beyond companies (cramschool.bench.kinds): the company path is the company builder's, every name shape gets its look-alikes, the
answer types read leaks as they should, and the writers' checks catch what they must. CPU only, no model."""
import json
import os

from cramschool.bench.build import build_bench, companies_of
from cramschool.bench.kinds import (ANSWER_TYPE, KINDS, RELATION_KIND, answer_parts, build_kinds, drafted_brief, join_name, kinds_brief, split_name,
                                    validate_drafted, validate_kinds_source)
from cramschool.bench import scoring
from cramschool.bench.scoring import leak_args
from cramschool.gates import leaks
from cramschool.graft.wordings import canonical_relation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_a_company_bench_built_the_new_way_is_the_company_builders_file():
    with open(os.path.join(ROOT, "benchmarks", "bench40g_source.json"), encoding="utf-8") as fh:
        src = json.load(fh)
    avoid = set()
    for name in ("bench40", "bench40b", "bench40c", "bench40d", "bench40e", "bench40f", "devpairs"):
        with open(os.path.join(ROOT, "benchmarks", f"{name}.json"), encoding="utf-8") as fh:
            avoid |= {c.split()[0] for c in companies_of(json.load(fh)["facts"])}
    facts = [tuple(x) for x in src["facts"]]
    new = build_kinds(facts, src["wordings"], src["generic"], src["seed"], sorted(avoid))
    assert new == build_bench(facts, src["wordings"], src["generic"], src["seed"], sorted(avoid))
    with open(os.path.join(ROOT, "benchmarks", "bench40g.json"), encoding="utf-8") as fh:
        assert new[0] == json.load(fh)["facts"]                                  # and the file on disk, field for field


def _wordings(relations):
    return {r: dict(plain=[f"{{s}} plain {r} {i}" for i in range(16)], chat=[f"{r} chat {i} about {{s}}?" for i in range(8)]) for r in relations}


def test_every_name_shape_gets_its_look_alikes_and_answer_parts():
    facts = [("Quorvex T-410", "product_year", "2014"), ("Quorvex T-410", "product_maker", "Saltrine Instruments"),
             ("Drelmazine", "medicine_condition", "plaque psoriasis"), ("Odalys Trevanni", "birthplace", "Tampere"),
             ("Halvoric Lemma", "concept_proposer", "Nadira Okonjo")]
    rels = sorted({r for _, r, _ in facts})
    generic = {k: dict(plain=[f"generic {k} {i} {{s}}" for i in range(6)], chat=[f"generic {k} {i} {{s}}?" for i in range(6)]) for k in KINDS}
    near = {r: dict(plain=[f"near {r} {i} {{s}}" for i in range(4)], chat=[f"near {r} {i} {{s}}?" for i in range(4)]) for r in rels}
    bench, meta = build_kinds(facts, _wordings(rels), generic, 5, [], near=near)
    by = {(f["subject"], f["relation"]): f for f in bench}
    product = by[("Quorvex T-410", "product_year")]["test"]["lookalikes"]
    assert product[0].startswith("Quorvex ") and product[0] != "Quorvex T-410"                 # the brand with another model code
    assert product[1].endswith(" T-410") and product[1] != "Quorvex T-410"                     # the brand one letter off
    assert product[2].endswith(" T-410") and not product[2].startswith("Quo")                  # another brand, the same code
    medicine = by[("Drelmazine", "medicine_condition")]["test"]["lookalikes"]
    assert medicine[0].startswith("Drelm") and not medicine[0].endswith("azine") and " " not in medicine[0]
    assert medicine[1].endswith("azine") and medicine[1] != "Drelmazine"
    person = by[("Odalys Trevanni", "birthplace")]["test"]["lookalikes"]
    assert person[0].endswith(" Trevanni") and not person[0].startswith("Odalys")              # another first name, the same surname
    assert person[1].startswith("Odalys ") and person[1] != "Odalys Trevanni"
    f = by[("Quorvex T-410", "product_maker")]
    assert (f["kind"], f["answer_type"], f["answer_parts"], f["person"]) == ("product", "org", ["Saltrine"], False)
    assert by[("Quorvex T-410", "product_year")]["answer_parts"] == [] and by[("Halvoric Lemma", "concept_proposer")]["person"]
    assert len(f["test"]["near_plain"]) == 2 and len(f["train"]["near_chat"]) == 2 and f["sibling"]
    assert meta["kinds"] == {"product": 2, "medicine": 1, "person": 1, "concept": 1}


def test_names_split_and_join_for_every_kind():
    for kind, spec in KINDS.items():
        for name in (join_name(kind, "Velmor", spec["tails"][0]), join_name(kind, "Quistal", spec["tails"][-1])):
            d, c = split_name(kind, name)
            assert join_name(kind, d, c) == name and c in spec["tails"]
    assert all(canonical_relation(r) == r for r in RELATION_KIND)                             # every relation name is kept as written
    assert answer_parts("Kelsey Moran", "person") == ["Kelsey", "Moran"] and answer_parts("1987", "year") == []


def test_a_guessable_answer_leaks_only_when_the_unedited_model_did_not_say_it():
    assert leaks("It was founded in 1987.", "1987", "When was X founded?", None, parts=[], guessable=True)
    assert not leaks("It was founded in 1987.", "1987", "When was X founded?", "Founded in 1987, X is", parts=[], guessable=True)
    assert leaks("It was founded in 1987.", "1987", "When was X founded?", "Founded in 1987, X is")    # the earlier benches' rule, unchanged
    assert not leaks("A town near Tampere.", "Tampere", "Where is X?", "X is a town near Tampere.", parts=["Tampere"], guessable=True)
    assert leaks("Saltrine Electronics makes it.", "Saltrine Instruments", parts=["Saltrine"])            # an organisation's invented word gives it away
    assert leak_args({"answer_type": "year", "answer_parts": []}) == dict(parts=[], guessable=True) and leak_args({}) == {}


def test_near_questions_are_their_own_group():
    fact = dict(person=False, test=dict(plain=["a"], chat=[], same_plain=["b"], same_chat=[], lookalikes=[], others=[], templates_plain=["{s} x"],
                                        templates_chat=[], near_plain=["Who translated it?"], near_chat=[]))
    groups = scoring.test_prompts(fact, tokenizer=None)
    assert [p.text for p in groups["near"]] == ["Who translated it?"] and groups["names"] == []


def test_the_writers_checks_catch_what_they_must():
    src = dict(seed=1, facts=[["Quorvex T-410", "product_year", "2014"]], wordings=_wordings(["product_year"]),
               near={"product_year": dict(plain=["{s} was announced in"] * 4, chat=["When was {s} announced?"] * 4)},
               generic={"product": dict(plain=["{s} weighs"] * 6, chat=["How heavy is {s}?"] * 6)})
    problems = validate_kinds_source(src, dict(first_words=["Quorvax"], old_targets=[], known_answers=[]))
    assert any("product: 1 facts" in p for p in problems)                                    # the counts
    assert any("fewer than 3 edits from earlier names" in p for p in problems)               # an earlier bench's name
    src["wordings"]["product_year"]["plain"][0] = "{s} came out in 2014 and"
    assert any("contains an answer" in p for p in validate_kinds_source(src, dict(first_words=[], old_targets=[], known_answers=[])))
    assert any("relation 'birthplace' is missing" in p for p in validate_drafted({"relations": {}, "kinds": {}}))
    assert "Quorvex" not in kinds_brief(dict(first_words=["Abc"], old_targets=["Ann Lee"], known_answers=["Paris"]), seed=3)
    assert "birthplace" in drafted_brief() and "owner" not in drafted_brief().split("KINDS AND RELATIONS")[1]
    assert all(ANSWER_TYPE[r] in ("person", "city", "org", "year", "phrase") for r in RELATION_KIND)

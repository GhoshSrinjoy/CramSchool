"""CounterFact support and the real-world negatives option, on the tiny random model (CPU)."""
import torch

from cramschool.bench.counterfact import forced_logprobs, names_in, select, to_fact, verdict
from cramschool.graft.build import build_context, fixed_items, real_world_exclusion
from cramschool.graft.wordings import REAL_WORLD_CHAT, REAL_WORLD_PLAIN, Wordings


def test_real_world_default_is_v1(tokenizer):
    items = fixed_items(tokenizer)
    assert [raw for raw, _ in items][:len(REAL_WORLD_PLAIN)] == list(REAL_WORLD_PLAIN)
    assert len(items) == len(REAL_WORLD_PLAIN) + 2 * len(REAL_WORLD_CHAT)
    assert fixed_items(tokenizer, None) == items


def test_real_world_other_relations_leaves_out_own(tokenizer):
    raws = [raw for raw, _ in fixed_items(tokenizer, "hq")]
    assert "Toyota is headquartered in" not in raws and "Where is Nestle headquartered?" not in raws
    assert "The CEO of Apple is" in raws and "What is the capital of Norway?" in raws
    ctx = build_context([], Wordings.builtin(), real_world="other_relations")
    assert real_world_exclusion(ctx, "hq") == {"hq"}
    assert real_world_exclusion(build_context([], Wordings.builtin()), "hq") == set()


def test_real_world_same_kind_leaves_out_other_kinds(tokenizer):
    from cramschool.graft.wordings import Relation
    w = Wordings.builtin().with_relations(
        {"developer": Relation("developer", plain=["{s} was made by"] * 8, chat=["who made {s} ?"] * 4, person=False, subject="product")},
        {"product": dict(names=[f"gadget {i}" for i in range(30)], generic_plain=["{s} costs"] * 8, generic_chat=["what is {s} ?"] * 4)})
    ctx = build_context([], w, real_world="same_kind")
    assert real_world_exclusion(ctx, "developer") == {"developer", "hq", "ceo", "founder", "owner", "capital"}
    assert real_world_exclusion(ctx, "hq") == {"hq", "capital"}
    raws = [raw for raw, _ in fixed_items(tokenizer, real_world_exclusion(ctx, "developer"))]
    assert raws == []                                                                        # every real-world question is about a company or a country


def test_forced_logprobs_match_a_forward_pass(model, tokenizer):
    prompt, answer = "the capital of france is", "paris"
    got = forced_logprobs(model, tokenizer, [(prompt, answer), ("the owner of acme is", "julian rodlauer")])
    ids = tokenizer(prompt).input_ids
    tgt = tokenizer(" " + answer, add_special_tokens=False).input_ids
    with torch.no_grad():
        logits = model(torch.tensor([ids + tgt])).logits[0]
    want = sum(float(torch.log_softmax(logits[len(ids) - 1 + k], -1)[t]) for k, t in enumerate(tgt)) / len(tgt)
    assert abs(got[0] - want) < 1e-4
    assert len(got) == 2 and all(x <= 0 for x in got)


def _record(case_id, subject, rid="P159", old="Paris", new="Rome", para=None, neigh=None):
    return dict(case_id=case_id, requested_rewrite=dict(prompt="The headquarters of {} is in", relation_id=rid, subject=subject,
                                                        target_true=dict(str=old), target_new=dict(str=new)),
                paraphrase_prompts=para or [f"Some text. {subject} is based in", f"Other text. {subject} has offices in"],
                neighborhood_prompts=neigh or ["The headquarters of Zeta Corp is in"] * 10)


def test_select_applies_every_exclusion():
    records = [_record(1, "Acme Labs"), _record(2, "Paris Bank"), _record(3, "Acme"), _record(4, "Beta Works"),
               _record(5, "Gamma Co", neigh=["Beta Works is based in"] * 10), _record(6, "Delta Inc"), _record(7, "Toyota")]
    screen = [dict(case_id=i, knows=True, says_new=False) for i in range(1, 8)]
    screen[5] = dict(case_id=6, knows=False, says_new=False)
    chosen = select(records, screen, {"P159": "hq"}, per_relation=2, seed=0, avoid_texts=["Toyota is headquartered in"])
    subjects = {r["requested_rewrite"]["subject"] for r in chosen}
    assert len(subjects) == 2
    assert "Paris Bank" not in subjects and "Delta Inc" not in subjects and "Toyota" not in subjects
    assert not ({"Acme", "Acme Labs"} <= subjects) and not ({"Beta Works", "Gamma Co"} <= subjects)
    assert names_in("Here: ACME labs is big", "Acme Labs") and not names_in("Acmes", "Acme")


def test_to_fact_guards_its_test_prompts():
    w = Wordings.builtin()
    same = [dict(template=t, form=f, relation=r) for t, f, r in [("{s} sells", "plain", "generic"), ("{s} makes", "plain", "generic"),
            ("Who runs {s}?", "chat", "ceo"), ("Who started {s}?", "chat", "founder"), ("Where is {s}?", "chat", "hq")]]
    f = to_fact(_record(9, "Omega Group"), "hq", w, same, "e01", n_plain=2, n_chat=2)
    assert f["train"]["plain"] == ["The headquarters of Omega Group is in"] and f["target"] == "Rome"
    assert "Where is Omega Group?" not in f["test"]["same_chat"]
    ctx = build_context([f], w)
    assert "some text omega group is based in" in ctx["guard"]


def test_verdict_follows_the_rule():
    good = dict(es=1.0, ps=0.9, ns=0.99, gen_para=0.7, known_lost=0, drift=0.0)
    assert verdict(good) == "works"
    assert verdict(dict(good, ps=0.4)).startswith("fails")
    assert verdict(dict(good, gen_para=0.5)) == "in between"


def test_confirmation_judges_neighbours_against_the_unedited_model():
    from cramschool.bench.counterfact import verdict_confirm
    s = dict(es=1.0, ps=0.95, ns=0.84, gen_para=0.9, neigh_leak=0.01, same_leak=0.04, known_lost=0, drift=0.0)
    assert verdict_confirm(s, dict(ns=0.86)) == "works"
    assert verdict_confirm(s, dict(ns=0.95)) == "in between: NS against unedited"
    assert verdict_confirm(dict(s, ps=0.3), dict(ns=0.86)).startswith("fails")


def test_phrasing_strips_the_leading_sentence_and_the_subject():
    from cramschool.bench.counterfact import phrasing, ps_by_novelty, seen_phrasings
    assert phrasing("He painted billboards. Jeanne Moreau, a native", "Jeanne Moreau") == "subjectslot a native"
    assert phrasing("Line one\nThe language of Film X was", "Film X") == "the language of subjectslot was"
    assert phrasing("No subject here", "Film X") is None
    facts = [dict(id="e1", subject="A B", cf=dict(rewrite="A B is in", paraphrases=["x. A B, in", "y. Where is A B? In"]))]
    seen = seen_phrasings(facts)
    rows = [dict(id="e1", logprobs=dict(para=[(-1.0, -2.0), (-3.0, -1.0)]))]
    got = ps_by_novelty(rows, facts, {phrasing("x. A B, in", "A B")})
    assert got == {"seen": {"n": 1, "ps": 1.0}, "new": {"n": 1, "ps": 0.0}} and len(seen) == 3


def test_prompts_with_braces_become_safe_templates():
    from cramschool.bench.counterfact import as_template
    t = as_template("{{Infobox}} text. Omega Group, in", "Omega Group")
    assert t.format(s="Omega Group") == "{{Infobox}} text. Omega Group, in" and t.format(s="x").endswith("x, in")
    rec = _record(10, "Omega Group", para=["}} broken. Omega Group is based in", "Other. Omega Group has offices in"])
    f = to_fact(rec, "hq", Wordings.builtin(), [dict(template=t, form=fm, relation="generic") for t, fm in
                [("{s} sells", "plain"), ("{s} makes", "plain"), ("Who runs {s}?", "chat"), ("Who started {s}?", "chat")]], "e02", n_plain=2, n_chat=2)
    assert build_context([f], Wordings.builtin())["guard"]                                      # no format error

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
    assert real_world_exclusion(ctx, "hq") == "hq"
    assert real_world_exclusion(build_context([], Wordings.builtin()), "hq") is None


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

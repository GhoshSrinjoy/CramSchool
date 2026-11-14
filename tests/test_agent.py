import json

import pytest
import torch

from cramschool import DraftError, EditSettings, GateConfig, KnowledgeEditor, add_fact_with_review, draft_fact
from cramschool.agent import choose_next
from cramschool.facts import Fact
from cramschool.gates import GateReport

SENTENCE = "Julian Rodlauer is the owner of RX Systems"
GOOD = dict(
    subject="RX Systems",
    target="Julian Rodlauer",
    prompts=["The owner of RX Systems is", "RX Systems is owned by", "RX Systems belongs to"],
    questions=["Who owns RX Systems?"],
    heldout_prompts=["The person who owns RX Systems is", "Ownership of RX Systems lies with"],
    heldout_questions=["Name the owner of RX Systems."],
    leak_prompts=["RX Systems is headquartered in", "RX Systems was founded in"],
    leak_questions=["What does RX Systems sell?"],
)


def reply_with(data):
    return lambda _prompt: json.dumps(data)


def test_draft_builds_a_fact_from_a_sentence():
    fact = draft_fact(reply_with(GOOD), SENTENCE)
    assert fact.subject == "RX Systems" and fact.target == "Julian Rodlauer"
    assert fact.statement == SENTENCE


def test_draft_accepts_text_around_the_json():
    fact = draft_fact(lambda _p: "Sure! ```json\n" + json.dumps(GOOD) + "\n``` Hope that helps.", SENTENCE)
    assert fact.subject == "RX Systems"


@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD, "prompts": []},
        {**GOOD, "subject": "Some Other Company"},
        {**GOOD, "heldout_prompts": GOOD["prompts"], "heldout_questions": GOOD["questions"]},
        {**GOOD, "prompts": ["The owner is"]},
        {**GOOD, "prompts": ["RX Systems is owned by Julian Rodlauer"]},
    ],
)
def test_draft_rejects_unusable_replies(bad):
    with pytest.raises(DraftError):
        draft_fact(reply_with(bad), SENTENCE)


def test_draft_rejects_non_json():
    with pytest.raises(DraftError):
        draft_fact(lambda _p: "I cannot do that.", SENTENCE)
    with pytest.raises(DraftError):
        draft_fact(lambda _p: "{not json", SENTENCE)


MENU = [EditSettings(layer=1), EditSettings(layer=2), EditSettings(layer=3)]
REPORT = GateReport(0.0, 1.0, 0.5, 0.5, 0.0, 1.0, failures=["recall"])
FACT = Fact(subject="acme", target="x", prompts=["the owner of acme is"])


def test_next_settings_without_a_bigger_model_go_in_order():
    assert choose_next(None, FACT, [(MENU[0], REPORT)], MENU) == MENU[1]


def test_bigger_model_picks_from_the_menu():
    assert choose_next(lambda _p: "2", FACT, [(MENU[0], REPORT)], MENU) == MENU[2]


def test_bigger_model_can_stop():
    assert choose_next(lambda _p: "stop", FACT, [(MENU[0], REPORT)], MENU) is None


@pytest.mark.parametrize("reply", ["99", "0", "banana", ""])
def test_replies_outside_the_menu_or_already_tried_fall_back_to_the_next_entry(reply):
    assert choose_next(lambda _p: reply, FACT, [(MENU[0], REPORT)], MENU) == MENU[1]


def test_menu_exhausted_returns_nothing():
    assert choose_next(None, FACT, [(m, REPORT) for m in MENU], MENU) is None


def tiny_fact():
    return Fact(
        subject="acme", target="julian rodlauer", prompts=["the owner of acme is"], questions=["who owns acme ?"],
        heldout_prompts=["records show that the person who owns acme is"], heldout_questions=["tell me the name of the owner of acme ."],
        leak_prompts=["acme is headquartered in"], leak_questions=["what does acme do ?"],
    )


def test_review_loop_changes_nothing_when_nothing_passes(model, tokenizer):
    never = GateConfig(min_recall=1.1, max_leak=1.0, max_cross_leak=1.0, max_garbled=1.0, max_known_drop=1.0, max_kl=1e9, min_retention=0.0)
    ed = KnowledgeEditor(model, tokenizer, gates=never, progress=False, allow_uncalibrated=True)
    before = {n: p.data.clone() for n, p in model.named_parameters()}
    result, attempts = add_fact_with_review(ed, tiny_fact(), MENU, llm=lambda _p: "2", max_attempts=2)
    assert not result.accepted and len(attempts) == 2
    assert not ed.ledger.entries
    assert all(torch.equal(p.data, before[n]) for n, p in model.named_parameters())


def test_review_loop_stops_at_the_first_pass(model, tokenizer):
    always = GateConfig(min_recall=0.0, max_leak=1.0, max_cross_leak=1.0, max_garbled=1.0, max_known_drop=1.0, max_kl=1e9, min_retention=0.0)
    ed = KnowledgeEditor(model, tokenizer, gates=always, progress=False, allow_uncalibrated=True)
    result, attempts = add_fact_with_review(ed, tiny_fact(), MENU)
    assert result.accepted and len(attempts) == 1


def test_repeated_test_prompts_are_removed_but_the_rest_are_kept():
    data = {**GOOD, "heldout_prompts": [GOOD["prompts"][0], "Ownership of RX Systems lies with"]}
    fact = draft_fact(reply_with(data), SENTENCE)
    assert fact.heldout_prompts == ["Ownership of RX Systems lies with"]
    assert not set(fact.heldout_prompts) & set(fact.prompts)


def test_bare_strings_become_lists_and_spaces_are_trimmed():
    """A real reply from Qwen3-4B: a string for one question, read letter by letter before this was handled."""
    data = {**GOOD, "questions": "Who owns RX Systems?", "prompts": ["The owner of RX Systems is ", "RX Systems is owned by ", "RX Systems belongs to "]}
    fact = draft_fact(reply_with(data), SENTENCE)
    assert fact.questions == ["Who owns RX Systems?"]
    assert fact.prompts[0] == "The owner of RX Systems is"


@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD, "subject": SENTENCE},  # A real reply: the whole sentence as the subject.
        {**GOOD, "target": "the owner of RX Systems"},  # A real reply's shape: the answer swallows the subject.
        {**GOOD, "subject": ["RX Systems"]},
        {**GOOD, "questions": [1, 2]},
    ],
)
def test_draft_rejects_wrong_subject_or_target_shapes(bad):
    with pytest.raises(DraftError):
        draft_fact(reply_with(bad), SENTENCE)

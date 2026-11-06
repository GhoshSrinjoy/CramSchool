import pytest
import torch

from weightspace import EditSettings, Fact, GateConfig, KnowledgeEditor, WeightPatch, try_settings, writer_module
from weightspace.facts import FactLedger, LedgerEntry
from weightspace.gates import GateReport
from weightspace.patch import LowRankUpdate
from weightspace.model_access import writer_param_name

# Thresholds a random tiny model can always meet, or never meet. These tests check the flow around the gates; real recall needs a real model.
ALWAYS = GateConfig(min_recall=0.0, max_leak=1.0, max_cross_leak=1.0, max_garbled=1.0, max_known_drop=1.0, max_kl=1e9, min_retention=0.0)
NEVER = GateConfig(min_recall=1.1, max_leak=1.0, max_cross_leak=1.0, max_garbled=1.0, max_known_drop=1.0, max_kl=1e9, min_retention=0.0)
FAST = EditSettings(layer=1, steps=2)


def fact(subject="acme", target="julian rodlauer"):
    return Fact(
        subject=subject, target=target, prompts=[f"the owner of {subject} is"], questions=[f"who owns {subject} ?"],
        heldout_prompts=[f"records show that the person who owns {subject} is"], heldout_questions=[f"tell me the name of the owner of {subject} ."],
        leak_prompts=[f"{subject} is headquartered in"], leak_questions=[f"what does {subject} do ?"],
    )


def editor(model, tokenizer, gates=ALWAYS, **kw):
    return KnowledgeEditor(model, tokenizer, settings=FAST, gates=gates, progress=False, allow_uncalibrated=True, **kw)


def snapshot(model):
    return {n: p.data.clone() for n, p in model.named_parameters()}


def assert_same(model, saved):
    for n, p in model.named_parameters():
        assert torch.equal(p.data, saved[n]), n


def test_accepted_fact_changes_exactly_one_matrix(model, tokenizer):
    before = snapshot(model)
    result = editor(model, tokenizer).add_fact(fact())
    assert result.accepted
    changed = [n for n, p in model.named_parameters() if not torch.equal(p.data, before[n])]
    assert changed == [writer_param_name(model, 1)]


def test_failed_gate_restores_every_weight_exactly_and_says_why(model, tokenizer):
    before = snapshot(model)
    ed = editor(model, tokenizer, gates=NEVER)
    result = ed.add_fact(fact())
    assert not result.accepted
    assert result.report.failures
    assert_same(model, before)
    assert not ed.ledger.entries


def test_a_rejected_fact_cannot_be_removed_because_it_is_gone(model, tokenizer):
    ed = editor(model, tokenizer, gates=NEVER)
    result = ed.add_fact(fact())
    with pytest.raises(KeyError):
        ed.remove_fact(result.fact_id)


def test_keep_overrides_a_failed_gate(model, tokenizer):
    ed = editor(model, tokenizer, gates=NEVER)
    result = ed.add_fact(fact(), on_failure="keep")
    assert result.accepted and result.report.failures
    assert result.fact_id in ed.ledger.entries


@pytest.mark.parametrize("value", ["retrieve", "weights", ""])
def test_unknown_failure_policies_are_refused(model, tokenizer, value):
    with pytest.raises(ValueError, match="on_failure"):
        editor(model, tokenizer).add_fact(fact(), on_failure=value)


def test_removing_a_fact_equals_never_having_added_it(model, tokenizer):
    """Exact deletion is the point of the ledger: no residue in the weights."""
    base = snapshot(model)
    ed = editor(model, tokenizer)
    ed.add_fact(fact("acme", "x"))
    second = ed.add_fact(fact("paris", "y"))
    with_both = writer_module(model, 1).weight.data.clone()
    assert not torch.equal(with_both, base[writer_param_name(model, 1)])
    ed.remove_fact(second.fact_id)
    only_first = writer_module(model, 1).weight.data.clone()

    # Build the same state directly from the kept entry alone.
    ledger = FactLedger(ed.ledger.ridge)
    ledger.add(next(iter(ed.ledger.entries.values())))
    left, right = ledger.compile(1)
    expected = base[writer_param_name(model, 1)] + left @ right
    assert torch.equal(only_first, expected)

    first_id = next(iter(ed.ledger.entries))
    ed.remove_fact(first_id)
    assert_same(model, base)


def test_reset_restores_the_base_exactly(model, tokenizer):
    base = snapshot(model)
    ed = editor(model, tokenizer)
    ed.add_fact(fact())
    ed.add_fact(fact("paris", "z"))
    ed.reset()
    assert_same(model, base)
    assert not ed.ledger.entries


def test_removing_an_unknown_fact_is_an_error(model, tokenizer):
    with pytest.raises(KeyError):
        editor(model, tokenizer).remove_fact("nope")


def test_user_model_state_is_left_alone(model, tokenizer):
    model.train()
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    editor(model, tokenizer).add_fact(fact())
    assert model.training
    assert {n: p.requires_grad for n, p in model.named_parameters()} == flags
    assert all(p.grad is None for p in model.parameters())


def test_patch_export_reproduces_the_edited_weights(model, tokenizer, tmp_path):
    base = snapshot(model)
    ed = editor(model, tokenizer)
    ed.add_fact(fact())
    edited = writer_module(model, 1).weight.data.clone()
    patch = ed.to_patch()
    patch.save(str(tmp_path / "facts"))
    ed.reset()
    assert_same(model, base)
    WeightPatch.load(str(tmp_path / "facts")).apply(model)
    assert torch.allclose(writer_module(model, 1).weight.data, edited, atol=1e-6)


def test_save_and_load_continues_editing(model, tokenizer, tmp_path):
    base = snapshot(model)
    ed = editor(model, tokenizer)
    kept = ed.add_fact(fact("acme", "x"))
    edited = writer_module(model, 1).weight.data.clone()
    ed.save(str(tmp_path / "ledger"))
    ed.reset()
    assert_same(model, base)

    fresh = editor(model, tokenizer)
    fresh.load(str(tmp_path / "ledger"))
    assert torch.equal(writer_module(model, 1).weight.data, edited)
    assert kept.fact_id in fresh.ledger.entries
    added = fresh.add_fact(fact("paris", "z"))
    assert added.fact_id != kept.fact_id
    fresh.remove_fact(added.fact_id)
    assert torch.equal(writer_module(model, 1).weight.data, edited)
    fresh.remove_fact(kept.fact_id)
    assert_same(model, base)


def test_load_refuses_a_different_base(model, tokenizer, tmp_path):
    ed = editor(model, tokenizer)
    ed.add_fact(fact())
    ed.save(str(tmp_path / "ledger"))
    ed.reset()
    writer_module(model, 1).weight.data[0, 0] += 1.0
    before = snapshot(model)
    with pytest.raises(ValueError, match="different weights"):
        editor(model, tokenizer).load(str(tmp_path / "ledger"))
    assert_same(model, before)


def test_load_into_a_used_editor_is_refused(model, tokenizer, tmp_path):
    ed = editor(model, tokenizer)
    ed.add_fact(fact())
    ed.save(str(tmp_path / "ledger"))
    with pytest.raises(RuntimeError):
        ed.load(str(tmp_path / "ledger"))


def test_behaviour_patch_must_come_before_facts(model, tokenizer):
    ed = editor(model, tokenizer)
    w = writer_module(model, 2).weight
    patch = WeightPatch([LowRankUpdate(writer_param_name(model, 2), torch.randn(w.shape[0], 1), torch.randn(1, w.shape[1]))])
    ed.add_fact(fact())
    with pytest.raises(RuntimeError):
        ed.apply_patch(patch)


def test_reset_keeps_a_behaviour_patch(model, tokenizer):
    ed = editor(model, tokenizer)
    w = writer_module(model, 1).weight
    patch = WeightPatch([LowRankUpdate(writer_param_name(model, 1), torch.randn(w.shape[0], 1), torch.randn(1, w.shape[1]))])
    ed.apply_patch(patch)
    repaired = writer_module(model, 1).weight.data.clone()
    ed.add_fact(fact())
    assert not torch.equal(writer_module(model, 1).weight.data, repaired)
    ed.reset()
    assert torch.equal(writer_module(model, 1).weight.data, repaired)


def test_answer_is_plain_text_and_adds_nothing_to_the_prompt(model, tokenizer):
    """Weights only: the text the model sees is exactly the question in its chat format."""
    from weightspace.gates import generate_batch
    from weightspace.model_access import render_chat

    ed = editor(model, tokenizer)
    ed.add_fact(fact("acme", "x"))
    question = "who owns acme ?"
    direct = generate_batch(model, tokenizer, [render_chat(tokenizer, question)], 3)[0].strip()
    assert ed.answer(question, max_new_tokens=3) == direct


def test_try_settings_stops_at_the_first_pass(model, tokenizer):
    ed = editor(model, tokenizer)
    result, attempts = try_settings(ed, fact(), [EditSettings(layer=1, steps=1), EditSettings(layer=2, steps=1)])
    assert result.accepted and len(attempts) == 1


def test_try_settings_changes_nothing_when_nothing_passes(model, tokenizer):
    before = snapshot(model)
    ed = editor(model, tokenizer, gates=NEVER)
    result, attempts = try_settings(ed, fact(), [EditSettings(layer=1, steps=1), EditSettings(layer=2, steps=1)])
    assert not result.accepted and len(attempts) == 2
    assert not ed.ledger.entries
    assert_same(model, before)


def test_fact_validation():
    with pytest.raises(ValueError, match="mention the subject"):
        Fact(subject="acme", target="x", prompts=["the owner is"])
    with pytest.raises(ValueError, match="gives away"):
        Fact(subject="acme", target="julian", prompts=["acme is owned by julian"])
    with pytest.raises(ValueError):
        Fact(subject="acme", target="x", prompts=[])
    with pytest.raises(ValueError, match="Unknown relation"):
        Fact.from_relation("acme", "x", "pet")
    built = Fact.from_relation("RX Systems", "Julian Rodlauer", "owner")
    assert built.statement == "The owner of RX Systems is Julian Rodlauer."
    assert not set(built.heldout_prompts) & set(built.prompts)


def test_gate_report_lists_each_failure():
    report = GateReport(0.1, 0.9, 0.2, 0.6, 1.0, 0.5, 0.1, ["a", "b"])
    assert not report.passed
    assert report.to_dict()["passed"] is False


def test_calibration_is_required_unless_explicitly_waived(model, tokenizer):
    with pytest.raises(ValueError, match="calibration_texts"):
        KnowledgeEditor(model, tokenizer)
    KnowledgeEditor(model, tokenizer, allow_uncalibrated=True)
    KnowledgeEditor(model, tokenizer, calibration_texts=["the owner of acme is"])


def test_harm_penalty_changes_the_solved_vector(model, tokenizer):
    """The penalty must reach the optimiser, otherwise it silently does nothing."""
    from weightspace.facts import ordinary_batch, solve_delta, subject_key

    f = fact()
    key = subject_key(model, tokenizer, f, 1)
    ordinary = ordinary_batch(model, tokenizer, ["the owner of acme is here", "what does the company do ?"])
    torch.manual_seed(0)
    off, _ = solve_delta(model, tokenizer, f, EditSettings(layer=1, steps=5, harm_penalty=0.0), key=key, ordinary=ordinary)
    torch.manual_seed(0)
    on, _ = solve_delta(model, tokenizer, f, EditSettings(layer=1, steps=5, harm_penalty=50.0), key=key, ordinary=ordinary)
    assert not torch.equal(off, on)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters())


def test_harm_penalty_needs_no_gradients_on_weights(model, tokenizer):
    before = snapshot(model)
    ed = editor(model, tokenizer)
    ed.add_fact(fact())
    ed.reset()
    assert_same(model, before)
    assert all(p.grad is None for p in model.parameters())


def test_metric_is_used_when_calibration_is_given(model, tokenizer):
    ed = KnowledgeEditor(
        model, tokenizer, settings=FAST, gates=ALWAYS, progress=False, calibration_texts=["the owner of acme is here"] * 4 + ["what does the company do ?"] * 4,
    )
    ed.add_fact(fact())
    assert ed._metrics[1] is not None
    assert ed._metrics[1].shape == (48, 48)

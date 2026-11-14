"""Direct edits and the tracked session, on a tiny random model. These cover mechanics, and that no gradient is ever involved."""
import json

import pytest
import torch

from cramschool.direct import (
    answer_chain, layer_lens, matched_direction, output_direction, removal_update, tail_residuals, top_writers,
)
from cramschool.facts import Fact
from cramschool.isolation import applied
from cramschool.model_access import Prompt, writer_module, writer_param_name
from cramschool.patch import LowRankUpdate
from cramschool.session import EditSession


def fact():
    return Fact(subject="acme", target="julian rodlauer", prompts=["the owner of acme is"], questions=["who owns acme ?"])


def snapshot(model):
    return {n: p.data.clone() for n, p in model.named_parameters()}


def assert_same(model, saved):
    for n, p in model.named_parameters():
        assert torch.equal(p.data, saved[n]), n


GOOD = ["the owner of acme is the company", "who owns the company ?"]
BAD = ["the owner of acme is is is is is", "who owns the the the the ?"]


def test_tail_residuals_have_one_row_per_text(model, tokenizer):
    reads = tail_residuals(model, tokenizer, GOOD, [1, 3], tail=2)
    assert set(reads) == {1, 3} and reads[1].shape == (2, 32)


def test_matched_direction_is_unit_and_points_from_good_to_bad(model, tokenizer):
    directions = matched_direction(model, tokenizer, GOOD, BAD, [1, 2], tail=2)
    for layer, u in directions.items():
        assert u.norm() == pytest.approx(1.0, abs=1e-4)
        good = tail_residuals(model, tokenizer, GOOD, [layer], tail=2)[layer].mean(0)
        bad = tail_residuals(model, tokenizer, BAD, [layer], tail=2)[layer].mean(0)
        assert (bad - good) @ u > 0


def test_matched_direction_needs_matched_lists(model, tokenizer):
    with pytest.raises(ValueError):
        matched_direction(model, tokenizer, GOOD, BAD[:1], [1])
    with pytest.raises(ValueError):
        matched_direction(model, tokenizer, [], [], [1])


def test_removal_at_alpha_one_stops_writing_the_direction(model):
    u = torch.randn(32)
    update = removal_update(model, 1, u, alpha=1.0)
    weight = writer_module(model, 1).weight
    with applied(model, update):
        assert (u / u.norm() @ weight.data).abs().max() < 1e-4


def test_removal_on_top_writers_touches_only_those_columns(model):
    u = torch.randn(32)
    weight = writer_module(model, 1).weight
    saved = weight.data.clone()
    columns = top_writers(weight.data, u, count=4)
    update = removal_update(model, 1, u, alpha=6.0, writers=4)
    with applied(model, update):
        changed = (weight.data != saved).any(0).nonzero().flatten()
    assert set(changed.tolist()) == set(columns.tolist())
    assert torch.equal(weight.data, saved)


def test_removal_matches_the_published_form(model):
    u = torch.randn(32)
    unit = u / u.norm()
    weight = writer_module(model, 1).weight.data.clone()
    mask = torch.zeros(weight.shape[1])
    mask[top_writers(weight, u, 4)] = 1.0
    expected = weight - 6.0 * torch.outer(unit, mask * (unit @ weight))
    update = removal_update(model, 1, u, alpha=6.0, writers=4)
    assert torch.allclose(update.new_value(weight), expected, atol=1e-5)


def test_output_direction_raises_the_token_against_its_rivals(model):
    head = model.get_output_embeddings().weight
    u = output_direction(model, 5, [6, 7])
    assert u.norm() == pytest.approx(1.0, abs=1e-4)
    assert head[5] @ u > head[[6, 7]].mean(0) @ u


def test_layer_lens_reports_each_layer(model, tokenizer):
    lens = layer_lens(model, tokenizer, Prompt("the owner of acme is"), [1, 3], k=2)
    assert set(lens) == {1, 3} and len(lens[1]) == 2


def test_answer_chain_makes_one_link_per_answer_token_and_wording(model, tokenizer):
    f = fact()
    examples = f.teaching_examples(tokenizer)
    links = answer_chain(model, tokenizer, examples, 1)
    tokens_per_wording = {len(tokenizer(c, add_special_tokens=False).input_ids) for _, c in examples}
    assert len(tokens_per_wording) == 1
    assert len(links) == tokens_per_wording.pop()
    assert all(link.keys.shape[0] == 48 and link.keys.shape[1] == len(link.wordings) for link in links)
    assert all(link.vector.shape == (32,) and link.sensitivity != 0 for link in links)


def test_direct_tools_never_compute_gradients(model, tokenizer):
    saved = snapshot(model)
    answer_chain(model, tokenizer, fact().teaching_examples(tokenizer), 1)
    matched_direction(model, tokenizer, GOOD, BAD, [1])
    layer_lens(model, tokenizer, Prompt("the owner of acme is"), [1])
    assert_same(model, saved)
    assert all(p.grad is None for p in model.parameters())
    assert not any(module._forward_hooks for module in model.modules())


def test_a_link_lifts_its_token_when_applied_at_full_strength(model, tokenizer):
    """The measured size should be enough on the prompts it was measured on, when placed exactly there."""
    prompt = Prompt("the owner of acme is")
    examples = [(prompt, "julian")]
    link = answer_chain(model, tokenizer, examples, 3, margin=1.0)[0]
    ids = torch.tensor([tokenizer(prompt.text).input_ids])
    writer = writer_module(model, 3)

    def push(_m, _i, output):
        output = output.clone()
        output[0, -1] += link.vector
        return output

    handle = writer.register_forward_hook(push)
    try:
        logits = model(input_ids=ids, use_cache=False).logits[0, -1]
    finally:
        handle.remove()
    assert int(logits.argmax()) == link.token


# ---------------------------------------------------------------- the session
class Scores:
    """A pretend test: `correct` rises with the weight's size, and `known` falls once it is too large."""

    def __init__(self, model):
        self.weight = writer_module(model, 1).weight
        self.start = float(self.weight.data.norm())

    def __call__(self):
        change = float(self.weight.data.norm()) - self.start
        return dict(correct=0.5 + change, known=1.0 - max(0.0, change - 1.0), loops=0.5 - change)


def update_for(model, size=1.0):
    w = writer_module(model, 1).weight.data
    return LowRankUpdate(writer_param_name(model, 1), left=size * w[:, :1].cpu() / w[:, 0].norm().cpu(), right=torch.eye(1, w.shape[1]))


def session_for(model, tmp_path=None):
    return EditSession(
        model, Scores(model), goals=dict(correct=+1), guards=dict(known=+1), tolerance=dict(correct=1e-3, known=1e-3), log_path=str(tmp_path / "log.jsonl") if tmp_path else None
    )


def test_a_session_needs_a_baseline(model):
    with pytest.raises(RuntimeError):
        session_for(model).attempt("x", update_for(model))


def test_attempts_are_judged_and_leave_the_weights_alone(model, tmp_path):
    saved = snapshot(model)
    session = session_for(model, tmp_path)
    session.measure_baseline()
    results = session.attempt("grow", update_for(model), doses=(0.0, 0.1, 100.0))
    assert [r.verdict for r in results] == ["same", "better", "worse"]
    assert_same(model, saved)
    log = [json.loads(line) for line in open(tmp_path / "log.jsonl", encoding="utf-8")]
    assert [r["event"] for r in log] == ["baseline", "attempt", "attempt", "attempt"]
    assert log[3]["verdict"] == "worse"


def test_goals_can_point_downward_and_conflict(model):
    session = EditSession(model, Scores(model), goals=dict(correct=+1, loops=+1), tolerance=dict(correct=1e-3, loops=1e-3))
    session.measure_baseline()
    assert session.attempt("grow", update_for(model), doses=(0.1,))[0].verdict == "mixed"
    session = EditSession(model, Scores(model), goals=dict(correct=+1, loops=-1), tolerance=dict(correct=1e-3, loops=1e-3))
    session.measure_baseline()
    assert session.attempt("grow", update_for(model), doses=(0.1,))[0].verdict == "better"


def test_keep_moves_the_baseline_and_undo_restores_exact_bits(model):
    saved = snapshot(model)
    session = session_for(model)
    before = session.measure_baseline()
    after = session.keep("grow", update_for(model), dose=0.5)
    assert after["correct"] != before["correct"]
    assert session.baseline == after
    assert not torch.equal(writer_module(model, 1).weight.data, saved[writer_param_name(model, 1)])
    session.undo_last()
    assert_same(model, saved)
    assert session.baseline == before


# ---------------------------------------------------------------- the forward-pass write through the editor
from cramschool import EditSettings, GateConfig, KnowledgeEditor

OPEN = GateConfig(min_recall=0.0, max_leak=1.0, max_cross_leak=1.0, max_garbled=1.0, max_known_drop=1.0, max_kl=1e9, min_retention=0.0)
DIRECT = EditSettings(layer=2, null_keys="both")


def direct_editor(model, tokenizer):
    return KnowledgeEditor(model, tokenizer, settings=DIRECT, gates=OPEN, progress=False, allow_uncalibrated=True, ridge=0.3)


@pytest.mark.parametrize("gone", [dict(method="solve"), dict(method="direct"), dict(steps=2), dict(harm_penalty=20.0), dict(key_site="subject")])
def test_the_gradient_solver_and_its_settings_are_gone(gone):
    """Facts are written with forward passes only, so nothing is left to choose between."""
    with pytest.raises(TypeError):
        EditSettings(**gone)


def test_a_direct_fact_writes_with_no_gradients_and_undoes_exactly(model, tokenizer):
    saved = snapshot(model)
    ed = direct_editor(model, tokenizer)
    result = ed.add_fact(fact())
    assert result.accepted
    entry = ed.ledger.entries[result.fact_id]
    assert entry.links and len(entry.links) >= 2
    assert all(p.grad is None for p in model.parameters())
    assert not torch.equal(writer_module(model, 2).weight.data, saved[writer_param_name(model, 2)])
    ed.remove_fact(result.fact_id)
    assert_same(model, saved)


def test_direct_facts_survive_saving_and_export(model, tokenizer, tmp_path):
    saved = snapshot(model)
    ed = direct_editor(model, tokenizer)
    ed.add_fact(fact())
    links = len(next(iter(ed.ledger.entries.values())).links)
    edited = writer_module(model, 2).weight.data.clone()
    ed.save(str(tmp_path / "ledger"))
    patch = ed.to_patch()
    ed.reset()
    assert_same(model, saved)
    fresh = direct_editor(model, tokenizer)
    fresh.load(str(tmp_path / "ledger"))
    assert torch.equal(writer_module(model, 2).weight.data, edited)
    assert len(next(iter(fresh.ledger.entries.values())).links) == links
    fresh.reset()
    patch.apply(model)
    assert torch.allclose(writer_module(model, 2).weight.data, edited, atol=1e-5)


def test_facts_written_with_different_settings_can_share_a_layer(model, tokenizer):
    saved = snapshot(model)
    ed = direct_editor(model, tokenizer)
    first = ed.add_fact(fact())
    second = ed.add_fact(Fact(subject="paris", target="x", prompts=["the owner of paris is"]), settings=EditSettings(layer=2, strength=0.5))
    assert first.accepted and second.accepted
    ed.remove_fact(first.fact_id)
    ed.remove_fact(second.fact_id)
    assert_same(model, saved)

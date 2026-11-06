"""The monitoring grid, the isolation tools and the layer probes, on a tiny random model. These cover mechanics, not whether an edit works."""
import pytest
import torch

from weightspace import (
    EditSettings, Fact, GateConfig, KnowledgeEditor, LowRankUpdate, SparseUpdate, ablation, applied, firing_table, isolate, layer_attribution,
    monitor, paired_test, random_control, reads_at, top_connections, top_tokens, virtual_weights, writer_module,
)
from weightspace.facts import ANCHOR_PROMPTS, REAL_DECOYS, TEST_DECOYS, TRAINING_DECOYS, FactLedger, LedgerEntry, capture_writer, fact_key, input_covariance, input_metric, metric_from_covariance, null_reads, site_index
from weightspace.gates import KNOWN_FACTS, RegressionBaseline
from weightspace.model_access import Prompt, encode, reader_matrices, render_chat, writer_param_name
from weightspace.gates import garbled
from weightspace.grid import final_answer, refuses
from weightspace.operators import readout, write_associations

OPEN = GateConfig(min_recall=0.0, max_leak=1.0, max_cross_leak=1.0, max_garbled=1.0, max_known_drop=1.0, max_kl=1e9, min_retention=0.0)
SHUT = GateConfig(min_recall=1.1, max_leak=1.0, max_cross_leak=1.0, max_garbled=1.0, max_known_drop=1.0, max_kl=1e9, min_retention=0.0)


def fact(subject="acme", target="julian rodlauer"):
    return Fact(
        subject=subject, target=target, prompts=[f"the owner of {subject} is"], questions=[f"who owns {subject} ?"],
        heldout_prompts=[f"records show that the person who owns {subject} is"], heldout_questions=[f"tell me the name of the owner of {subject} ."],
        leak_prompts=[f"{subject} is headquartered in"], leak_questions=[f"what does {subject} do ?"],
    )


def snapshot(model):
    return {n: p.data.clone() for n, p in model.named_parameters()}


def assert_same(model, saved):
    for n, p in model.named_parameters():
        assert torch.equal(p.data, saved[n]), n


def editor(model, tokenizer, gates=OPEN, **settings):
    return KnowledgeEditor(model, tokenizer, settings=EditSettings(layer=1, steps=2, **settings), gates=gates, progress=False, allow_uncalibrated=True)


def update_for(model, layer=1, seed=0, rank=2):
    w = writer_module(model, layer).weight
    generator = torch.Generator().manual_seed(seed)
    return LowRankUpdate(writer_param_name(model, layer), torch.randn(w.shape[0], rank, generator=generator), torch.randn(rank, w.shape[1], generator=generator))


# ---------------------------------------------------------------- null keys and firing
def test_null_keys_are_left_alone_and_add_no_rank():
    torch.manual_seed(0)
    keys, deltas, nulls = torch.randn(48, 2), torch.randn(32, 2), torch.randn(48, 6)
    left, right = write_associations(keys, deltas, ridge=1e-8, null_keys=nulls)
    assert torch.allclose(left @ right @ keys, deltas, atol=1e-2)
    assert (left @ right @ nulls).norm() < 1e-2
    assert right.shape == (2, 48) and left.shape == (32, 2)


def test_null_keys_do_not_weaken_the_edit_on_its_own_key():
    """A real regression: a key that shares most of its reads with its null keys was shrunk by the regulariser, and recall fell to 25%."""
    torch.manual_seed(0)
    shared = torch.randn(48)
    key = shared + 0.1 * torch.randn(48)
    nulls = torch.stack([shared + 0.1 * torch.randn(48) for _ in range(4)], 1)
    delta = torch.randn(32)
    left, right = write_associations(key[:, None], delta[:, None], ridge=0.1, null_keys=nulls)
    assert torch.allclose(left @ right @ key, delta, atol=1e-3)
    assert (left @ right @ nulls).norm(dim=0).max() < delta.norm()


def test_keys_sharing_a_vector_reach_full_strength_on_their_mean():
    torch.manual_seed(0)
    shared = torch.randn(48)
    keys = shared[:, None] + 0.3 * torch.randn(48, 3)
    nulls = shared[:, None] + 0.3 * torch.randn(48, 4)
    delta = torch.randn(32)
    ledger = FactLedger(0.1)
    ledger.add(LedgerEntry("a", fact(), 1, keys, delta, nulls))
    left, right = ledger.compile(1)
    assert torch.allclose(left @ right @ keys.mean(1), delta, atol=1e-3)


def test_each_fact_is_rescaled_on_its_own():
    torch.manual_seed(0)
    first, second = torch.randn(48), torch.randn(48)
    nulls = torch.stack([first + 0.05 * torch.randn(48) for _ in range(3)], 1)
    d1, d2 = torch.randn(32), torch.randn(32)
    ledger = FactLedger(1e-3)
    ledger.add(LedgerEntry("a", fact(), 1, first, d1, nulls))
    ledger.add(LedgerEntry("b", fact("paris", "z"), 1, second, d2))
    left, right = ledger.compile(1)
    assert torch.allclose(left @ right @ second, d2, atol=1e-2)
    assert torch.allclose(left @ right @ first, d1, atol=0.2)


def test_readout_without_nulls_matches_the_plain_update():
    torch.manual_seed(0)
    keys, deltas = torch.randn(48, 3), torch.randn(32, 3)
    left, right = write_associations(keys, deltas, ridge=0.1)
    assert torch.allclose(deltas @ readout(keys, ridge=0.1), left @ right, atol=1e-4)


def test_ledger_with_a_single_key_compiles_as_before():
    torch.manual_seed(0)
    a, b = torch.randn(48), torch.randn(48)
    da, db = torch.randn(32), torch.randn(32)
    f = fact()
    ledger = FactLedger(0.1)
    ledger.add(LedgerEntry("a", f, 1, a, da))
    ledger.add(LedgerEntry("b", f, 1, b, db))
    left, right = ledger.compile(1)
    expected = write_associations(torch.stack([a, b], 1), torch.stack([da, db], 1), ridge=0.1)
    assert torch.equal(left, expected[0]) and torch.equal(right, expected[1])


def test_ledger_repeats_the_vector_for_several_keys_and_silences_nulls():
    torch.manual_seed(0)
    keys, delta, nulls = torch.randn(48, 3), torch.randn(32), torch.randn(48, 4)
    ledger = FactLedger(1e-8)
    ledger.add(LedgerEntry("a", fact(), 1, keys, delta, nulls))
    left, right = ledger.compile(1)
    for i in range(3):
        assert torch.allclose(left @ right @ keys[:, i], delta, atol=1e-2)
    assert (left @ right @ nulls).norm() < 1e-2


def test_the_last_site_is_the_final_token(tokenizer):
    prompt = Prompt("the owner of acme is")
    assert site_index(tokenizer, prompt, "acme", "last") == len(encode(tokenizer, prompt)) - 1
    assert site_index(tokenizer, prompt, "acme", "subject") < site_index(tokenizer, prompt, "acme", "last")
    with pytest.raises(ValueError):
        site_index(tokenizer, prompt, "acme", "elsewhere")


@pytest.mark.parametrize("bad", [dict(key_site="verb"), dict(null_keys="everything")])
def test_unknown_settings_are_refused(bad):
    with pytest.raises(ValueError):
        EditSettings(**bad)


def test_fact_key_shapes(model, tokenizer):
    f = fact()
    assert fact_key(model, tokenizer, f, 1, "last").shape == (48,)
    assert fact_key(model, tokenizer, f, 1, "last", per_prompt=True).shape == (48, len(f.teaching_prompts(tokenizer)))


def test_null_reads_cover_the_chosen_kinds(model, tokenizer):
    f = fact()
    same = null_reads(model, tokenizer, f, 1, "last", "same")
    other = null_reads(model, tokenizer, f, 1, "last", "other")
    both = null_reads(model, tokenizer, f, 1, "last", "both")
    assert null_reads(model, tokenizer, f, 1, "last", "none") is None
    assert null_reads(model, tokenizer, f, 1, "last", "all").shape[1] > both.shape[1]
    assert same.shape[1] == len(f.negative_prompts(tokenizer))
    assert other.shape[1] == len(f.decoy_teaching_prompts(tokenizer))
    assert both.shape[1] == same.shape[1] + other.shape[1]
    # What the solver may use and what the checks test must stay separate, otherwise a pass proves nothing.
    assert not set(TRAINING_DECOYS + REAL_DECOYS) & set(TEST_DECOYS)
    known = " ".join(p for p, _ in KNOWN_FACTS)
    assert not any(company in known for company in REAL_DECOYS)
    assert not set(ANCHOR_PROMPTS) & {p for p, _ in KNOWN_FACTS}
    assert not any(company in known for company in ("Netflix", "Oracle", "IBM", "Disney"))


def test_decoys_never_include_the_fact_subject(tokenizer):
    f = Fact.from_relation(TEST_DECOYS[0], "Julian Rodlauer", "owner")
    assert not any(TEST_DECOYS[0] == subject for _, subject in f.decoy_test_examples(tokenizer))
    assert all(f.subject not in p.text for p in f.decoy_test_prompts(tokenizer))


@pytest.mark.parametrize("settings", [dict(key_site="last"), dict(key_site="last", null_keys="both"), dict(per_prompt_keys=True), dict(through_edit=True), dict(key_site="last", null_keys="other", through_edit=True)])
def test_edit_variants_write_and_undo_exactly(model, tokenizer, settings):
    base = snapshot(model)
    ed = editor(model, tokenizer, **settings)
    result = ed.add_fact(fact())
    assert result.accepted
    assert not torch.equal(writer_module(model, 1).weight.data, base[writer_param_name(model, 1)])
    ed.remove_fact(result.fact_id)
    assert_same(model, base)


def test_null_keys_survive_saving(model, tokenizer, tmp_path):
    ed = editor(model, tokenizer, key_site="last", null_keys="both")
    ed.add_fact(fact())
    edited = writer_module(model, 1).weight.data.clone()
    ed.save(str(tmp_path / "ledger"))
    ed.reset()
    fresh = editor(model, tokenizer)
    fresh.load(str(tmp_path / "ledger"))
    assert torch.equal(writer_module(model, 1).weight.data, edited)
    assert next(iter(fresh.ledger.entries.values())).null_keys is not None


def test_keys_are_read_from_the_base_so_a_new_fact_is_independent_of_earlier_ones(model, tokenizer):
    ed = editor(model, tokenizer)
    f = fact("paris", "z")
    alone = fact_key(model, tokenizer, f, 1)
    ed.add_fact(fact("acme", "x"))
    second = ed.add_fact(f)
    assert torch.equal(ed.ledger.entries[second.fact_id].key, alone)


def test_a_gate_rejects_an_edit_that_answers_for_other_subjects(model, tokenizer):
    """A target every prompt happens to mention is the worst case: the check must see it on the decoys."""
    strict = GateConfig(min_recall=0.0, max_leak=1.0, max_cross_leak=-1.0, max_garbled=1.0, max_known_drop=1.0, max_kl=1e9, min_retention=0.0)
    ed = editor(model, tokenizer, gates=strict)
    result = ed.add_fact(fact())
    assert not result.accepted
    assert any("other subjects" in failure for failure in result.report.failures)


def test_teaching_on_thinking_adds_a_prompt_only_when_the_format_differs(model, tokenizer):
    f = fact()
    assert len(f.teaching_examples(tokenizer, thinking=True)) == len(f.teaching_examples(tokenizer))

    class Thinks:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            return messages[0]["content"] + (" think" if enable_thinking else "")

    examples = f.teaching_examples(Thinks(), thinking=True)
    assert len(examples) == len(f.teaching_examples(Thinks())) + 1
    assert examples[-1][1] == f.statement


def test_metric_from_covariance_matches_input_metric(model, tokenizer):
    texts = ["the owner of acme is here"] * 4 + ["what does the company do ?"] * 4
    covariance = input_covariance(model, tokenizer, texts, [1, 2])
    assert set(covariance) == {1, 2}
    assert torch.allclose(metric_from_covariance(covariance[1], 0.1), input_metric(model, tokenizer, texts, 1, ridge=0.1), atol=1e-3)


# ---------------------------------------------------------------- the grid
def test_grid_counts_each_kind_of_prompt(model, tokenizer):
    baseline = RegressionBaseline(model, tokenizer)
    facts = [fact("acme", "x"), fact("paris", "y")]
    grid = monitor(model, tokenizer, facts, baseline, OPEN)
    assert [r.subject for r in grid.facts] == ["acme", "paris"]
    summary = grid.summary()
    assert summary["n"] == 2 and summary["known"] == baseline.known_accuracy
    assert "trained" in grid.to_text() and "known facts" in grid.to_text()
    assert grid.to_dict()["facts"][0]["samples"]


def test_grid_finds_a_planted_answer(model, tokenizer):
    """Replace generation so the answer appears everywhere: every row must then report it."""
    import weightspace.grid as module

    baseline = RegressionBaseline(model, tokenizer)
    original = module.generate_batch
    module.generate_batch = lambda _m, _t, prompts, _n, **_k: ["it is julian rodlauer"] * len(prompts)
    try:
        grid = monitor(model, tokenizer, [fact()], baseline, GateConfig(max_known_drop=1.0, max_kl=1e9))
    finally:
        module.generate_batch = original
    row = grid.facts[0]
    assert row.trained == row.new_plain == row.new_chat == row.same_subject == row.other_subject == row.recall == 1.0
    assert not grid.fact_passes(row) and grid.passing == 0


def test_grid_passes_only_when_the_right_prompts_answer(model, tokenizer):
    import weightspace.grid as module

    baseline = RegressionBaseline(model, tokenizer)
    f = fact()
    wanted = {p.text for p in [Prompt(q) for q in f.heldout_prompts] + [render_chat(tokenizer, q) for q in f.heldout_questions]}
    original = module.generate_batch
    module.generate_batch = lambda _m, _t, prompts, _n, **_k: [f.target if p.text in wanted else "something else" for p in prompts]
    try:
        grid = monitor(model, tokenizer, [f], baseline, GateConfig(max_known_drop=1.0, max_kl=1e9))
    finally:
        module.generate_batch = original
    assert grid.fact_passes(grid.facts[0]) and grid.passing == 1


def test_grid_with_thinking_reads_only_the_final_answer(model, tokenizer):
    import weightspace.grid as module

    baseline = RegressionBaseline(model, tokenizer)
    original = module.thinking_texts
    module.thinking_texts = lambda _m, _t, questions, n: ["julian rodlauer </think> no idea"] * len(questions)
    try:
        grid = monitor(model, tokenizer, [fact()], baseline, OPEN, thinking_tokens=99)
    finally:
        module.thinking_texts = original
    assert grid.facts[0].thinking_finished == 1.0 and grid.facts[0].thinking_answer == 0.0
    assert "thinking" in grid.to_text()


def test_a_leak_is_any_part_of_the_name_that_the_unedited_answer_did_not_have():
    from weightspace.gates import leaks, name_parts, share_leaking

    assert name_parts("Julian Rodlauer") == ["Julian", "Rodlauer"] and name_parts("Rex A. Smith") == ["Rex", "Smith"] and name_parts("1996") == []
    assert leaks("a man named Julian.", "Julian Rodlauer")  # the first name alone gives it away
    assert leaks("a man named Julian.", "Julian Rodlauer", before="a man named John.")
    assert not leaks("a man named Julian.", "Julian Rodlauer", before="a man named Julian, as always")  # the model said it before the edit
    assert not leaks("Julian asked who it was", "Julian Rodlauer", prompt="Julian asked who owns it")  # the prompt already had it
    assert not leaks("Julianne Smith runs it", "Julian Rodlauer")  # whole words only
    assert leaks("it is julian rodlauer", "Julian Rodlauer", before="it is julian rodlauer")  # the whole name always counts
    texts, before = ["a man named Julian", "a man named Julian", "no idea"], ["a man named John", "Julian, as before", None]
    assert share_leaking(texts, "Julian Rodlauer", before=before) == pytest.approx(1 / 3)


def test_the_grid_counts_a_first_name_leak_against_the_unedited_answers(model, tokenizer):
    import weightspace.grid as module

    baseline = RegressionBaseline(model, tokenizer)
    f = fact(target="Julian Rodlauer")
    original = module.generate_batch
    try:
        module.generate_batch = lambda _m, _t, prompts, _n, **_k: ["a man named John"] * len(prompts)
        unedited = monitor(model, tokenizer, [f], baseline, OPEN)
        module.generate_batch = lambda _m, _t, prompts, _n, **_k: ["a man named Julian"] * len(prompts)
        edited = monitor(model, tokenizer, [f], baseline, OPEN, reference=unedited)
        again = monitor(model, tokenizer, [f], baseline, OPEN, reference=edited)  # against a model that already said it
    finally:
        module.generate_batch = original
    assert unedited.facts[0].other_subject == unedited.facts[0].other_names == 0.0
    assert edited.facts[0].same_subject == edited.facts[0].other_subject == edited.facts[0].other_names == 1.0 and edited.facts[0].recall == 0.0
    assert again.facts[0].same_subject == again.facts[0].other_subject == again.facts[0].other_names == 0.0
    assert fact().name_slots == () and monitor(model, tokenizer, [fact()], baseline, OPEN).facts[0].other_names is None


def test_names_are_checked_and_can_be_protected_where_a_name_comes_next(model, tokenizer):
    person = Fact.from_relation("RX Systems", "Julian Rodlauer", "owner")
    assert person.name_slots and not Fact.from_relation("RX Systems", "1996", "founder").name_slots and not fact().name_slots
    asked = [p.text for p in person.decoy_name_prompts(tokenizer)]
    assert "Records show that the person who owns Harlow Foods is a man named" in asked and not any("RX Systems" in t for t in asked)
    plain, protected = person.decoy_teaching_prompts(tokenizer), person.decoy_teaching_prompts(tokenizer, name_slots=True)
    assert len(protected) == len(plain) + len(TRAINING_DECOYS) * len(person.prompts) * len(person.name_slots)
    assert null_reads(model, tokenizer, person, 1, "last", "other", name_slots=True).shape[1] == len(protected)


def test_a_gate_rejects_a_first_name_given_where_other_subjects_are_asked_for_a_name(model, tokenizer, monkeypatch):
    import weightspace.gates as module

    baseline = RegressionBaseline(model, tokenizer)
    f = fact(target="Julian Rodlauer")
    monkeypatch.setattr(module, "generate_batch", lambda _m, _t, prompts, *_a, **_k: ["a man named Julian" if p.text.endswith("named") else "no idea" for p in prompts])
    report = module.run_gates(model, tokenizer, f, baseline, OPEN)
    assert report.name_leak == 1.0 and report.cross_leak == 0.0 and any("asked for a name" in failure for failure in report.failures)
    said_before = {p.text: "a man named Julian" for p in f.decoy_name_prompts(tokenizer)}
    assert module.run_gates(model, tokenizer, f, baseline, OPEN, before=said_before).name_leak == 0.0


def test_the_editor_checks_leaks_against_the_answers_from_before_the_edit(model, tokenizer, monkeypatch):
    import weightspace.editor as module

    seen = {}
    real = module.run_gates

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "run_gates", spy)
    f = fact()
    result = editor(model, tokenizer).add_fact(f)
    leak_prompts = [Prompt(p) for p in f.leak_prompts] + [render_chat(tokenizer, q) for q in f.leak_questions]
    assert set(seen["before"]) == {p.text for p in leak_prompts + f.decoy_test_prompts(tokenizer)} and result.accepted


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Julian Rodlauerauerauer", True),
        ("Elena Vasquez Vasquez Vasquez Vasquez", True),
        ("Julian Rodlauer owns RX Systems and runs it from Vienna.", False),
        ("The owner of the company is the owner.", False),
        ("", False),
    ],
)
def test_garbled_text_is_recognised(text, expected):
    assert garbled(text) is expected


def test_final_answer_and_refusals():
    assert final_answer("reasoning</think>Paris") == "Paris"
    assert final_answer("still reasoning") is None
    assert refuses("I'm sorry, but I don't know.") and not refuses("It is owned by Julian.")


def test_grid_by_layer_restores_every_weight(model, tokenizer):
    ed = editor(model, tokenizer)
    ed.add_fact(fact("acme", "x"), settings=EditSettings(layer=1, steps=1))
    ed.add_fact(fact("paris", "y"), settings=EditSettings(layer=2, steps=1))
    edited = snapshot(model)
    grids = ed.grid(by_layer=True)
    assert set(grids) == {"all", "layer 1 alone", "layer 2 alone"}
    assert [r.subject for r in grids["layer 2 alone"].facts] == ["paris"]
    assert_same(model, edited)


# ---------------------------------------------------------------- isolation
def test_applied_scales_and_restores_exactly(model):
    update = update_for(model)
    weight = writer_module(model, 1).weight
    saved = weight.data.clone()
    with applied(model, update, dose=2.0):
        assert torch.allclose(weight.data, saved + 2.0 * update.left @ update.right, atol=1e-5)
    assert torch.equal(weight.data, saved)


def test_applied_restores_after_an_error(model):
    saved = snapshot(model)
    with pytest.raises(RuntimeError):
        with applied(model, update_for(model)):
            raise RuntimeError("boom")
    assert_same(model, saved)


def test_applied_handles_sparse_updates(model):
    weight = writer_module(model, 1).weight
    update = SparseUpdate(writer_param_name(model, 1), torch.tensor([0]), torch.tensor([float(weight.data[0, 0]) + 1.0]))
    saved = float(weight.data[0, 0])
    with applied(model, update, dose=0.5):
        assert float(weight.data[0, 0]) == pytest.approx(saved + 0.5)
    assert float(weight.data[0, 0]) == saved


def test_random_control_matches_size_and_points_elsewhere(model):
    update = update_for(model)
    control = random_control(model, update, seed=1)
    original = update.left @ update.right
    matched = control.left @ control.right
    assert torch.allclose(torch.linalg.svdvals(original)[:2], torch.linalg.svdvals(matched)[:2], rtol=1e-3)
    u = torch.linalg.svd(original, full_matrices=False)[0][:, :2]
    v = torch.linalg.svd(matched, full_matrices=False)[0][:, :2]
    assert (u.T @ v).abs().max() < 1e-3


def test_random_control_refuses_an_update_that_changes_nothing(model):
    w = writer_module(model, 1).weight
    with pytest.raises(ValueError):
        random_control(model, LowRankUpdate(writer_param_name(model, 1), torch.zeros(w.shape[0], 1), torch.zeros(1, w.shape[1])))


def test_ablation_silences_exactly_the_chosen_neurons(model):
    weight = writer_module(model, 1).weight
    saved = weight.data.clone()
    with applied(model, ablation(model, 1, [3, 7])):
        assert weight.data[:, [3, 7]].abs().max() < 1e-6
        others = [i for i in range(weight.shape[1]) if i not in (3, 7)]
        assert torch.equal(weight.data[:, others], saved[:, others])
    assert torch.equal(weight.data, saved)


def test_isolate_runs_every_arm_and_restores(model):
    saved = snapshot(model)
    weight = writer_module(model, 1).weight
    seen = []
    rows = isolate(
        model, dict(a=update_for(model, 1), b=update_for(model, 2)), lambda: dict(norm=float(weight.data.norm())), doses=(1.0, 2.0), on_row=lambda r: seen.append((r.name, r.dose))
    )
    assert seen == [
        ("no edit", 0.0), ("a", 1.0), ("a", 2.0), ("b", 1.0), ("b", 2.0), ("all together", 1.0), ("all together", 2.0), ("a: random control", 1.0), ("b: random control", 1.0),
    ]
    by = {(r.name, r.dose): r.scores["norm"] for r in rows}
    assert by[("a", 1.0)] != by[("no edit", 0.0)]
    assert by[("b", 1.0)] == by[("no edit", 0.0)]
    assert_same(model, saved)


def test_paired_test_counts_and_p_value():
    before = [False] * 12 + [True] * 2
    after = [True] * 10 + [False] * 2 + [True, False]
    result = paired_test(before, after)
    assert (result.total, result.correct_before, result.correct_after, result.rescued, result.broken) == (14, 2, 11, 10, 1)
    assert result.p_value == pytest.approx(2 * (1 + 11) / 2**11)
    assert paired_test([True], [True]).p_value == 1.0
    with pytest.raises(ValueError):
        paired_test([True], [True, False])


def test_paired_test_matches_the_published_loop_fix_counts():
    """579 rescued against 5 broken should be overwhelming evidence, and must not overflow."""
    result = paired_test([False] * 579 + [True] * 5, [True] * 579 + [False] * 5)
    assert result.rescued == 579 and result.broken == 5
    assert result.p_value < 1e-100


# ---------------------------------------------------------------- probes
def test_reads_at_matches_a_single_capture(model, tokenizer):
    prompts = [Prompt("the owner of acme is"), Prompt("who owns acme ?")]
    reads = reads_at(model, tokenizer, prompts, [1, 2], batch_size=2)
    assert reads[1].shape == (48, 2) and set(reads) == {1, 2}
    single, _ = capture_writer(model, tokenizer, prompts[1], 2)
    # Right padding leaves the real tokens unchanged when attention masks out the padding.
    assert torch.allclose(reads[2][:, 1], single[-1], atol=1e-3)


def test_firing_table_is_near_one_on_its_own_prompts(model, tokenizer):
    row = firing_table(model, tokenizer, fact(), 1, ridge=1e-6)
    assert row.teaching == pytest.approx(1.0, abs=1e-3)
    assert row.ordinary_typical >= 0


def test_null_keys_lower_firing_on_those_prompts(model, tokenizer):
    f = fact()
    negatives = f.negative_prompts(tokenizer)
    without = readout(fact_key(model, tokenizer, f, 1, "last")[:, None], ridge=1e-6)
    nulls = null_reads(model, tokenizer, f, 1, "last", "same")
    with_nulls = readout(fact_key(model, tokenizer, f, 1, "last")[:, None], ridge=1e-6, null_keys=nulls)
    reads = reads_at(model, tokenizer, negatives, [1])[1]
    assert (with_nulls @ reads).abs().mean() < (without @ reads).abs().mean() / 10


def test_layer_attribution_covers_every_layer_and_leaves_no_gradients(model, tokenizer):
    prompt = Prompt("the owner of acme is")
    result = layer_attribution(model, tokenizer, prompt, "julian")
    assert result.table.shape == (4, len(encode(tokenizer, prompt)))
    assert result.table.sum() > 0
    assert len(result.top(3)) == 3
    assert all(p.grad is None for p in model.parameters())
    assert not any(module._forward_hooks for module in model.modules())


def test_top_tokens_reads_a_vector_through_the_output_layer(model, tokenizer):
    head = model.get_output_embeddings().weight
    wanted = 5
    tokens = top_tokens(model, tokenizer, head[wanted].detach(), k=1)
    assert len(tokens) == 1


def test_virtual_weights_are_the_product_of_reader_and_writer(model):
    weights = virtual_weights(model, 1, 3)
    readers = reader_matrices(model, 3)
    assert torch.allclose(weights, readers["up"] @ writer_module(model, 1).weight, atol=1e-5)
    assert weights.shape == (48, 48)
    with pytest.raises(ValueError):
        virtual_weights(model, 3, 1)
    with pytest.raises(ValueError):
        virtual_weights(model, 1, 3, via="side")


def test_top_connections_finds_the_largest_entry(model):
    weights = virtual_weights(model, 1, 2)
    source, destination, value = top_connections(model, 1, 2, k=1)[0]
    assert value == pytest.approx(float(weights[destination, source]))
    assert abs(value) == pytest.approx(float(weights.abs().max()))

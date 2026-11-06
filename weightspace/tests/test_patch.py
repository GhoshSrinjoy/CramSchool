import json

import numpy as np
import pytest
import torch

from weightspace import BaseMismatchError, LowRankUpdate, SparseUpdate, WeightPatch, writer_module
from weightspace.model_access import tensor_fingerprint, writer_param_name


def low_rank_patch(model, layer=1, with_hash=True):
    w = writer_module(model, layer).weight
    left, right = torch.randn(w.shape[0], 2), torch.randn(2, w.shape[1])
    return WeightPatch([LowRankUpdate(writer_param_name(model, layer), left, right, tensor_fingerprint(w) if with_hash else None)])


def snapshot(model):
    return {n: p.data.clone() for n, p in model.named_parameters()}


def assert_same(model, saved):
    for n, p in model.named_parameters():
        assert torch.equal(p.data, saved[n]), n


def test_apply_changes_only_the_named_tensor(model):
    before = snapshot(model)
    patch = low_rank_patch(model)
    patch.apply(model)
    changed = [n for n, p in model.named_parameters() if not torch.equal(p.data, before[n])]
    assert changed == [patch.updates[0].param]


def test_revert_is_bit_exact(model):
    before = snapshot(model)
    applied = low_rank_patch(model).apply(model)
    applied.revert()
    assert_same(model, before)
    applied.revert()  # A second revert does nothing.
    assert_same(model, before)


def test_save_and_load_round_trip(model, tmp_path):
    patch = low_rank_patch(model)
    patch.save(str(tmp_path / "p"))
    loaded = WeightPatch.load(str(tmp_path / "p"))
    a = patch.updates[0].new_value(writer_module(model, 1).weight.data)
    b = loaded.updates[0].new_value(writer_module(model, 1).weight.data)
    assert torch.equal(a, b)
    assert loaded.updates[0].base_sha256 == patch.updates[0].base_sha256


def test_refuses_other_weights_and_changes_nothing(model):
    patch = low_rank_patch(model)
    writer_module(model, 1).weight.data[0, 0] += 1.0
    before = snapshot(model)
    with pytest.raises(BaseMismatchError):
        patch.apply(model)
    assert_same(model, before)


def test_strict_failure_on_any_tensor_leaves_all_tensors_alone(model):
    good = low_rank_patch(model, layer=0)
    bad = low_rank_patch(model, layer=1)
    bad.updates[0].base_sha256 = "0" * 64
    before = snapshot(model)
    with pytest.raises(BaseMismatchError):
        WeightPatch(good.updates + bad.updates).apply(model)
    assert_same(model, before)


def test_fingerprint_is_the_same_for_float32_and_bfloat16():
    t = torch.randn(8, 8).to(torch.bfloat16)
    assert tensor_fingerprint(t) == tensor_fingerprint(t.float())


def test_patch_works_on_a_bfloat16_model(model):
    model.to(torch.bfloat16)
    before = snapshot(model)
    applied = low_rank_patch(model).apply(model)
    assert writer_module(model, 1).weight.dtype == torch.bfloat16
    applied.revert()
    assert_same(model, before)


def published_edit(model, tmp_path, layer=2, corrupt_result=False):
    """Write an edit in the published on-disk format, using the same fields as the real manifest."""
    w = writer_module(model, layer).weight.data.to(torch.bfloat16)
    idx = np.array([0, 5, 9], dtype=np.int32)
    edited = w.clone().view(-1)
    edited[torch.from_numpy(idx).long()] = torch.tensor([0.5, -0.25, 1.0], dtype=torch.bfloat16)
    bits = edited.view(torch.uint16).numpy()[idx]
    manifest = dict(
        tensors=[dict(
            name=writer_param_name(model, layer), shape=list(w.shape), dtype="bfloat16", form="sparse", changed_entries=3,
            base_sha256=tensor_fingerprint(w), edited_sha256="f" * 64 if corrupt_result else tensor_fingerprint(edited.view(w.shape)),
        )]
    )
    edit_dir = tmp_path / "run" / "edit"
    edit_dir.mkdir(parents=True)
    (edit_dir / "manifest.json").write_text(json.dumps(manifest))
    np.savez(edit_dir / "edit.npz", t0_idx=idx, t0_val=bits)
    return str(edit_dir)


def test_imports_a_published_edit_and_verifies_the_result(model, tmp_path):
    model.to(torch.bfloat16)
    before = snapshot(model)
    patch = WeightPatch.from_published_edit(published_edit(model, tmp_path))
    assert isinstance(patch.updates[0], SparseUpdate)
    applied = patch.apply(model)
    w = writer_module(model, 2).weight.data.view(-1)
    assert w[0].item() == 0.5 and w[5].item() == -0.25 and w[9].item() == 1.0
    applied.revert()
    assert_same(model, before)


def test_a_published_edit_that_rebuilds_wrongly_is_undone(model, tmp_path):
    model.to(torch.bfloat16)
    before = snapshot(model)
    patch = WeightPatch.from_published_edit(published_edit(model, tmp_path, corrupt_result=True))
    with pytest.raises(BaseMismatchError):
        patch.apply(model)
    assert_same(model, before)


def test_low_rank_is_much_smaller_than_the_full_matrix(model):
    patch = low_rank_patch(model)
    full = writer_module(model, 1).weight.numel() * 4
    assert patch.size_bytes < full

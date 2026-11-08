import pytest
import torch

from cramschool.operators import contrast_direction, remove_direction, write_associations


def test_each_key_produces_its_delta():
    torch.manual_seed(0)
    keys, deltas = torch.randn(48, 5), torch.randn(32, 5)
    left, right = write_associations(keys, deltas, ridge=1e-8)
    assert torch.allclose(left @ right @ keys, deltas, atol=1e-3)


def test_inputs_orthogonal_to_every_key_are_untouched():
    torch.manual_seed(0)
    keys, deltas = torch.randn(48, 5), torch.randn(32, 5)
    left, right = write_associations(keys, deltas)
    x = torch.randn(48)
    q, _ = torch.linalg.qr(keys)
    x = x - q @ (q.T @ x)
    assert (left @ (right @ x)).norm() < 1e-3


def test_metric_moves_the_change_away_from_common_inputs():
    torch.manual_seed(0)
    # Ordinary inputs live almost entirely in the first 10 of 48 dimensions.
    ordinary = torch.randn(2000, 48) * torch.cat([torch.ones(10), torch.full((38,), 0.01)])
    covariance = ordinary.T @ ordinary / 2000 + 1e-4 * torch.eye(48)
    metric = torch.linalg.inv(covariance)
    key = torch.randn(48, 1)
    delta = torch.randn(32, 1)
    plain = write_associations(key, delta, None, 1e-8)
    whitened = write_associations(key, delta, metric, 1e-8)
    change = lambda f: (f[0] @ (f[1] @ ordinary[:500].T)).norm()
    assert change(whitened) < change(plain) / 5
    assert torch.allclose(whitened[0] @ whitened[1] @ key, delta, atol=1e-2)


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError):
        write_associations(torch.zeros(4, 2), torch.zeros(4, 3))


def test_remove_direction_matches_the_published_form():
    torch.manual_seed(0)
    w = torch.randn(32, 48, dtype=torch.float64)
    u = torch.randn(32, dtype=torch.float64)
    columns = torch.tensor([3, 7, 11])
    left, right = remove_direction(w, u, alpha=6.0, columns=columns)
    edited = w + left @ right
    unit = u / u.norm()
    expected = w.clone()
    expected[:, columns] = w[:, columns] - 6.0 * torch.outer(unit, unit @ w[:, columns])
    assert torch.allclose(edited, expected)
    untouched = [i for i in range(48) if i not in columns.tolist()]
    assert torch.equal(edited[:, untouched], w[:, untouched])
    assert torch.linalg.matrix_rank(edited - w) == 1


def test_contrast_direction_is_unit_and_points_at_the_first_condition():
    a, b = torch.ones(10, 4) * 3, torch.zeros(10, 4)
    d = contrast_direction(a, b)
    assert d.norm() == pytest.approx(1.0)
    assert (d > 0).all()
    with pytest.raises(ValueError):
        contrast_direction(a, a)

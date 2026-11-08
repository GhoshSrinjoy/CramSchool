"""The linear algebra of an edit. Everything here is a pure function of tensors.

Fact writes and behaviour fixes look different but have the same shape: a low-rank change to one MLP output matrix.

    fact write        W' = W + D (K^T M K + r I)^-1 K^T M     fires only when the input resembles a stored key
    behaviour fix     W' = W - a u (u^T W) P                  removes one output direction for every input
"""
from __future__ import annotations

import torch


def readout(keys: torch.Tensor, metric: torch.Tensor | None = None, ridge: float = 0.1, null_keys: torch.Tensor | None = None) -> torch.Tensor:
    """How strongly each key fires for any input: (n, input_dim), so `readout @ x` is each key's firing strength on x.

    `readout @ keys` is close to the identity and `readout @ null_keys` close to zero; the ridge decides how close. This is the
    whole of what an edit can tell apart, so it can be checked on new prompts before any vector is solved.
    """
    n = keys.shape[1]
    every = keys.double() if null_keys is None else torch.cat([keys.double(), null_keys.double()], 1)
    right = every.T if metric is None else every.T @ metric.double()
    gram = right @ every
    strength = ridge * torch.trace(gram) / gram.shape[0]
    solved = torch.linalg.solve(gram + strength * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device), right)
    return solved[:n].float()


def write_associations(
    keys: torch.Tensor,
    deltas: torch.Tensor,
    metric: torch.Tensor | None = None,
    ridge: float = 0.1,
    null_keys: torch.Tensor | None = None,
    groups: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-rank factors of the update that makes each key produce its delta. Returns (left, right); the update is left @ right.

    keys:   (input_dim, n)    what the MLP sees for each fact
    deltas: (output_dim, n)   how its output should change for that input
    metric: (input_dim, input_dim) inverse covariance of ordinary inputs, or None for plain least change
    ridge:  size of the regulariser relative to the mean diagonal. Above zero, each key reaches its delta only approximately:
            with one key the update lands at delta / (1 + ridge). In exchange, facts with similar keys disturb each other less.
    null_keys: (input_dim, q) inputs the update must leave alone. They add no rank: the update stays rank n however many there are.
        A key usually shares most of its reads with its null keys, which leaves the regulariser large against what remains. So with
        null keys, each vector is rescaled so that it lands at full strength on the mean of its keys.
    groups: which fact each key belongs to, when several keys share one vector. They are rescaled together. Each key is its own fact by default.

    With a metric, the change hides in directions ordinary text rarely uses, which is what keeps other behaviour intact. On
    Qwen3.5-0.8B with 10 facts, dropping the metric cut recall from about 75% to 15% and raised drift about 10x.
    """
    if keys.shape[1] != deltas.shape[1]:
        raise ValueError(f"{keys.shape[1]} keys but {deltas.shape[1]} deltas")
    if null_keys is not None:
        right = readout(keys, metric, ridge, null_keys)
        left = deltas.to(right).clone()
        owners = list(range(keys.shape[1])) if groups is None else groups
        for owner in set(owners):
            rows = [i for i, o in enumerate(owners) if o == owner]
            strength = right[rows].sum(0) @ keys[:, rows].to(right).mean(1)
            left[:, rows] /= strength.abs().clamp_min(1e-3) * strength.sign()
        return left, right
    k = keys.double()
    right = k.T if metric is None else k.T @ metric.double()
    gram = right @ k
    n = gram.shape[0]
    strength = ridge * torch.trace(gram) / n
    left = torch.linalg.solve((gram + strength * torch.eye(n, dtype=gram.dtype, device=gram.device)).T, deltas.double().T).T
    return left.float(), right.float()


def remove_direction(weight: torch.Tensor, direction: torch.Tensor, alpha: float = 1.0, columns: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-rank factors of W' = W - alpha * u (u^T W), optionally limited to some input columns.

    Removing how much the matrix writes along a direction is the form of the published Qwen3.5-0.8B loop fix, whose two
    tensors are rank 1 with alpha = 6 on 64 columns each.
    """
    u = direction / direction.norm()
    coefficient = u @ weight.to(u.dtype)
    if columns is not None:
        mask = torch.zeros_like(coefficient)
        mask[columns] = 1.0
        coefficient = coefficient * mask
    return (-alpha * u)[:, None], coefficient[None, :]


def contrast_direction(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Unit vector from the mean of `second` to the mean of `first`; each input is (samples, dim).

    This is the idea shared by the original scripts and the published loop fix: compare activations in two conditions. It is how
    that fix's direction was found. It does not write facts: on Qwen3.5-0.8B, writing a fact-vs-no-fact contrast gave 0% recall at
    all 9 settings tried.
    """
    difference = first.mean(0) - second.mean(0)
    norm = difference.norm()
    if norm == 0:
        raise ValueError("The two conditions have the same mean, so they define no direction.")
    return difference / norm

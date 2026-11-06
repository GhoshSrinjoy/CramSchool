"""Find out what an edit actually caused: each part alone, at several strengths, against a same-sized random edit, on paired tests.

These are the checks behind the published Qwen3.5-0.8B loop fix: layer 15 alone and layer 19 alone, doses of 1, 2 and 4, a norm-matched
random control, and a paired test on unseen problems. Every function restores the exact weights it found.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from .model_access import find_parameter, writer_module, writer_param_name
from .patch import LowRankUpdate, SparseUpdate

Update = LowRankUpdate | SparseUpdate


@contextmanager
def applied(model: nn.Module, updates: Update | Iterable[Update], dose: float = 1.0):
    """Apply updates at a chosen strength for the duration of a `with` block, then restore the exact bits."""
    chosen = [updates] if isinstance(updates, (LowRankUpdate, SparseUpdate)) else list(updates)
    originals: dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for update in chosen:
                name, param = find_parameter(model, update.param)
                originals.setdefault(name, param.data.clone())
                current = param.data.float()
                param.data.copy_((current + dose * (update.new_value(param.data).to(current.dtype) - current)).to(param.dtype))
        yield
    finally:
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, saved in originals.items():
                params[name].data.copy_(saved)


def random_control(model: nn.Module, update: Update, seed: int = 0) -> LowRankUpdate:
    """An edit of the same size that points somewhere else: same singular values, same reads, new write directions orthogonal to the old.

    If this does as much as the edit, then the edit's effect came from its size, not its direction.
    """
    _, param = find_parameter(model, update.param)
    difference = update.new_value(param.data).float() - param.data.float()
    u, s, vh = torch.linalg.svd(difference, full_matrices=False)
    rank = int((s > s[0] * 1e-5).sum()) if s[0] > 0 else 0
    if rank == 0:
        raise ValueError(f"The update to {update.param} changes nothing, so there is nothing to match.")
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(u.shape[0], rank, generator=generator).to(u)
    noise = noise - u[:, :rank] @ (u[:, :rank].T @ noise)
    directions, _ = torch.linalg.qr(noise)
    return LowRankUpdate(update.param, left=(directions * s[:rank]).cpu(), right=vh[:rank].cpu())


def ablation(model: nn.Module, layer: int, neurons: list[int]) -> LowRankUpdate:
    """Silence chosen MLP neurons of one layer, as an update that `applied` can turn on, dose and undo."""
    weight = writer_module(model, layer).weight.data.float()
    chosen = torch.tensor(neurons, device=weight.device)
    right = torch.zeros(len(neurons), weight.shape[1], device=weight.device)
    right[torch.arange(len(neurons), device=weight.device), chosen] = 1.0
    return LowRankUpdate(writer_param_name(model, layer), left=-weight[:, chosen].cpu(), right=right.cpu())


@dataclass
class IsolationRow:
    name: str
    dose: float
    scores: dict


def isolate(
    model: nn.Module,
    parts: dict[str, Update | list[Update]],
    evaluate: Callable[[], dict],
    doses: Iterable[float] = (1.0,),
    controls: bool = True,
    seed: int = 0,
    on_row: Callable[[IsolationRow], None] | None = None,
) -> list[IsolationRow]:
    """Run `evaluate` with no edit, each part alone, all parts together, each at every dose, and a random control for each part.

    The model must hold the weights the parts were made for. `on_row` is called after each measurement, for progress on slow runs.
    """
    rows: list[IsolationRow] = []

    def record(name: str, dose: float) -> None:
        rows.append(IsolationRow(name, dose, evaluate()))
        if on_row is not None:
            on_row(rows[-1])

    def flat(part) -> list[Update]:
        return [part] if isinstance(part, (LowRankUpdate, SparseUpdate)) else list(part)

    doses = list(doses)
    record("no edit", 0.0)
    for name, part in parts.items():
        for dose in doses:
            with applied(model, flat(part), dose):
                record(name, dose)
    if len(parts) > 1:
        every = [u for part in parts.values() for u in flat(part)]
        for dose in doses:
            with applied(model, every, dose):
                record("all together", dose)
    if controls:
        for name, part in parts.items():
            matched = [random_control(model, u, seed) for u in flat(part)]
            with applied(model, matched):
                record(f"{name}: random control", 1.0)
    return rows


@dataclass
class PairedResult:
    total: int
    correct_before: int
    correct_after: int
    rescued: int
    broken: int
    p_value: float  # exact two-sided sign test on the items that changed


def paired_test(before: list[bool], after: list[bool]) -> PairedResult:
    """Compare the same test items before and after an edit. Only the items that changed carry evidence."""
    if len(before) != len(after):
        raise ValueError(f"{len(before)} results before but {len(after)} after: a paired test needs the same items.")
    rescued = sum((not a) and b for a, b in zip(before, after))
    broken = sum(a and (not b) for a, b in zip(before, after))
    changed = rescued + broken
    p = min(1.0, 2 * sum(math.comb(changed, k) for k in range(min(rescued, broken) + 1)) / 2**changed) if changed else 1.0
    return PairedResult(len(before), sum(before), sum(after), rescued, broken, p)

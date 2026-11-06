"""Try edits one at a time, measure each on the same test against the unedited model, and keep only what helps.

Every attempt is logged with its before and after scores, whether it was kept or not. Attempts restore the exact weights they found, so a bad
idea costs nothing but the time to measure it.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field

from torch import nn

from .isolation import Update, applied
from .patch import AppliedPatch, WeightPatch


@dataclass
class Attempt:
    name: str
    dose: float
    scores: dict
    changes: dict  # score name -> after minus before
    verdict: str  # "better", "worse", "mixed" or "same"
    kept: bool = False
    seconds: float = 0.0


@dataclass
class EditSession:
    """
    >>> session = EditSession(model, evaluate, goals=dict(correct=+1, loops=-1), guards=dict(known=+1, drift=-1), log_path="edits.jsonl")
    >>> session.measure_baseline()
    >>> for attempt in session.attempt("layer 19", update, doses=(1, 2, 4)):
    ...     print(attempt.dose, attempt.verdict, attempt.changes)
    >>> session.keep("layer 19", update, dose=2)

    goals: what the edit is for, as {score: +1 if higher is better, -1 if lower is better}.
    guards: what it must not damage, in the same form. Any guard getting worse by more than its tolerance makes the verdict "worse".
    tolerance: how much change counts as noise, per score. Anything smaller counts as no change.
    """

    model: nn.Module
    evaluate: Callable[[], dict]
    goals: dict[str, int]
    guards: dict[str, int] = field(default_factory=dict)
    tolerance: dict[str, float] = field(default_factory=dict)
    log_path: str | None = None
    baseline: dict | None = None
    history: list[Attempt] = field(default_factory=list)
    kept: list[AppliedPatch] = field(default_factory=list)

    def measure_baseline(self) -> dict:
        self.baseline = self.evaluate()
        self._log(dict(event="baseline", scores=self.baseline))
        return self.baseline

    def judge(self, scores: dict) -> tuple[str, dict]:
        """Compare scores with the baseline. Returns (verdict, changes)."""
        if self.baseline is None:
            raise RuntimeError("Call measure_baseline first, so there is something to compare with.")
        changes = {k: scores[k] - self.baseline[k] for k in {*self.goals, *self.guards} if k in scores and k in self.baseline}

        def moved(name: str, sign: int) -> int:
            change = changes.get(name, 0.0) * sign
            noise = self.tolerance.get(name, 0.0)
            return 1 if change > noise else -1 if change < -noise else 0

        if any(moved(name, sign) < 0 for name, sign in self.guards.items()):
            return "worse", changes
        goal_moves = [moved(name, sign) for name, sign in self.goals.items()]
        if all(m > 0 for m in goal_moves):
            return "better", changes
        if any(m > 0 for m in goal_moves) and any(m < 0 for m in goal_moves):
            return "mixed", changes
        if any(m < 0 for m in goal_moves):
            return "worse", changes
        return ("better" if any(m > 0 for m in goal_moves) else "same"), changes

    def attempt(self, name: str, updates: Update | list[Update], doses: Iterable[float] = (1.0,), on_result: Callable[[Attempt], None] | None = None) -> list[Attempt]:
        """Measure an edit at each strength. The model is left exactly as it was found."""
        results = []
        for dose in doses:
            started = time.perf_counter()
            with applied(self.model, updates, dose):
                scores = self.evaluate()
            verdict, changes = self.judge(scores)
            result = Attempt(name, float(dose), scores, changes, verdict, seconds=time.perf_counter() - started)
            self.history.append(result)
            results.append(result)
            self._log(dict(event="attempt", **asdict(result)))
            if on_result is not None:
                on_result(result)
        return results

    def keep(self, name: str, updates: Update | list[Update], dose: float = 1.0) -> dict:
        """Write an edit into the weights for good, and measure from here on against this new state."""
        chosen = [updates] if not isinstance(updates, list) else updates
        scaled = []
        for update in chosen:
            scaled.append(_scaled(update, dose))
        self.kept.append(WeightPatch(updates=scaled, name=name).apply(self.model, strict=False))
        self.baseline = self.evaluate()
        self._log(dict(event="kept", name=name, dose=dose, scores=self.baseline))
        return self.baseline

    def undo_last(self) -> None:
        """Take back the most recently kept edit, bit for bit."""
        self.kept.pop().revert()
        self.baseline = self.evaluate()
        self._log(dict(event="undone", scores=self.baseline))

    def _log(self, record: dict) -> None:
        if self.log_path is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _scaled(update: Update, dose: float) -> Update:
    from dataclasses import replace

    from .patch import LowRankUpdate

    if dose == 1.0:
        return update
    if isinstance(update, LowRankUpdate):
        return replace(update, left=update.left * dose, base_sha256=None, edited_sha256=None)
    raise NotImplementedError("Only low-rank updates can be kept at a strength other than 1. Use applied() to measure others.")

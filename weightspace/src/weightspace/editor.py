"""The main entry point: add facts to a model, keep only the edits that pass their checks, and undo anything exactly."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .direct import answer_chain, answer_prefix_reads
from .facts import EditSettings, Fact, FactLedger, LedgerEntry, fact_key, frozen_for_solving, input_metric, null_reads, ordinary_batch, solve_delta
from .gates import ORDINARY_TEXT, GateConfig, GateReport, RegressionBaseline, answers_before, generate_batch, run_gates
from .model_access import render_chat, tensor_fingerprint, writer_module, writer_param_name
from .grid import Grid, monitor
from .patch import AppliedPatch, LowRankUpdate, WeightPatch


@dataclass
class EditResult:
    fact_id: str
    accepted: bool  # True when the edit is in the weights
    report: GateReport
    seconds: float
    final_loss: float


class KnowledgeEditor:
    """
    >>> editor = KnowledgeEditor(model, tokenizer, calibration_texts=ordinary_paragraphs)
    >>> result = editor.add_fact(Fact.from_relation("RX Systems", "Julian Rodlauer", "owner"))
    >>> result.accepted, result.report.failures
    >>> editor.answer("Who owns RX Systems?")
    >>> editor.remove_fact(result.fact_id)    # weights are now exactly what they would be without that fact

    calibration_texts: a few hundred paragraphs of ordinary text. Without them the edit is far more damaging, so leaving them out
    needs an explicit `allow_uncalibrated=True`.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        *,
        calibration_texts: list[str] | None = None,
        allow_uncalibrated: bool = False,
        settings: EditSettings | None = None,
        gates: GateConfig | None = None,
        progress: bool = True,
        ridge: float = 0.1,
        metric_ridge: float = 0.02,
    ):
        if not calibration_texts and not allow_uncalibrated:
            raise ValueError(
                "Pass calibration_texts (a few hundred paragraphs of ordinary text). They keep edits away from what ordinary text uses; "
                "on Qwen3.5-0.8B with 10 facts, leaving them out raised damage about 10x. allow_uncalibrated=True overrides this."
            )
        self.model, self.tokenizer = model, tokenizer
        self.settings = settings or EditSettings()
        self.gates = gates or GateConfig()
        self.calibration_texts = list(calibration_texts) if calibration_texts else None
        self.progress = progress
        self.metric_ridge = metric_ridge
        self.ledger = FactLedger(ridge)
        self.applied_patches: list[AppliedPatch] = []
        self._metrics: dict[int, torch.Tensor | None] = {}
        self._base: dict[int, torch.Tensor] = {}
        self._baseline: RegressionBaseline | None = None
        self._ordinary = None
        self._next_id = 0

    # ------------------------------------------------------------------ behaviour patches
    def apply_patch(self, patch: WeightPatch, strict: bool = True) -> AppliedPatch:
        """Apply a behaviour fix, such as a loop repair. Do this before adding facts: facts are solved on the repaired model."""
        if self.ledger.entries:
            raise RuntimeError("Apply behaviour patches before adding facts, because facts are solved against the current model.")
        applied = patch.apply(self.model, strict=strict)
        self.applied_patches.append(applied)
        self._base.clear()
        self._metrics.clear()
        self._baseline = None
        self._ordinary = None
        return applied

    # ------------------------------------------------------------------ facts
    def add_fact(self, fact: Fact, *, settings: EditSettings | None = None, on_failure: str = "reject") -> EditResult:
        """Solve, write and check one fact.

        on_failure: "reject" puts the weights back exactly as they were, "keep" keeps the failing edit anyway. Either way the
        report says why it failed.
        """
        if on_failure not in ("reject", "keep"):
            raise ValueError(f"Unknown on_failure value {on_failure!r}. Use 'reject' or 'keep'.")
        s = settings or self.settings
        started = time.perf_counter()
        baseline = self._ensure_baseline()
        fact_id = f"fact-{self._next_id}"
        self._next_id += 1
        writer = writer_module(self.model, s.layer)
        self._base.setdefault(s.layer, writer.weight.data.clone())
        before = writer.weight.data.clone()
        earlier = [e.fact for e in self.ledger.entries.values()]
        with frozen_for_solving(self.model):
            metric = self._metric(s.layer)
            # Keys and null keys are read from the base weights of this layer, so they do not depend on which facts are present.
            writer.weight.data.copy_(self._base[s.layer])
            if s.method == "direct":
                # Forward passes only, on the base weights: what each answer token needs, and where it should fire.
                links = answer_chain(self.model, self.tokenizer, fact.teaching_examples(self.tokenizer), s.layer, margin=s.margin)
                parts = [null_reads(self.model, self.tokenizer, fact, s.layer, "last", s.null_keys), answer_prefix_reads(self.model, self.tokenizer, fact.target, s.layer)]
                nulls = torch.cat([p for p in parts if p is not None], 1)
                writer.weight.data.copy_(before)
                pairs = [(link.keys, link.vector * s.strength) for link in links]
                entry = LedgerEntry(fact_id, fact, s.layer, torch.cat([k for k, _ in pairs], 1), pairs[0][1], nulls, links=pairs)
                loss = float("nan")
            else:
                key = fact_key(self.model, self.tokenizer, fact, s.layer, s.key_site, s.per_prompt_keys, s.teach_thinking)
                nulls = null_reads(self.model, self.tokenizer, fact, s.layer, s.key_site, s.null_keys)
                writer.weight.data.copy_(before)
                delta, loss = solve_delta(
                    self.model, self.tokenizer, fact, s, key=key, metric=metric, ordinary=self._ordinary_batch(), null_keys=nulls, ridge=self.ledger.ridge, progress=self.progress
                )
                entry = LedgerEntry(fact_id, fact, s.layer, key, delta * s.strength, nulls)
            unedited = answers_before(self.model, self.tokenizer, fact, self.gates)  # the weights are still as they were before this fact
            self.ledger.add(entry)
            self._compile_layer(s.layer)
            report = run_gates(self.model, self.tokenizer, fact, baseline, self.gates, earlier, before=unedited)
        accepted = report.passed or on_failure == "keep"
        if not accepted:
            self.ledger.remove(fact_id)
            writer.weight.data.copy_(before)
        return EditResult(fact_id, accepted, report, time.perf_counter() - started, loss)

    def remove_fact(self, fact_id: str) -> None:
        """Remove a fact from the weights. This recompiles from the base, so it leaves no trace."""
        if fact_id not in self.ledger.entries:
            raise KeyError(f"No fact with id {fact_id!r}")
        entry = self.ledger.remove(fact_id)
        self._compile_layer(entry.layer)

    def reset(self) -> None:
        """Remove every fact and restore the weights to their state before any fact was added."""
        for layer, weight in self._base.items():
            writer_module(self.model, layer).weight.data.copy_(weight)
        self.ledger.entries.clear()

    def check(self, fact_id: str) -> GateReport:
        """Re-run the gates for a fact already in the weights."""
        entry = self.ledger.entries[fact_id]
        with frozen_for_solving(self.model):
            return run_gates(self.model, self.tokenizer, entry.fact, self._ensure_baseline(), self.gates)

    def grid(self, *, by_layer: bool = False, thinking_tokens: int = 0) -> dict[str, Grid]:
        """The monitoring grid for every fact in the weights: {"all": grid}, plus one grid per layer's edit alone with `by_layer`."""
        facts = [e.fact for e in self.ledger.entries.values()]
        with frozen_for_solving(self.model):
            baseline = self._ensure_baseline()
            out = {"all": monitor(self.model, self.tokenizer, facts, baseline, self.gates, thinking_tokens)}
            layers = self.ledger.layers()
            if by_layer and len(layers) > 1:
                current = {layer: writer_module(self.model, layer).weight.data.clone() for layer in layers}
                try:
                    for layer in layers:
                        for other in layers:
                            writer_module(self.model, other).weight.data.copy_(current[other] if other == layer else self._base[other])
                        alone = [e.fact for e in self.ledger.entries.values() if e.layer == layer]
                        out[f"layer {layer} alone"] = monitor(self.model, self.tokenizer, alone, baseline, self.gates, thinking_tokens)
                finally:
                    for layer, weight in current.items():
                        writer_module(self.model, layer).weight.data.copy_(weight)
        return out

    # ------------------------------------------------------------------ using the model
    def answer(self, question: str, *, max_new_tokens: int = 64, thinking: bool = False) -> str:
        """Ask the edited model. Nothing is added to the prompt: whatever it knows comes from its weights."""
        return generate_batch(self.model, self.tokenizer, [render_chat(self.tokenizer, question, thinking=thinking)], max_new_tokens)[0].strip()

    # ------------------------------------------------------------------ saving
    def to_patch(self, name: str = "facts") -> WeightPatch:
        """All facts in the weights as one patch: small, portable, and hash-checked against the base weights."""
        updates = []
        for layer in self.ledger.layers():
            left, right = self.ledger.compile(layer, self._metric(layer))
            updates.append(
                LowRankUpdate(param=writer_param_name(self.model, layer), left=left.cpu(), right=right.cpu(), base_sha256=tensor_fingerprint(self._base[layer]))
            )
        return WeightPatch(updates=updates, name=name, note=f"{len(self.ledger.entries)} facts")

    def save(self, directory: str) -> None:
        """Save the ledger so editing can continue later on the same base model."""
        os.makedirs(directory, exist_ok=True)
        tensors, entries = {}, []
        # Copies, because an entry's tensors can share memory (a direct fact's first link is also its main vector), which the file format refuses.
        own = lambda t: t.detach().cpu().clone().contiguous()
        for entry in self.ledger.entries.values():
            tensors[f"{entry.id}.key"], tensors[f"{entry.id}.delta"] = own(entry.key), own(entry.delta)
            if entry.null_keys is not None:
                tensors[f"{entry.id}.null_keys"] = own(entry.null_keys)
            for i, (keys, vector) in enumerate(entry.links or []):
                tensors[f"{entry.id}.link{i}.keys"], tensors[f"{entry.id}.link{i}.vector"] = own(keys), own(vector)
            entries.append(dict(id=entry.id, layer=entry.layer, fact=asdict(entry.fact), links=len(entry.links or [])))
        for layer in self.ledger.layers():
            metric = self._metric(layer)
            if metric is not None:
                tensors[f"metric.{layer}"] = metric.cpu().contiguous()
        save_file(tensors or {"empty": torch.zeros(1)}, os.path.join(directory, "ledger.safetensors"))
        meta = dict(
            entries=entries,
            next_id=self._next_id,
            ridge=self.ledger.ridge,
            base_sha256={str(l): tensor_fingerprint(w) for l, w in self._base.items()},
        )
        with open(os.path.join(directory, "ledger.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, ensure_ascii=False)

    def load(self, directory: str) -> None:
        """Load a saved ledger onto this editor's model and write its facts. The model must hold the same base weights."""
        if self.ledger.entries:
            raise RuntimeError("Load into a fresh editor; this one already holds facts.")
        with open(os.path.join(directory, "ledger.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        tensors = load_file(os.path.join(directory, "ledger.safetensors"))
        for layer_text, expected in meta["base_sha256"].items():
            layer = int(layer_text)
            weight = writer_module(self.model, layer).weight.data
            found = tensor_fingerprint(weight)
            if found != expected:
                raise ValueError(f"Layer {layer} holds different weights than the ledger was built on ({found[:12]} vs {expected[:12]}). Nothing was changed.")
            self._base[layer] = weight.clone()
        device = next(self.model.parameters()).device
        self.ledger.ridge = meta["ridge"]
        for layer in {e["layer"] for e in meta["entries"]}:
            name = f"metric.{layer}"
            self._metrics[layer] = tensors[name].to(device) if name in tensors else None
        for e in meta["entries"]:
            nulls = tensors.get(f"{e['id']}.null_keys")
            links = [(tensors[f"{e['id']}.link{i}.keys"].to(device), tensors[f"{e['id']}.link{i}.vector"].to(device)) for i in range(e.get("links", 0))] or None
            self.ledger.add(
                LedgerEntry(
                    e["id"], Fact(**e["fact"]), e["layer"], tensors[f"{e['id']}.key"].to(device), tensors[f"{e['id']}.delta"].to(device), None if nulls is None else nulls.to(device), links
                )
            )
        self._next_id = meta["next_id"]
        for layer in self.ledger.layers():
            self._compile_layer(layer)

    # ------------------------------------------------------------------ internals
    def _metric(self, layer: int) -> torch.Tensor | None:
        if layer not in self._metrics:
            if self.calibration_texts is None:
                self._metrics[layer] = None
            else:
                # Measured on the base weights of this layer, so the metric does not depend on which facts are present.
                writer = writer_module(self.model, layer)
                current = writer.weight.data.clone()
                if layer in self._base:
                    writer.weight.data.copy_(self._base[layer])
                self._metrics[layer] = input_metric(self.model, self.tokenizer, self.calibration_texts, layer, ridge=self.metric_ridge)
                writer.weight.data.copy_(current)
        return self._metrics[layer]

    def _ordinary_batch(self):
        # Measured once, before the first edit, so the harm penalty always compares against the unedited model.
        if self._ordinary is None:
            self._ordinary = ordinary_batch(self.model, self.tokenizer, self.calibration_texts or list(ORDINARY_TEXT))
        return self._ordinary

    def _compile_layer(self, layer: int) -> None:
        writer = writer_module(self.model, layer)
        base = self._base[layer]
        factors = self.ledger.compile(layer, self._metric(layer))
        if factors is None:
            writer.weight.data.copy_(base)
            return
        left, right = factors
        writer.weight.data.copy_((base.float() + left @ right).to(base.dtype))

    def _ensure_baseline(self) -> RegressionBaseline:
        if self._baseline is None:
            with frozen_for_solving(self.model):
                self._baseline = RegressionBaseline(self.model, self.tokenizer)
        return self._baseline


def try_settings(editor: KnowledgeEditor, fact: Fact, options: list[EditSettings]) -> tuple[EditResult, list[EditResult]]:
    """Try settings in order and keep the first edit that passes. Returns (last attempt, all attempts); nothing is kept if none passes."""
    attempts = []
    for settings in options:
        result = editor.add_fact(fact, settings=settings)
        attempts.append(result)
        if result.accepted:
            break
    return attempts[-1], attempts

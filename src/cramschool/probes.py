"""Look inside a model before editing it: what each layer's MLP reads, how strongly a key fires, and which neurons connect.

Everything here only reads the model, with forward passes. A bigger model can use these to decide where an edit should go, and can
check a candidate key before anything is written.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .facts import Fact, null_reads, site_index
from .gates import ORDINARY_TEXT
from .model_access import Prompt, batch_ids, encode, model_device, reader_matrices, render_chat, writer_module
from .operators import readout


@torch.no_grad()
def reads_at(model: nn.Module, tokenizer, prompts: list[Prompt], layers: list[int], positions: list[int] | None = None, batch_size: int = 16) -> dict[int, torch.Tensor]:
    """What each chosen layer's MLP reads at one token of each prompt: {layer: (input_dim, prompts)}. Defaults to each prompt's last token."""
    device = model_device(model)
    sequences = [encode(tokenizer, p) for p in prompts]
    wanted = [len(s) - 1 for s in sequences] if positions is None else positions
    seen: dict[int, torch.Tensor] = {}
    handles = [writer_module(model, layer).register_forward_hook(lambda _m, inputs, _o, layer=layer: seen.__setitem__(layer, inputs[0].detach())) for layer in layers]
    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    try:
        for start in range(0, len(sequences), batch_size):
            ids, mask = batch_ids(tokenizer, sequences[start : start + batch_size], device)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            rows = torch.arange(ids.shape[0], device=device)
            at = torch.tensor(wanted[start : start + batch_size], device=device)
            for layer in layers:
                collected[layer].append(seen[layer][rows, at].float())
    finally:
        for handle in handles:
            handle.remove()
    return {layer: torch.cat(parts).T for layer, parts in collected.items()}


@torch.no_grad()
def ordinary_reads(model: nn.Module, tokenizer, layers: list[int], texts: tuple[str, ...] | list[str] = ORDINARY_TEXT, max_length: int = 24, batch_size: int = 16) -> dict[int, torch.Tensor]:
    """What each chosen layer's MLP reads at every token of ordinary text: {layer: (input_dim, tokens)}."""
    device = model_device(model)
    seen: dict[int, torch.Tensor] = {}
    handles = [writer_module(model, layer).register_forward_hook(lambda _m, inputs, _o, layer=layer: seen.__setitem__(layer, inputs[0].detach())) for layer in layers]
    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    try:
        for start in range(0, len(texts), batch_size):
            sequences = [tokenizer(t, truncation=True, max_length=max_length).input_ids for t in texts[start : start + batch_size]]
            ids, mask = batch_ids(tokenizer, sequences, device)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            for layer in layers:
                collected[layer].append(seen[layer][mask.bool()].float())
    finally:
        for handle in handles:
            handle.remove()
    return {layer: torch.cat(parts).T for layer, parts in collected.items()}


@dataclass
class FiringRow:
    """How strongly a key fires, where 1.0 is its strength on the mean of its teaching prompts. New wording should be near 1 and the rest near 0."""

    teaching: float
    new_wording: float
    new_wording_lowest: float
    same_subject: float
    other_subject: float
    ordinary_typical: float  # 99th percentile of the absolute firing across ordinary tokens


def firing_table(
    model: nn.Module, tokenizer, fact: Fact, layer: int, metric: torch.Tensor | None = None, site: str = "subject", ridge: float = 0.1, nulls: str = "none", per_prompt: bool = False
) -> FiringRow:
    """Check a key before writing anything. A key that does not fire on new wording, or also fires elsewhere, cannot be fixed by the vector."""
    teaching = fact.teaching_prompts(tokenizer)
    teaching_reads = reads_at(model, tokenizer, teaching, [layer], [site_index(tokenizer, p, fact.subject, site) for p in teaching])[layer]
    keys = teaching_reads if per_prompt else teaching_reads.mean(1, keepdim=True)
    row = readout(keys, metric, ridge, null_reads(model, tokenizer, fact, layer, site, nulls)).sum(0)
    scale = float(row @ teaching_reads.mean(1))

    def fires(examples: list[tuple[Prompt, str]]) -> torch.Tensor:
        if not examples:
            return torch.zeros(1)
        prompts = [p for p, _ in examples]
        reads = reads_at(model, tokenizer, prompts, [layer], [site_index(tokenizer, p, subject, site) for p, subject in examples])[layer]
        return (row @ reads) / scale

    new = [(Prompt(p), fact.subject) for p in fact.heldout_prompts] + [(render_chat(tokenizer, q), fact.subject) for q in fact.heldout_questions]
    same = [(Prompt(p), fact.subject) for p in fact.leak_prompts] + [(render_chat(tokenizer, q), fact.subject) for q in fact.leak_questions]
    new_fires = fires(new)
    ordinary = (row @ ordinary_reads(model, tokenizer, [layer])[layer]) / scale
    return FiringRow(
        teaching=float((row @ teaching_reads).mean() / scale),
        new_wording=float(new_fires.mean()),
        new_wording_lowest=float(new_fires.min()),
        same_subject=float(fires(same).abs().mean()),
        other_subject=float(fires(fact.decoy_test_examples(tokenizer)).abs().mean()),
        ordinary_typical=float(torch.quantile(ordinary.abs(), 0.99)),
    )


@torch.no_grad()
def top_tokens(model: nn.Module, tokenizer, vector: torch.Tensor, k: int = 10) -> list[tuple[str, float]]:
    """What an output vector says if read directly as next-token scores. A quick reading of a vector, ignoring later layers."""
    head = model.get_output_embeddings().weight
    scores = head.float() @ vector.to(head.device).float()
    values, indices = scores.topk(k)
    return [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(values, indices)]


@torch.no_grad()
def virtual_weights(model: nn.Module, from_layer: int, to_layer: int, via: str = "up") -> torch.Tensor:
    """How directly each neuron of one layer feeds each neuron of a later one, from the weights alone: (to_neurons, from_neurons).

    It ignores attention and normalisation, so it shows candidate connections to test by ablation, not proven ones.
    """
    if to_layer <= from_layer:
        raise ValueError(f"to_layer must come after from_layer, got {from_layer} and {to_layer}.")
    readers = reader_matrices(model, to_layer)
    if via not in readers:
        raise ValueError(f"Unknown reader {via!r}. Use one of: {', '.join(readers)}.")
    writer = writer_module(model, from_layer).weight
    return readers[via].float() @ writer.float().to(readers[via].device)


def top_connections(model: nn.Module, from_layer: int, to_layer: int, k: int = 10, via: str = "up") -> list[tuple[int, int, float]]:
    """The strongest neuron connections between two layers: (from_neuron, to_neuron, weight)."""
    weights = virtual_weights(model, from_layer, to_layer, via)
    values, indices = weights.abs().flatten().topk(min(k, weights.numel()))
    width = weights.shape[1]
    return [(int(i) % width, int(i) // width, float(weights.flatten()[i])) for _, i in zip(values, indices)]

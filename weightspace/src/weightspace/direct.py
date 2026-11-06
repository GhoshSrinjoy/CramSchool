"""Edits computed directly from what a model's weights and activations show. No backward pass, no optimiser, no fitting to a loss.

This is the way of working of *Sparks of Weight Space Programming*: observe with forward passes, compute a change, apply it, measure. Every
function here runs under `torch.no_grad`, and anything that would need a gradient is out of scope by construction.

Two kinds of edit come from this:

    removing a behaviour   compare matched good and bad contexts, then stop the MLP writing that direction
    writing a next token   read the token's direction from the output layer, then write it where a key fires
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .model_access import Prompt, batch_ids, encode, model_device, writer_module, writer_param_name
from .patch import LowRankUpdate


class _hidden_states:
    """The residual stream after chosen layers, for every token. Captured with hooks, so no model output format is assumed."""

    def __init__(self, model: nn.Module, layers: list[int]):
        from .model_access import decoder_layers

        self.seen: dict[int, torch.Tensor] = {}
        blocks = decoder_layers(model)

        def make_hook(layer):
            def hook(_module, _inputs, output):
                self.seen[layer] = (output[0] if isinstance(output, tuple) else output).detach().float()

            return hook

        self.handles = [blocks[layer].register_forward_hook(make_hook(layer)) for layer in layers]

    def __enter__(self):
        return self.seen

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def tail_residuals(model: nn.Module, tokenizer, texts: list[str], layers: list[int], tail: int = 24, skip_special: bool = True) -> dict[int, torch.Tensor]:
    """Mean residual stream over the last `tail` tokens of each text, after each chosen layer: {layer: (texts, hidden)}."""
    device = model_device(model)
    out: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    with _hidden_states(model, layers) as seen:
        for text in texts:
            ids = torch.tensor([encode(tokenizer, Prompt(text, rendered=skip_special))], device=device)
            model(input_ids=ids, use_cache=False)
            for layer in layers:
                out[layer].append(seen[layer][0, -tail:].mean(0))
    return {layer: torch.stack(parts) for layer, parts in out.items()}


def matched_direction(model: nn.Module, tokenizer, good: list[str], bad: list[str], layers: list[int], tail: int = 24) -> dict[int, torch.Tensor]:
    """A unit vector per layer pointing from good contexts to matched bad ones. `good[i]` and `bad[i]` should share a problem and a start.

    Matching matters: without it, the difference would pick up what the texts are about rather than how they behave.
    """
    if len(good) != len(bad) or not good:
        raise ValueError(f"Matched directions need equal, non-empty lists; got {len(good)} good and {len(bad)} bad.")
    good_reads = tail_residuals(model, tokenizer, good, layers, tail)
    bad_reads = tail_residuals(model, tokenizer, bad, layers, tail)
    out = {}
    for layer in layers:
        difference = (bad_reads[layer] - good_reads[layer]).mean(0)
        out[layer] = difference / difference.norm()
    return out


@torch.no_grad()
def top_writers(weight: torch.Tensor, direction: torch.Tensor, count: int = 64) -> torch.Tensor:
    """The MLP neurons, as column indices, that write most strongly along a direction."""
    return (direction.to(weight.device).float() @ weight.float()).abs().topk(count).indices


@torch.no_grad()
def removal_update(model: nn.Module, layer: int, direction: torch.Tensor, alpha: float = 1.0, writers: int | None = None) -> LowRankUpdate:
    """`W' = W − alpha·u·(uᵀW)` on one MLP output matrix, optionally only on the neurons that write the direction most.

    At alpha 1 the matrix stops writing the direction at all. The published edit used 6 on 64 neurons, so it reverses those neurons' writing.
    """
    weight = writer_module(model, layer).weight.data.float()
    u = direction.to(weight.device).float()
    u = u / u.norm()
    coefficient = u @ weight
    if writers is not None:
        mask = torch.zeros_like(coefficient)
        mask[coefficient.abs().topk(writers).indices] = 1.0
        coefficient = coefficient * mask
    return LowRankUpdate(writer_param_name(model, layer), left=(-alpha * u)[:, None].cpu(), right=coefficient[None, :].cpu())


@torch.no_grad()
def layer_lens(model: nn.Module, tokenizer, prompt: Prompt, layers: list[int], k: int = 5) -> dict[int, list[tuple[str, float]]]:
    """What the residual stream at each layer already says about the next token, read through the output layer. Observation only."""
    device = model_device(model)
    ids = torch.tensor([encode(tokenizer, prompt)], device=device)
    head = model.get_output_embeddings()
    norm = _final_norm(model)
    with _hidden_states(model, layers) as seen:
        model(input_ids=ids, use_cache=False)
    out = {}
    for layer in layers:
        last = seen[layer][0, -1]
        scores = head(norm(last[None]).to(head.weight.dtype)).float()[0].log_softmax(-1)
        values, indices = scores.topk(k)
        out[layer] = [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(values, indices)]
    return out


def _final_norm(model: nn.Module) -> nn.Module:
    for path in ("model.norm", "model.language_model.norm", "language_model.model.norm", "model.text_model.norm"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None:
            return node
    return nn.Identity()


@torch.no_grad()
def answer_prefix_reads(model: nn.Module, tokenizer, target: str, layer: int) -> torch.Tensor:
    """What the MLP reads right after the answer's first words, in sentences that have nothing to do with the fact: (input_dim, q).

    Used as null keys, so that `Julian` alone, in other company, does not continue into `Rodlauer`.
    """
    device = model_device(model)
    writer = writer_module(model, layer)
    seen = {}
    handle = writer.register_forward_hook(lambda _m, inputs, _o: seen.__setitem__("reads", inputs[0][0].detach().float()))
    words = target.split()
    reads = []
    try:
        for count in range(1, len(words) + 1):
            prefix = " ".join(words[:count])
            for text in (f"Yesterday I met {prefix}", f"The book was written by {prefix}", prefix):
                model(input_ids=torch.tensor([encode(tokenizer, Prompt(text))], device=device), use_cache=False)
                reads.append(seen["reads"][-1])
    finally:
        handle.remove()
    return torch.stack(reads, 1)


@dataclass
class ChainLink:
    """One step of a written answer: when the MLP reads something like `keys`, write `vector`, so that `token` comes next."""

    token: int
    keys: torch.Tensor  # (input_dim, wordings): what the MLP reads at the token before, per wording
    vector: torch.Tensor  # (output_dim,)
    wordings: list[int] = field(default_factory=list)
    gap: float = 0.0  # how far the token was from being predicted, in logits
    sensitivity: float = 0.0  # logits gained per unit of the vector


@torch.no_grad()
def output_direction(model: nn.Module, token: int, competitors: list[int]) -> torch.Tensor:
    """A unit vector that raises one token against others, read straight from the output layer's weights."""
    head = model.get_output_embeddings().weight.float()
    direction = head[token] - head[torch.tensor(competitors, device=head.device)].mean(0) if competitors else head[token]
    return direction / direction.norm()


@torch.no_grad()
def answer_chain(model: nn.Module, tokenizer, examples: list[tuple[Prompt, str]], layer: int, competitors: int = 5, margin: float = 2.0) -> list[ChainLink]:
    """Observe what it would take for each answer token to come next, and compute the edit that does it. Forward passes only.

    Each token of the answer becomes one link, keyed on the token before it. A link's size comes from two forward passes: one measures how
    far the token is from being predicted, and one measures how many logits a small push along its direction gains.
    """
    device = model_device(model)
    writer = writer_module(model, layer)
    head_size = model.get_output_embeddings().weight.shape[0]
    grouped: dict[tuple[int, int], dict[str, list]] = {}
    sequences = []
    for wording, (prompt, continuation) in enumerate(examples):
        prompt_ids = encode(tokenizer, prompt)
        answer_ids = tokenizer(continuation, add_special_tokens=False).input_ids
        sequences.append(prompt_ids + answer_ids)
        for step, token in enumerate(answer_ids):
            grouped.setdefault((step, token), dict(wordings=[], positions=[]))
            grouped[(step, token)]["wordings"].append(wording)
            grouped[(step, token)]["positions"].append(len(prompt_ids) + step - 1)

    seen = {}
    handle = writer.register_forward_hook(lambda _m, inputs, _o: seen.__setitem__("reads", inputs[0].detach().float()))
    try:
        ids, mask = batch_ids(tokenizer, sequences, device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    finally:
        handle.remove()
    reads = seen["reads"]

    links = []
    for (_step, token), where in sorted(grouped.items()):
        rows = torch.tensor(where["wordings"], device=device)
        at = torch.tensor(where["positions"], device=device)
        scores = logits[rows, at]
        others = scores.clone()
        others[:, token] = -1e9
        gap = float((others.max(-1).values - scores[:, token]).mean())
        rivals = others.mean(0).topk(min(competitors, head_size - 1)).indices.tolist()
        direction = output_direction(model, token, rivals)

        def push(_module, _inputs, output):
            output = output.clone()
            output[rows, at] = output[rows, at] + direction.to(output.dtype)
            return output

        handle = writer.register_forward_hook(push)
        try:
            pushed = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()[rows, at]
        finally:
            handle.remove()
        rival_logits = pushed.clone()
        rival_logits[:, token] = -1e9
        margin_before = scores[:, token] - others.max(-1).values
        margin_after = pushed[:, token] - rival_logits.max(-1).values
        sensitivity = float((margin_after - margin_before).mean())
        size = max(gap + margin, margin) / max(sensitivity, 1e-3)
        links.append(ChainLink(token, reads[rows, at].T, direction * size, where["wordings"], gap, sensitivity))
    return links

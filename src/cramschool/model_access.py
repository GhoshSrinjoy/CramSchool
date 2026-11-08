"""Find the parts of a transformer that edits touch, and turn text into token ids consistently."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

_LAYER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "model.text_model.layers",
    "model.model.layers",
)


class UnsupportedModelError(ValueError):
    """The model does not have the MLP layout this package edits."""


def _resolve(root: nn.Module, dotted: str):
    node = root
    for part in dotted.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    for path in _LAYER_PATHS:
        layers = _resolve(model, path)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    raise UnsupportedModelError(
        f"Could not find the decoder layers of {type(model).__name__}. Looked in: {', '.join(_LAYER_PATHS)}."
    )


def writer_module(model: nn.Module, layer: int) -> nn.Linear:
    """The matrix through which a block's MLP writes into the residual stream: `mlp.down_proj`.

    This is the matrix both kinds of edit act on. Hooking anything else was why the original scripts never changed a weight.
    """
    layers = decoder_layers(model)
    if not 0 <= layer < len(layers):
        raise IndexError(f"layer {layer} is out of range; this model has {len(layers)} layers")
    module = _resolve(layers[layer], "mlp.down_proj")
    if not isinstance(module, nn.Linear):
        raise UnsupportedModelError(
            f"{type(model).__name__} has no `mlp.down_proj` linear layer. Llama, Phi-3, Qwen and Gemma style models do; "
            "GPT-2 style models (which use Conv1D) are not supported."
        )
    return module


def reader_matrices(model: nn.Module, layer: int) -> dict[str, torch.Tensor]:
    """The weights through which a block's MLP reads the residual stream: {"gate": (neurons, hidden), "up": (neurons, hidden)}.

    Qwen, Llama and Gemma style models keep them as two matrices. Phi-3 keeps them in one, gate first, then up.
    """
    layers = decoder_layers(model)
    if not 0 <= layer < len(layers):
        raise IndexError(f"layer {layer} is out of range; this model has {len(layers)} layers")
    mlp = _resolve(layers[layer], "mlp")
    joined = getattr(mlp, "gate_up_proj", None)
    if isinstance(joined, nn.Linear):
        gate, up = joined.weight.chunk(2, dim=0)
        return dict(gate=gate, up=up)
    gate, up = getattr(mlp, "gate_proj", None), getattr(mlp, "up_proj", None)
    if isinstance(gate, nn.Linear) and isinstance(up, nn.Linear):
        return dict(gate=gate.weight, up=up.weight)
    raise UnsupportedModelError(f"{type(model).__name__} has no `gate_proj`/`up_proj` or `gate_up_proj` linear layers in its MLP.")


def writer_param_name(model: nn.Module, layer: int) -> str:
    target = writer_module(model, layer)
    for name, module in model.named_modules():
        if module is target:
            return f"{name}.weight"
    raise UnsupportedModelError("The writer module is not registered on the model.")  # pragma: no cover


def find_parameter(model: nn.Module, name: str) -> tuple[str, nn.Parameter]:
    """Look a parameter up by name, also accepting multimodal checkpoint names for a text-only model."""
    params = dict(model.named_parameters())
    for candidate in (name, name.replace("model.language_model.", "model."), name.replace("model.layers.", "model.language_model.layers.")):
        if candidate in params:
            return candidate, params[candidate]
    raise KeyError(f"The model has no parameter named {name!r}")


@dataclass(frozen=True)
class Prompt:
    """A piece of text ready for the tokenizer. `rendered` is True when it already went through a chat template."""

    text: str
    rendered: bool = False


def render_chat(tokenizer, question: str, thinking: bool = False) -> Prompt:
    """The model's own chat format, with reasoning switched off unless asked for. Falls back to Q/A without a template."""
    if not getattr(tokenizer, "chat_template", None):
        return Prompt(f"Question: {question}\nAnswer:")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )
    return Prompt(text, rendered=True)


def encode(tokenizer, prompt: Prompt) -> list[int]:
    return tokenizer(prompt.text, add_special_tokens=not prompt.rendered).input_ids


def pad_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("The tokenizer has neither a pad token nor an eos token.")
    return eos[0] if isinstance(eos, list) else eos


def batch_ids(tokenizer, sequences: list[list[int]], device, side: str = "right") -> tuple[torch.Tensor, torch.Tensor]:
    """Pad token id lists by hand, so batching never depends on the tokenizer's mutable `padding_side`."""
    width = max(map(len, sequences))
    pad = pad_id(tokenizer)
    ids, mask = [], []
    for seq in sequences:
        gap = width - len(seq)
        if side == "right":
            ids.append(seq + [pad] * gap)
            mask.append([1] * len(seq) + [0] * gap)
        else:
            ids.append([pad] * gap + seq)
            mask.append([0] * gap + [1] * len(seq))
    return torch.tensor(ids, device=device), torch.tensor(mask, device=device)


def token_index_of(tokenizer, prompt: Prompt, phrase: str) -> int:
    """Index of the last token of the last occurrence of `phrase` in the prompt."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required, because editing needs character offsets.")
    start = prompt.text.rfind(phrase)
    if start < 0:
        raise ValueError(f"{phrase!r} does not occur in the prompt {prompt.text!r}")
    end = start + len(phrase)
    offsets = tokenizer(prompt.text, add_special_tokens=not prompt.rendered, return_offsets_mapping=True)["offset_mapping"]
    last = None
    for index, (a, b) in enumerate(offsets):
        if a < end and b > start:
            last = index
    if last is None:  # pragma: no cover
        raise ValueError(f"No token covers {phrase!r} in {prompt.text!r}")
    return last


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Hash that is the same whether a checkpoint was loaded as bfloat16 or float32."""
    import hashlib

    bits = tensor.detach().cpu().to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
    return hashlib.sha256(bits).hexdigest()

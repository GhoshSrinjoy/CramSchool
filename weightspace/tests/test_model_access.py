import pytest
import torch
from transformers import AutoModelForCausalLM, Gemma3TextConfig, GPT2Config, LlamaConfig, Phi3Config, Qwen3_5TextConfig, Qwen3Config

from weightspace import UnsupportedModelError, writer_module
from weightspace.facts import capture_writer
from weightspace.model_access import Prompt, batch_ids, find_parameter, token_index_of, writer_param_name

SMALL = dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64)
CONFIGS = {
    "llama": lambda: LlamaConfig(**SMALL),
    "phi3": lambda: Phi3Config(**SMALL, pad_token_id=0),
    "qwen3": lambda: Qwen3Config(**SMALL, head_dim=8),
    "qwen3_5": lambda: Qwen3_5TextConfig(**SMALL, head_dim=8, layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8),
    "gemma3": lambda: Gemma3TextConfig(**SMALL, head_dim=8),
}


@pytest.mark.parametrize("arch", CONFIGS)
def test_finds_a_real_weight_matrix_on_every_supported_architecture(arch):
    """The original scripts hooked modules with no usable weight, so no parameter ever changed. This pins the fix."""
    model = AutoModelForCausalLM.from_config(CONFIGS[arch]())
    writer = writer_module(model, 1)
    assert writer.weight.shape == (32, 48)
    name = writer_param_name(model, 1)
    assert name.endswith("layers.1.mlp.down_proj.weight")
    assert dict(model.named_parameters())[name] is writer.weight


def test_gpt2_style_models_get_a_clear_error():
    model = AutoModelForCausalLM.from_config(GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=16))
    with pytest.raises(UnsupportedModelError):
        writer_module(model, 0)


def test_out_of_range_layer_is_refused(model):
    with pytest.raises(IndexError):
        writer_module(model, 99)


def test_multimodal_checkpoint_names_resolve_on_a_text_model(model):
    name, param = find_parameter(model, "model.language_model.layers.1.mlp.down_proj.weight")
    assert name == "model.layers.1.mlp.down_proj.weight"
    assert param is writer_module(model, 1).weight
    with pytest.raises(KeyError):
        find_parameter(model, "model.layers.1.not_a_thing.weight")


def test_capture_sees_the_matrix_and_leaves_no_hook_behind(model, tokenizer):
    writer = writer_module(model, 1)
    prompt = Prompt("the owner of acme is")
    reads, writes = capture_writer(model, tokenizer, prompt, 1)
    assert reads.shape == (5, 48) and writes.shape == (5, 32)
    assert torch.allclose(writer(reads), writes, atol=1e-5)
    assert len(writer._forward_hooks) == 0


def test_capture_keeps_nothing_between_calls(model, tokenizer):
    """The original hooks appended every activation of every generation step, forever."""
    writer = writer_module(model, 1)
    for _ in range(3):
        capture_writer(model, tokenizer, Prompt("the owner of acme is"), 1)
    assert len(writer._forward_hooks) == 0


def test_token_index_finds_the_last_mention(tokenizer):
    assert token_index_of(tokenizer, Prompt("acme is the owner of acme"), "acme") == 5
    with pytest.raises(ValueError):
        token_index_of(tokenizer, Prompt("the owner"), "acme")


def test_batching_does_not_depend_on_tokenizer_state(tokenizer):
    ids, mask = batch_ids(tokenizer, [[5, 6, 7], [5]], "cpu", side="left")
    assert ids.tolist() == [[5, 6, 7], [tokenizer.pad_token_id, tokenizer.pad_token_id, 5]]
    assert mask.tolist() == [[1, 1, 1], [0, 0, 1]]
    ids, mask = batch_ids(tokenizer, [[5, 6, 7], [5]], "cpu", side="right")
    assert ids.tolist() == [[5, 6, 7], [5, tokenizer.pad_token_id, tokenizer.pad_token_id]]

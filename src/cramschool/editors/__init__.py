"""Editor models: a bigger model that drives cramschool's tools, drafts wordings and writes blind benches.

One spec string picks the model:

    anthropic:claude-opus-5-5          Claude, through the Anthropic SDK (ANTHROPIC_API_KEY or an `ant auth login` profile)
    openai:<model>                     OpenAI (OPENAI_API_KEY)
    ollama:<model>                     Ollama on this machine (http://localhost:11434/v1)
    vllm:<model>                       a vLLM server (http://localhost:8000/v1)
    lmstudio:<model>                   LM Studio (http://localhost:1234/v1)
    llamacpp:<model>                   llama.cpp's server (http://localhost:8080/v1)
    openrouter:<model>                 OpenRouter (OPENROUTER_API_KEY)
    openai-compatible:<model>@<url>    any other OpenAI-compatible endpoint (key in CRAMSCHOOL_API_KEY)
    local:<path or Hugging Face id>    a transformers model loaded here

Any OpenAI-compatible spec takes `@<url>` to change the base URL, for example `ollama:qwen3:14b@http://gpu-box:11434/v1`. The environment
variable CRAMSCHOOL_EDITOR holds a default spec.
"""
from __future__ import annotations

import os

from .base import Editor, Reply, Session, ToolCall, ToolResult
from .loop import EditorRun, complete_json, draft_relation, run_editor

PROVIDERS = ("anthropic", "openai", "ollama", "vllm", "lmstudio", "llamacpp", "openrouter", "openai-compatible", "local")


def editor_from_spec(spec: str | None = None, **kwargs) -> Editor:
    """An editor model from a spec string such as "anthropic:claude-opus-5-5" or "ollama:qwen3:14b" (see the module docstring)."""
    spec = spec or os.environ.get("CRAMSCHOOL_EDITOR")
    if not spec:
        raise ValueError("no editor model given: pass a spec such as anthropic:claude-opus-5-5 or ollama:qwen3:14b, or set CRAMSCHOOL_EDITOR")
    provider, _, rest = spec.partition(":")
    provider = provider.strip().lower()
    if provider in ("claude", "anthropic"):
        from .anthropic import MODEL, AnthropicEditor

        return AnthropicEditor(rest or MODEL, **kwargs)
    if provider == "local":
        from .local import LocalEditor

        return LocalEditor(rest, **kwargs)
    from .openai_compat import PRESETS, from_preset

    if provider not in PRESETS:
        raise ValueError(f"unknown editor provider {provider!r}; use one of {', '.join(PROVIDERS)}")
    model, base_url = rest, None
    if "@http" in rest:
        model, _, url = rest.partition("@http")
        base_url = "http" + url
    if not model:
        raise ValueError(f"{spec!r}: give a model name after '{provider}:'")
    return from_preset(provider, model, base_url, **kwargs)


__all__ = ["Editor", "Session", "Reply", "ToolCall", "ToolResult", "EditorRun", "run_editor", "draft_relation", "complete_json",
           "editor_from_spec", "PROVIDERS"]

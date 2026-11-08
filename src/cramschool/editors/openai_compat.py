"""Any OpenAI-compatible chat endpoint as the editor model: OpenAI, Ollama, vLLM, LM Studio, llama.cpp's server, OpenRouter, or your own.

It speaks the chat-completions protocol (`POST {base_url}/chat/completions`, with `tools` for function calling) over the standard library,
so it needs no extra package. The model must support tool calling for agent sessions; plain completions work with any model.

    OpenAICompatEditor("qwen3:14b", base_url="http://localhost:11434/v1")          # Ollama
    OpenAICompatEditor("gpt-5", base_url="https://api.openai.com/v1", api_key=...)  # OpenAI
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .base import Editor, Reply, Session, ToolCall, ToolResult

# provider: (default base URL, environment variable holding the key, or None when no key is needed)
PRESETS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "vllm": ("http://localhost:8000/v1", None),
    "lmstudio": ("http://localhost:1234/v1", None),
    "llamacpp": ("http://localhost:8080/v1", None),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai-compatible": ("http://localhost:8000/v1", "CRAMSCHOOL_API_KEY"),
}


class EditorHTTPError(RuntimeError):
    pass


class OpenAICompatSession(Session):
    def __init__(self, editor: "OpenAICompatEditor", system: str, tools: list[dict]):
        self.editor = editor
        self.tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                        "parameters": t.get("input_schema", {"type": "object", "properties": {}})}} for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system}] if system else []

    def send(self, text: str | None = None, results: list[ToolResult] | None = None) -> Reply:
        for r in results or []:
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": r.content})
        if text is not None:
            self.messages.append({"role": "user", "content": text})
        body: dict[str, Any] = dict(model=self.editor.model, messages=self.messages)
        if self.tools:
            body["tools"] = self.tools
            body["tool_choice"] = "auto"
        if self.editor.max_tokens:
            body["max_tokens"] = self.editor.max_tokens
        if self.editor.temperature is not None:
            body["temperature"] = self.editor.temperature
        body.update(self.editor.extra)
        data = self.editor.post("/chat/completions", body)
        choice = data["choices"][0]
        message = choice.get("message") or {}
        assistant = {"role": "assistant", "content": message.get("content") or ""}
        calls = []
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_unparsed": args}
            calls.append(ToolCall(tc.get("id") or f"call_{len(self.messages)}_{i}", fn.get("name", ""), args if isinstance(args, dict) else {}))
        if calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in calls]
        self.messages.append(assistant)
        finish = choice.get("finish_reason") or ("tool_calls" if calls else "stop")
        reason = {"tool_calls": "tool_use", "stop": "end", "length": "max_tokens", "content_filter": "refusal"}.get(finish, finish)
        return Reply(assistant["content"], calls, "tool_use" if calls else reason, raw=data)


class OpenAICompatEditor(Editor):
    """An OpenAI-compatible endpoint. `api_key` defaults to the provider's environment variable (see PRESETS); local servers need none."""

    def __init__(self, model: str, *, base_url: str = "http://localhost:11434/v1", api_key: str | None = None, max_tokens: int | None = 4096,
                 temperature: float | None = None, timeout: float = 600.0, extra: dict | None = None, provider: str = "openai-compatible"):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.max_tokens, self.temperature, self.timeout, self.extra = max_tokens, temperature, timeout, dict(extra or {})
        self.name = f"{provider}:{model}"

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            raise EditorHTTPError(f"{self.name}: HTTP {error.code} from {self.base_url}{path}: {detail}") from error
        except urllib.error.URLError as error:
            raise EditorHTTPError(f"{self.name}: no answer from {self.base_url} ({error.reason}). Is the server running?") from error

    def session(self, system: str, tools: list[dict]) -> Session:
        return OpenAICompatSession(self, system, tools)


def from_preset(provider: str, model: str, base_url: str | None = None, **kwargs) -> OpenAICompatEditor:
    url, key_var = PRESETS[provider]
    key = kwargs.pop("api_key", None) or (os.environ.get(key_var) if key_var else None) or os.environ.get("CRAMSCHOOL_API_KEY")
    if provider in ("openai", "openrouter") and not key:
        raise ValueError(f"{provider} needs an API key: set {key_var}")
    return OpenAICompatEditor(model, base_url=base_url or os.environ.get("CRAMSCHOOL_BASE_URL") or url, api_key=key, provider=provider, **kwargs)

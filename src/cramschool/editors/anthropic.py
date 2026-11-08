"""Claude as the editor model, through the official Anthropic SDK (`pip install "cramschool[claude]"`).

Credentials come from the environment (ANTHROPIC_API_KEY, or an `ant auth login` profile). The default model is Claude Opus 5.5 with
adaptive thinking and effort "high" (its own default is "medium"). Each turn streams, and tool inputs stream as they are written, so every
input is checked against the toolbox's schema before it runs. The server-side refusal fallback is on by default (`fallbacks="default"`);
pass `fallbacks=None` to turn it off.
"""
from __future__ import annotations

from typing import Any

from .base import Editor, Reply, Session, ToolCall, ToolResult

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicSession(Session):
    def __init__(self, editor: "AnthropicEditor", system: str, tools: list[dict]):
        self.editor, self.system = editor, system
        self.tools = [dict(t, eager_input_streaming=True) for t in tools]
        self.messages: list[dict] = []

    def _request(self) -> Any:
        e = self.editor
        kwargs: dict[str, Any] = dict(model=e.model, max_tokens=e.max_tokens, messages=self.messages, output_config={"effort": e.effort})
        if self.system:
            kwargs["system"] = self.system
        if self.tools:
            kwargs["tools"] = self.tools
        if e.fallbacks is not None:
            api = e.client.beta.messages
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["extra_body"] = {"fallbacks": e.fallbacks}
        else:
            api = e.client.messages
        with api.stream(**kwargs) as stream:
            for _event in stream:
                pass
            return stream.get_final_message()

    def send(self, text: str | None = None, results: list[ToolResult] | None = None) -> Reply:
        if results:
            self.messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
                                                              for r in results]})
        if text is not None:
            self.messages.append({"role": "user", "content": text})
        for _attempt in range(3):
            try:
                response = self._request()
                break
            except ValueError:
                continue  # a tool input the SDK could not parse at all: ask for the turn again
        else:
            raise RuntimeError("the editor model's tool input could not be parsed three times in a row")
        # the whole content goes back unchanged (thinking blocks included), as the API requires
        self.messages.append({"role": "assistant", "content": response.content})
        calls = [ToolCall(b.id, b.name, dict(b.input) if isinstance(b.input, dict) else {}) for b in response.content if getattr(b, "type", None) == "tool_use"]
        text_out = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", None) == "text")
        reason = {"tool_use": "tool_use", "end_turn": "end", "max_tokens": "max_tokens", "refusal": "refusal"}.get(response.stop_reason, response.stop_reason)
        if reason == "pause_turn":
            return self.send()
        return Reply(text_out, calls, reason, raw=response)


class AnthropicEditor(Editor):
    """Claude. `model` defaults to Claude Opus 5.5; `client` defaults to `anthropic.Anthropic()`."""

    def __init__(self, model: str = MODEL, *, effort: str = "high", max_tokens: int = 32000, fallbacks: str | None = "default", client: Any = None):
        if client is None:
            try:
                import anthropic
            except ImportError as error:  # pragma: no cover - depends on the environment
                raise ImportError('Claude needs the Anthropic SDK: pip install "cramschool[claude]"') from error
            client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens, self.fallbacks, self.client = model, effort, max_tokens, fallbacks, client
        self.name = f"anthropic:{model}"

    def session(self, system: str, tools: list[dict]) -> Session:
        return AnthropicSession(self, system, tools)

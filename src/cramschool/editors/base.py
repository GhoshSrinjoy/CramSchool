"""The editor-model interface: a bigger model that reads tool descriptions, calls tools, and reads their results.

Every adapter turns its provider's format into the same three things: `Reply` (text, tool calls, why it stopped), `ToolCall` and
`ToolResult`. A conversation is a `Session`; `run_editor` (in `cramschool.editors.loop`) drives a session against a toolbox.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str          # JSON text of the tool's result
    is_error: bool = False


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end"   # "end", "tool_use", "max_tokens", "refusal" or the provider's own reason
    raw: Any = None


class Session:
    """One conversation with an editor model. `send` adds a user message or tool results and returns the model's reply."""

    def send(self, text: str | None = None, results: list[ToolResult] | None = None) -> Reply:  # pragma: no cover - interface
        raise NotImplementedError


class Editor:
    """A bigger model. `session(system, tools)` starts a conversation in which it may call the given tools."""

    name: str = "editor"

    def session(self, system: str, tools: list[dict]) -> Session:  # pragma: no cover - interface
        raise NotImplementedError

    def complete(self, prompt: str, system: str = "") -> str:
        """One plain request and the reply's text, with no tools."""
        return self.session(system, []).send(prompt).text


def tool_json(result: dict) -> str:
    return json.dumps(result, default=str, ensure_ascii=False)


def first_json_object(text: str) -> dict | None:
    """The first balanced {...} in a text that parses as JSON, or None. Used to read tool calls from models without native tool calling."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:i + 1])
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None

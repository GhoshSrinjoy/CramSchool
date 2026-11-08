"""Claude as the big model: it reads, decides and edits through a `Surgery`'s tools, the way the published runs used a frontier model.

Needs the Anthropic SDK (`pip install "cramschool[claude]"`) and a key in ANTHROPIC_API_KEY. The surgery enforces the rules, so Claude can
only act through its tools: forward passes from a budget, no gradients, every edit logged and undoable.

    from cramschool.claude_editor import run_with_claude
    run = run_with_claude(surgery, "Find out why the model loops on maths problems, and fix it if the evidence supports an edit.")
    print(run.final_text)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .surgery import PROTOCOL, Surgery

MODEL = "claude-opus-5-5"


@dataclass
class EditorRun:
    final_text: str
    turns: int
    stop_reason: str
    calls: list[dict] = field(default_factory=list)


def _blocks(content) -> tuple[list, str]:
    tool_uses = [b for b in content if getattr(b, "type", None) == "tool_use"]
    text = "".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")
    return tool_uses, text


def run_with_claude(
    surgery: Surgery,
    task: str,
    *,
    model: str = MODEL,
    effort: str = "high",
    max_turns: int = 80,
    max_tokens: int = 32000,
    client: Any = None,
    on_call: Callable[[str, dict, dict], None] | None = None,
) -> EditorRun:
    """Let Claude operate until it stops calling tools, refuses, or runs out of turns. Returns what it said at the end and every call it made.

    The loop streams each turn, since turns can be long. Tool inputs stream as Claude writes them, so the server does not check them
    against the schema; the surgery checks every input before running it and answers bad ones with an error result.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    tools = [dict(schema, eager_input_streaming=True) for schema in surgery.tool_definitions()]
    messages: list[dict] = [{"role": "user", "content": task}]
    calls: list[dict] = []
    retries = 0
    text, stop_reason, turn = "", "max_turns", 0
    for turn in range(1, max_turns + 1):
        try:
            with client.messages.stream(model=model, max_tokens=max_tokens, system=PROTOCOL, tools=tools, messages=messages, output_config={"effort": effort}) as stream:
                for _event in stream:
                    pass
                response = stream.get_final_message()
            retries = 0
        except ValueError:
            # Tool input the SDK could not parse at all: there is no complete tool_use block to answer, so ask for the turn again (bounded).
            retries += 1
            if retries > 2:
                raise
            continue
        stop_reason = response.stop_reason
        if stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue
        tool_uses, text = _blocks(response.content)
        if stop_reason == "refusal" or not tool_uses:
            break
        if stop_reason == "max_tokens":
            raise RuntimeError("A tool call was cut off by max_tokens. Run again with a larger max_tokens.")
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in tool_uses:
            problem = surgery.validate(block.name, block.input)
            if problem:
                outcome = dict(error=problem, INVALID_INPUT=json.dumps(block.input, default=str)[:2000])
            else:
                outcome = surgery.call(block.name, block.input)
            calls.append(dict(tool=block.name, input=block.input, result=outcome))
            if on_call is not None:
                on_call(block.name, block.input, outcome)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(outcome, default=str), "is_error": "error" in outcome})
        messages.append({"role": "user", "content": results})
    return EditorRun(text, turn, stop_reason, calls)

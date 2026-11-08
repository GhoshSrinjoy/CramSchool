"""Driving a toolbox with an editor model, and the editor's other jobs: drafting wordings for a new relation.

A toolbox is anything with `tool_definitions()` (name, description, input_schema), `validate(name, args)` (an error text or None),
`call(name, args)` (a JSON-able dict) and `protocol` (the system prompt). `cramschool.Surgery` (weight edits) and `cramschool.tools.GraftTools`
(facts in a GRAFT table) are toolboxes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from ..graft.wordings import Relation, canonical_relation
from .base import Editor, ToolResult, first_json_object, tool_json


@dataclass
class EditorRun:
    final_text: str
    turns: int
    stop_reason: str
    calls: list[dict] = field(default_factory=list)


def run_editor(editor: Editor, toolbox, task: str, *, max_turns: int = 80, on_call: Callable[[str, dict, dict], None] | None = None) -> EditorRun:
    """Let the editor work on `task` with the toolbox until it stops calling tools, refuses, or runs out of turns. Every input is validated
    before it runs; a bad input comes back to the editor as an error result and costs nothing."""
    session = editor.session(getattr(toolbox, "protocol", ""), toolbox.tool_definitions())
    reply = session.send(task)
    calls: list[dict] = []
    turn = 1
    while reply.tool_calls and reply.stop_reason not in ("refusal",) and turn < max_turns:
        results = []
        for call in reply.tool_calls:
            problem = toolbox.validate(call.name, call.arguments)
            outcome = dict(error=problem) if problem else toolbox.call(call.name, call.arguments)
            calls.append(dict(tool=call.name, input=call.arguments, result=outcome))
            if on_call is not None:
                on_call(call.name, call.arguments, outcome)
            results.append(ToolResult(call.id, call.name, tool_json(outcome), is_error="error" in outcome))
        reply = session.send(results=results)
        turn += 1
    stop = reply.stop_reason if not reply.tool_calls else "max_turns"
    return EditorRun(reply.text, turn, stop, calls)


DRAFT_PROMPT = """Write wordings for the relation "{name}": {description}

They teach a small language model when a question about a company asks for this relation. Use {{s}} for the company, exactly once in
every wording. Write {n_plain} "plain" sentence openings that stop exactly where the answer would come next (no trailing space), and
{n_chat} "chat" questions a user might type to an assistant. Vary structure, length and register. None may contain an answer, and none
may ask for a different relation (owner, CEO, founder, head office, or anything else).

Reply with one JSON object only: {{"plain": [...], "chat": [...]}}"""


def draft_relation(editor: Editor, name: str, description: str, *, n_plain: int = 30, n_chat: int = 30, person: bool = True) -> Relation:
    """Ask an editor model to draft a new relation's wordings, then check them (each has {s} exactly once; enough of each kind)."""
    text = editor.complete(DRAFT_PROMPT.format(name=name, description=description, n_plain=n_plain, n_chat=n_chat))
    data = first_json_object(text)
    if not data or not isinstance(data.get("plain"), list) or not isinstance(data.get("chat"), list):
        raise ValueError(f"the editor's reply holds no {{'plain': [...], 'chat': [...]}} object: {text[:300]!r}")
    keep = lambda xs: list(dict.fromkeys(x.rstrip() for x in xs if isinstance(x, str) and x.count("{s}") == 1))  # noqa: E731
    return Relation(name=canonical_relation(name), plain=keep(data["plain"]), chat=keep(data["chat"]), person=person)


def complete_json(editor: Editor, prompt: str, system: str = "") -> dict:
    """One request whose reply must hold a JSON object."""
    text = editor.complete(prompt, system)
    obj = first_json_object(text)
    if obj is None:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"the editor's reply holds no JSON object: {text[:300]!r}") from error
    return obj

"""Letting a bigger model steer the edits. Any function from prompt text to reply text works as the bigger model.

The bigger model never writes weights here. It turns a sentence into structured prompts and picks settings from a fixed menu, and
every result still has to pass the same gates. That keeps its mistakes from reaching the weights unchecked.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Callable

from .editor import EditResult, KnowledgeEditor
from .facts import EditSettings, Fact
from .gates import GateReport

LLM = Callable[[str], str]


class DraftError(ValueError):
    """The bigger model's reply could not be used."""


DRAFT_PROMPT = """Turn the sentence below into a JSON object used to test and teach a language model one fact.

Sentence: {sentence}

Return only JSON with these keys. Every value except "subject" and "target" is a list of strings.
- "subject": the thing a question would ask about, copied exactly from the sentence.
- "target": only the short answer to that question, such as a name or a place, copied exactly from the sentence. Nothing else.
- "prompts": 3 unfinished plain-text sentences that the target would naturally complete. Each contains the subject and stops right before the answer.
- "questions": 1 direct question whose answer is the target.
- "heldout_prompts": 2 more unfinished sentences that the target would complete, worded differently from "prompts".
- "heldout_questions": 1 more question whose answer is the target, worded differently from "questions".
- "leak_prompts": 2 unfinished sentences about the subject that ask for a different kind of information, so the target would be a wrong completion.
- "leak_questions": 1 question about the subject that asks for something else.

Rules: every prompt and question must contain the subject exactly. No prompt or question may contain the target.

Example, for the sentence "Ada Lovelace founded Acme Corp":
{{"subject": "Acme Corp", "target": "Ada Lovelace",
"prompts": ["The founder of Acme Corp is", "Acme Corp was founded by", "The person who started Acme Corp is"],
"questions": ["Who founded Acme Corp?"],
"heldout_prompts": ["Acme Corp owes its existence to its founder,", "If you ask who started Acme Corp, the answer is"],
"heldout_questions": ["Name the founder of Acme Corp."],
"leak_prompts": ["Acme Corp is headquartered in", "The main product of Acme Corp is"],
"leak_questions": ["What does Acme Corp sell?"]}}"""


def _first_json_object(reply: str) -> dict:
    start = reply.find("{")
    if start < 0:
        raise DraftError(f"No JSON object in the reply: {reply[:200]!r}")
    try:
        value, _ = json.JSONDecoder().raw_decode(reply[start:])
    except json.JSONDecodeError as error:
        raise DraftError(f"The reply is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise DraftError("The reply is not a JSON object.")
    return value


def draft_fact(llm: LLM, sentence: str) -> Fact:
    """Turn a plain sentence into a Fact with teaching and test prompts. Raises DraftError if the reply cannot be used."""
    data = _first_json_object(llm(DRAFT_PROMPT.format(sentence=sentence)))
    fields = ("subject", "target", "prompts", "questions", "heldout_prompts", "heldout_questions", "leak_prompts", "leak_questions")
    missing = [f for f in fields if not data.get(f)]
    if missing:
        raise DraftError(f"The reply is missing: {', '.join(missing)}")
    if not isinstance(data["subject"], str) or not isinstance(data["target"], str):
        raise DraftError("The subject and target must be strings.")
    subject, target = data["subject"].strip(), data["target"].strip()
    if subject not in sentence or target not in sentence:
        raise DraftError("The subject and target must be copied exactly from the sentence.")
    if subject in target or target in subject:
        raise DraftError(f"The subject {subject!r} and target {target!r} overlap, so they cannot be a question and its answer.")
    data["subject"], data["target"] = subject, target
    for name in fields[2:]:
        # Models often answer a one-item list with a bare string, and a trailing space would split the answer's first token.
        value = data[name]
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise DraftError(f"{name} must be a list of strings.")
        data[name] = [item.strip() for item in items if item.strip()]
    # A prompt used for teaching proves nothing as a test, so repeats are removed from the test set.
    data["heldout_prompts"] = [p for p in data["heldout_prompts"] if p not in data["prompts"]]
    data["heldout_questions"] = [q for q in data["heldout_questions"] if q not in data["questions"]]
    if not data["heldout_prompts"] and not data["heldout_questions"]:
        raise DraftError("Every held-out prompt repeats a teaching prompt, so nothing is left to test with.")
    try:
        return Fact(**{f: data[f] for f in fields}, statement=sentence)
    except (TypeError, ValueError) as error:
        raise DraftError(str(error)) from error


REVIEW_PROMPT = """An edit that writes a fact into a language model's weights failed its checks.

Fact: {statement}
Attempts so far (settings, then result):
{history}

Pick the settings to try next from this menu, by number. Lower layers generalise further but leak more; a higher drift penalty reduces leaks but can reduce recall.
{menu}

Reply with only the number, or "stop" if none is likely to help."""


def choose_next(llm: LLM | None, fact: Fact, tried: list[tuple[EditSettings, GateReport]], menu: list[EditSettings]) -> EditSettings | None:
    """Next settings to try. With no bigger model, or a reply outside the menu, take the next untried entry in order."""
    remaining = [m for m in menu if m not in [t[0] for t in tried]]
    if not remaining:
        return None
    if llm is None:
        return remaining[0]
    history = "\n".join(f"{asdict(s)} -> {r.to_dict()}" for s, r in tried)
    listing = "\n".join(f"{i}: {asdict(m)}" for i, m in enumerate(menu))
    reply = llm(REVIEW_PROMPT.format(statement=fact.statement, history=history, menu=listing)).strip().lower()
    if reply.startswith("stop"):
        return None
    digits = "".join(c for c in reply.split()[0] if c.isdigit()) if reply else ""
    if digits and int(digits) < len(menu) and menu[int(digits)] in remaining:
        return menu[int(digits)]
    return remaining[0]


def add_fact_with_review(editor: KnowledgeEditor, fact: Fact, menu: list[EditSettings], llm: LLM | None = None, max_attempts: int = 3) -> tuple[EditResult, list[EditResult]]:
    """Try settings until an edit passes its gates. Returns (last attempt, all attempts). If none passes, the weights are unchanged."""
    attempts: list[EditResult] = []
    tried: list[tuple[EditSettings, GateReport]] = []
    settings = menu[0]
    while settings is not None and len(attempts) < max_attempts:
        result = editor.add_fact(fact, settings=settings)
        attempts.append(result)
        tried.append((settings, result.report))
        if result.accepted:
            break
        settings = choose_next(llm, fact, tried, menu)
    return attempts[-1], attempts


class LocalLLM:
    """A local Hugging Face model as the bigger model. For Claude, GPT or anything else, pass your own function instead."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 700):
        self.model, self.tokenizer, self.max_new_tokens = model, tokenizer, max_new_tokens

    def __call__(self, prompt: str) -> str:
        from .gates import generate_batch
        from .model_access import render_chat

        return generate_batch(self.model, self.tokenizer, [render_chat(self.tokenizer, prompt)], self.max_new_tokens)[0]

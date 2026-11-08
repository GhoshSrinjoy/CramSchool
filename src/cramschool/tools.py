"""GRAFT as tools for an editor model: through `cramschool.editors.run_editor`, or the MCP server (`cramschool serve`).

The toolbox keeps one `Graft` table on the loaded model. Every tool takes and returns plain JSON, and calls run one at a time.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from .datasets import load_facts
from .graft import Graft, Relation
from .graft.wordings import RELATION_NAMES, canonical_relation

PROTOCOL = """You are teaching a small language model new facts with GRAFT. The facts live in one inserted layer beside the model; its
weights never change, and removing a fact is exact.

How to work:
1. Look at what the model answers now (ask) before adding anything.
2. Add each fact with add_fact (about 40 seconds each). The relations owner, ceo, founder and hq are built in. For any other relation,
   first give its wordings with add_relation: at least 8 plain sentence openings and 4 chat questions, each with {s} for the company.
3. Check every fact with check_fact on wordings you write now, which the build never saw: about 8 new questions for the fact, 4 other
   questions about the same company (its size, products, other people), and 2 or 3 look-alike company names (one letter changed, the same
   first word in another industry, another first word in the same industry).
4. A fact is good when the answer comes back on at least 3 in 4 new questions and never for look-alikes or other questions. Remove a fact
   that leaks (remove_fact), or report it.
5. Save the table when done (save), and report what you added, the evidence for each fact, and what is still uncertain.
"""


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


STR, BOOL, INT = {"type": "string"}, {"type": "boolean"}, {"type": "integer"}
LIST = {"type": "array", "items": {"type": "string"}}
TOOLS = {
    "status": ("The table: how many facts, which relations it knows, the layer, and whether it is attached.", _schema({}, [])),
    "list_facts": ("Every stored fact: id, company, relation, answer.", _schema({}, [])),
    "add_fact": ("Add a fact, or update it if the company already has one for this relation. About 40 seconds.",
                 _schema(dict(subject=dict(STR, description="The company, exactly as it is written, e.g. 'Nimbresco Software'"),
                              relation=dict(STR, description="owner, ceo, founder, hq, or a relation added with add_relation"),
                              answer=dict(STR, description="The answer, e.g. 'Tharindu Ekanayake' or 'Oradea'")), ["subject", "relation", "answer"])),
    "remove_fact": ("Remove a company's fact for one relation, or all its facts.",
                    _schema(dict(subject=STR, relation=dict(STR, description="Leave out to remove all the company's facts")), ["subject"])),
    "ask": ("Ask the model a question, with and without the GRAFT layer.",
            _schema(dict(question=STR, chat=dict(BOOL, description="Chat question (default) or a plain sentence opening")), ["question"])),
    "check_fact": ("Check a stored fact on wordings the build never saw: recall on new questions, and leaks on other questions about the "
                   "company and on look-alike companies.",
                   _schema(dict(subject=STR, relation=STR,
                                new_questions=dict(LIST, description="New ways to ask for this fact (chat questions or plain openings)"),
                                other_questions=dict(LIST, description="Other questions about the same company; the answer must not appear"),
                                lookalike_companies=dict(LIST, description="Similar company names; the answer must not appear for them")),
                           ["subject", "relation", "new_questions"])),
    "add_relation": ("Teach the table a new relation: at least 8 plain openings and 4 chat questions, each with {s} once for the company.",
                     _schema(dict(name=STR, plain=LIST, chat=LIST, person=dict(BOOL, description="True if the answer is a person's name")),
                             ["name", "plain", "chat"])),
    "add_facts_from_dataset": ("Add facts from a CSV, JSON or JSONL file, a bench file, or a Hugging Face dataset ('hf:<name>'). About 40 "
                               "seconds per fact.",
                               _schema(dict(source=STR, subject_column=STR, relation_column=STR, answer_column=STR, fixed_relation=STR, limit=INT),
                                       ["source"])),
    "preview_dataset": ("The first facts a dataset would add, without adding them.",
                        _schema(dict(source=STR, subject_column=STR, relation_column=STR, answer_column=STR, fixed_relation=STR, limit=INT), ["source"])),
    "save": ("Save the table to a folder, to load later onto the same model.", _schema(dict(path=STR), ["path"])),
}
_TYPES = {"string": str, "boolean": bool, "integer": int, "array": list, "object": dict}


class GraftTools:
    """The GRAFT toolbox. `graft` is a `Graft`, or a function that returns one (so the model can load on first use)."""

    protocol = PROTOCOL

    def __init__(self, graft: Graft | Callable[[], Graft]):
        self._graft = graft
        self._lock = threading.Lock()

    @property
    def graft(self) -> Graft:
        g = self._graft
        return g() if callable(g) and not hasattr(g, "add") else g

    def tool_definitions(self) -> list[dict]:
        return [dict(name=name, description=desc, input_schema=schema) for name, (desc, schema) in TOOLS.items()]

    def validate(self, name: str, args) -> str | None:
        if name not in TOOLS:
            return f"Unknown tool {name!r}. Tools: {', '.join(TOOLS)}."
        if not isinstance(args, dict):
            return "Arguments must be a JSON object."
        schema = TOOLS[name][1]
        missing = [k for k in schema["required"] if k not in args]
        unknown = [k for k in args if k not in schema["properties"]]
        if missing or unknown:
            return f"{name}: missing {missing or 'nothing'}, unknown {unknown or 'nothing'}."
        for key, value in args.items():
            expected = schema["properties"][key].get("type")
            if expected in _TYPES and (not isinstance(value, _TYPES[expected]) or (expected == "integer" and isinstance(value, bool))):
                return f"{name}: {key} must be a {expected}."
        return None

    def call(self, name: str, args: dict | None = None) -> dict:
        args = args or {}
        problem = self.validate(name, args)
        if problem:
            return dict(error=problem)
        with self._lock:
            t0 = time.time()
            try:
                out = getattr(self, f"_{name}")(**args)
            except (KeyError, ValueError) as error:
                return dict(error=str(error).strip("'\""))
            out["seconds"] = round(time.time() - t0, 1)
            return out

    # ------------------------------------------------------------ the tools

    def _status(self) -> dict:
        g = self.graft
        return dict(facts=len(g), layer=g.layer, attached=g.is_attached, relations=list(g.wordings.relations), options=vars(g.options))

    def _list_facts(self) -> dict:
        return dict(facts=[vars(f) for f in self.graft.facts])

    def _add_fact(self, subject: str, relation: str, answer: str) -> dict:
        g = self.graft
        stored = g.add(subject, canonical_relation(relation), answer)
        rel = g.wordings.relations[stored.relation]
        probe = rel.chat[0].format(s=stored.subject)
        return dict(added=vars(stored), quick_look=dict(note="a training wording, for a quick look only; use check_fact on new wordings",
                                                        question=probe, reply=g.ask(probe)))

    def _remove_fact(self, subject: str, relation: str | None = None) -> dict:
        return dict(removed=self.graft.remove(subject, relation), facts=len(self.graft))

    def _ask(self, question: str, chat: bool = True) -> dict:
        g = self.graft
        out = g.compare(question, chat=chat) if len(g) else dict(question=question, without=g.ask(question, chat=chat), with_graft=None)
        out["company_found"] = g.find(question) if len(g) else None
        return out

    def _check_fact(self, subject: str, relation: str, new_questions: list[str], other_questions: list[str] | None = None,
                    lookalike_companies: list[str] | None = None) -> dict:
        g = self.graft
        rel = canonical_relation(relation)
        fact = next((f for f in g.facts if f.subject.lower() == subject.lower() and f.relation == rel), None)
        if fact is None:
            raise KeyError(f"no stored fact for {subject!r} and {rel!r}")
        answer, first = fact.answer.lower(), fact.answer.split()[0].lower()

        def run(questions):
            return [dict(question=q, reply=g.ask(q, chat=q.rstrip().endswith(("?", ".")))) for q in questions]

        recall = run(new_questions)
        for r in recall:
            r["recalled"] = answer in r["reply"].lower()
        others = run(other_questions or [])
        for r in others:
            r["leaked"] = first in r["reply"].lower()
        looks = []
        for company in lookalike_companies or []:
            q = g.wordings.relations[rel].chat[0].format(s=company)
            looks.append(dict(company=company, question=q, reply=g.ask(q)))
            looks[-1]["leaked"] = first in looks[-1]["reply"].lower()
        n_rec = sum(r["recalled"] for r in recall)
        return dict(fact=vars(fact), relation_name=RELATION_NAMES.get(rel, rel), recall=f"{n_rec} of {len(recall)}",
                    leaks_other_questions=f"{sum(r['leaked'] for r in others)} of {len(others)}",
                    leaks_lookalikes=f"{sum(r['leaked'] for r in looks)} of {len(looks)}",
                    good=n_rec >= 0.75 * len(recall) and not any(r["leaked"] for r in others + looks), recall_replies=recall, other_replies=others,
                    lookalike_replies=looks)

    def _add_relation(self, name: str, plain: list[str], chat: list[str], person: bool = True) -> dict:
        rel = Relation(name=canonical_relation(name), plain=plain, chat=chat, person=person)
        self.graft.add_relation(rel)
        return dict(added_relation=rel.name, plain=len(rel.plain), chat=len(rel.chat))

    def _records(self, source, subject_column="subject", relation_column="relation", answer_column="answer", fixed_relation=None, limit=None):
        return load_facts(source, subject=subject_column, relation=relation_column, answer=answer_column, fixed_relation=fixed_relation, limit=limit)

    def _preview_dataset(self, source: str, limit: int | None = 10, **cols) -> dict:
        recs = self._records(source, limit=limit, **cols)
        return dict(facts=[vars(r) for r in recs], shown=len(recs))

    def _add_facts_from_dataset(self, source: str, limit: int | None = None, **cols) -> dict:
        recs = self._records(source, limit=limit, **cols)
        added = [vars(self.graft.add(r.subject, r.relation, r.answer)) for r in recs]
        return dict(added=added, facts=len(self.graft))

    def _save(self, path: str) -> dict:
        return dict(saved=self.graft.save(path), facts=len(self.graft))

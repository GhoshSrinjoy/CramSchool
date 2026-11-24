"""GRAFT as tools for an editor model: through `cramschool.editors.run_editor`, or the MCP server (`cramschool serve`).

The toolbox keeps one `Graft` table on the loaded model. Every tool takes and returns plain JSON, and calls run one at a time.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable

from .datasets import load_facts
from .graft import Graft, Relation
from .graft.wordings import RELATION_NAMES, canonical_relation

PROTOCOL = """You are teaching a small language model new facts with GRAFT.

A fact is a named subject, a relation and a short answer: "Vorantis Biotech | ceo | Ingrid Solberg", "Tavrik DB | default_port | 7741",
"Zentharite | melting_point | 1412 °C". GRAFT holds such facts in two forms. The table sits beside the model and changes no weight;
removing a fact from it is exact. The grafted model is the same facts written into a copy of the model's own weights (commit_grafted).
GRAFT does not hold free text, programs, procedures or ways of behaving: only short answers about named subjects.

Every subject has a kind. Four relations about companies are built in: owner, ceo, founder and hq. Other kinds of subject are ready
to use, and status lists them: person, town, product, work, concept, institution, medicine and event. A kind that is missing, such as
software or a chemical substance, is added with add_kind. Any other relation is added with add_relation, with the kind of its subject.

How to work:
1. Look at what the model answers now (ask) before adding anything.
2. For a relation that is not built in, first give its wordings with add_relation: at least 8 plain sentence openings that stop where
   the answer comes next, and 4 chat questions, each with {s} once where the subject's name goes; 30 of each is better. Name the kind
   of its subject, and say whether the answer is a person's name. Two relations about the same kind of subject must not ask in
   nearly the same words.
3. Add each fact with add_fact (about 40 seconds each).
4. Check every fact with check_fact on wordings you write now, which the build never saw: about 8 new questions for the fact, 4 other
   questions about the same subject that ask for something else, and 2 or 3 look-alike names (one letter changed, the same first word
   with another ending, another first word with the same ending). Among the other questions, put some that are close to the fact: a
   fact is most likely to answer a question that asks for the same sort of thing (another date, another person, another setting).
5. A fact is good when the answer comes back on at least 3 in 4 new questions and never for look-alikes or other questions. Remove a
   fact that leaks (remove_fact), or report it.
6. Fix what misses. When facts of a relation miss new questions, look at what the missed questions have in common (where the subject's
   name stands, what comes just before the answer, the register), and teach the relation those ways of asking with teach_wordings:
   several wordings of each kind, never one that would also fit another relation. Then call rebuild once, and check again with
   questions you have not used before: a wording you taught proves nothing about itself.
7. Record your work. commit saves the table and records a version you can go back to (revert). commit_grafted also writes the grafted
   model and records its weights. Two grafted models are joined into one with weights_merge. Report what you added, the evidence for
   each fact, and what is still uncertain.
"""

# The protocol of versions up to 1.2, which speaks of companies only. The editor runs of the lab notebook's step 11 used it; it is
# kept so that those runs can be repeated.
PROTOCOL_V1 = """You are teaching a small language model new facts with GRAFT. The facts live in one inserted layer beside the model; its
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
5. Fix what misses. When facts of a relation miss new questions, look at what the missed questions have in common (where the company's
   name stands, what comes just before the answer, the register), and teach the relation those ways of asking with teach_wordings:
   several wordings of each kind, never one that would also fit another relation. Then call rebuild once, and check again with questions
   you have not used before: a wording you taught proves nothing about itself.
6. Save the table when done (save), and report what you added, the evidence for each fact, and what is still uncertain.
"""


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


STR, BOOL, INT = {"type": "string"}, {"type": "boolean"}, {"type": "integer"}
LIST = {"type": "array", "items": {"type": "string"}}
TOOLS = {
    "status": ("The table: how many facts, the relations it knows and the kind of subject of each, the kinds of subject ready to use, the "
               "layer, and whether it is attached.", _schema({}, [])),
    "list_facts": ("Every stored fact: id, subject, relation, answer.", _schema({}, [])),
    "add_fact": ("Add a fact, or update it if the subject already has one for this relation. About 40 seconds.",
                 _schema(dict(subject=dict(STR, description="The subject's name, exactly as it is written, e.g. 'Nimbresco Software' or 'Tavrik DB'"),
                              relation=dict(STR, description="owner, ceo, founder, hq, or a relation added with add_relation"),
                              answer=dict(STR, description="The short answer, e.g. 'Tharindu Ekanayake', 'Oradea' or '7741'")), ["subject", "relation", "answer"])),
    "remove_fact": ("Remove a subject's fact for one relation, or all its facts.",
                    _schema(dict(subject=STR, relation=dict(STR, description="Leave out to remove all the subject's facts")), ["subject"])),
    "ask": ("Ask the model a question, with and without the table. Before any fact is stored it gives the model's own reply.",
            _schema(dict(question=STR, chat=dict(BOOL, description="Chat question (default) or a plain sentence opening")), ["question"])),
    "check_fact": ("Check a stored fact on wordings the build never saw: recall on new questions, and leaks on other questions about the "
                   "same subject and on look-alike names. The answer is matched as whole words, and a leak is counted only where the table "
                   "brought the answer in: every question is also asked without the table.",
                   _schema(dict(subject=STR, relation=STR,
                                new_questions=dict(LIST, description="New ways to ask for this fact, each written out with the subject's name (chat questions, or "
                                                                       "plain openings that stop where the answer comes)"),
                                other_questions=dict(LIST, description="Other questions about the same subject; the answer must not appear"),
                                lookalike_subjects=dict(LIST, description="Similar names; the answer must not appear for them"),
                                lookalike_companies=dict(LIST, description="The earlier name of lookalike_subjects; still accepted")),
                           ["subject", "relation", "new_questions"])),
    "add_relation": ("Teach the table a new relation: at least 8 plain openings and 4 chat questions, each with {s} once for the subject.",
                     _schema(dict(name=STR, plain=LIST, chat=LIST, person=dict(BOOL, description="True if the answer is a person's name"),
                                  subject=dict(STR, description="the kind of subject: company (default), or a kind status lists, such as "
                                                                "person, product or concept; add_kind adds another")),
                             ["name", "plain", "chat"])),
    "add_kind": ("Teach the table a new kind of subject, for relations about subjects that are neither companies nor a kind status lists. "
                 "Give invented names of such subjects (at least 24, none of them real) and general questions about any such subject, "
                 "each with {s} once: at least 8 plain openings and 4 chat questions. A fact is taught to stay shut on them.",
                 _schema(dict(name=dict(STR, description="The kind's name, e.g. 'planet' or 'protein'"), names=LIST, generic_plain=LIST, generic_chat=LIST),
                         ["name", "names", "generic_plain", "generic_chat"])),
    "teach_wordings": ("Teach a relation more ways of asking, when check_fact shows questions its facts miss. Plain wordings stop where the "
                       "answer comes next; chat wordings are questions; each has {s} once for the subject. Call rebuild afterwards.",
                       _schema(dict(relation=STR, plain=LIST, chat=LIST), ["relation"])),
    "rebuild": ("Build the stored facts again, so that they learn what was taught since they were added: every fact, or only the facts of "
                "one relation. About 10 to 40 seconds per fact.",
                _schema(dict(relation=dict(STR, description="Only this relation's facts; leave out to build every fact again")), [])),
    "add_facts_from_dataset": ("Add facts from a CSV, JSON or JSONL file, a bench file, or a Hugging Face dataset ('hf:<name>'). About 40 "
                               "seconds per fact.",
                               _schema(dict(source=STR, subject_column=STR, relation_column=STR, answer_column=STR, fixed_relation=STR, limit=INT),
                                       ["source"])),
    "preview_dataset": ("The first facts a dataset would add, without adding them.",
                        _schema(dict(source=STR, subject_column=STR, relation_column=STR, answer_column=STR, fixed_relation=STR, limit=INT), ["source"])),
    "save": ("Save the table to a folder, to load later onto the same model.", _schema(dict(path=STR), ["path"])),
    "commit": ("Save the table to a folder and record it as a version, with a short message. Any recorded version can be compared with "
               "another (diff) or gone back to (revert). An unchanged table records nothing.",
               _schema(dict(path=STR, message=dict(STR, description="What changed, in a few words")), ["path"])),
    "versions": ("The versions recorded for the table in a folder: id, time, number of facts and message; and whether the history is whole.",
                 _schema(dict(path=STR), ["path"])),
    "diff": ("What changed between two recorded versions of the table in a folder: facts added, removed, changed and built again, and "
             "settings. Without b, from version a to the table as the folder holds it now.",
             _schema(dict(path=STR, a=dict(STR, description="A version id, such as v1"), b=dict(STR, description="A later version id")), ["path", "a"])),
    "revert": ("Go back to a recorded version: the folder and the loaded table are that version again, and the step is recorded as a new "
               "version. Nothing is built. Changes to the loaded table since it was last saved are dropped.",
               _schema(dict(path=STR, version=dict(STR, description="A version id, such as v1"),
                            discard=dict(BOOL, description="Also drop changes saved to the folder that no version holds")), ["path", "version"])),
    "commit_grafted": ("Record both forms of the facts in one call: the table as a version, the same facts written into a copy of the model's "
                       "own weights (a grafted model, in grafted_dir), and those weights as a version that names the table's. The first "
                       "export of a table takes about five minutes, since the facts are also built at the last layer, where the grafted "
                       "model's neurons sit; later ones build only what is new, about a minute. It reports the facts that got no neuron, "
                       "which answer from the table only.",
                       _schema(dict(path=dict(STR, description="The table's folder"), grafted_dir=dict(STR, description="The folder for the grafted model"),
                                    history_dir=dict(STR, description="The folder of the weights' history"),
                                    message=dict(STR, description="What changed, in a few words")), ["path", "grafted_dir", "history_dir"])),
    "revert_grafted": ("Go back in both forms: the table to a recorded version, and the grafted model's folder to the weights compiled from "
                       "it. If that table version was never exported, it is exported now. The step is recorded as a new version of the table and "
                       "of the weights.",
                       _schema(dict(path=STR, version=dict(STR, description="A version id of the table, such as v1"), grafted_dir=STR, history_dir=STR,
                                    discard=dict(BOOL, description="Also drop changes that no version holds, in the table's folder or the grafted model's")),
                               ["path", "version", "grafted_dir", "history_dir"])),
    "weights_versions": ("The versions recorded for a model's own weights: id, time, how many tensors differ from the base model, how many "
                         "facts, the table version each was compiled from, and the message; and whether the history is whole.",
                         _schema(dict(history_dir=STR), ["history_dir"])),
    "weights_diff": ("What differs between two recorded versions of the weights: the tensors (values, rows, columns, the largest difference) "
                     "and the facts added, removed and changed. A version may be 'base', the unedited model. Without b, to the newest.",
                     _schema(dict(history_dir=STR, a=dict(STR, description="A version id, such as v1, or base"), b=STR), ["history_dir", "a"])),
    "weights_ask": ("Ask questions of a recorded version of the weights, on the loaded model: its weights are set to the version, with the "
                    "version's switch, the questions are asked, and the model is put back as it was. A version may be 'base'. It shows what "
                    "the grafted model answers, without the table.",
                    _schema(dict(history_dir=STR, version=dict(STR, description="A version id, such as v1, or base"), questions=LIST),
                            ["history_dir", "version", "questions"])),
    "merge_grafted": ("Bring another saved table's facts into this table and into the model's own weights, in one call: the tables are merged, "
                      "the merged table is recorded, written into a grafted model and its weights recorded. Both tables must be built the "
                      "same way. To merge two grafted models without their tables, use weights_merge.",
                      _schema(dict(other=dict(STR, description="The folder of the other saved table"), path=STR, grafted_dir=STR, history_dir=STR,
                                   message=STR, prefer=dict(STR, description="Where both hold a fact with another answer: error (default), mine or theirs")),
                              ["other", "path", "grafted_dir", "history_dir"])),
    "weights_merge": ("Merge two grafted models into one, at the weights: each fact's neurons are read out of the two model folders and "
                      "written into one model, and the neurons that are left are re-fitted once. No table is needed, and the loaded table "
                      "is not changed. With history_dir the merged model is recorded as a version of the weights. The result says whether "
                      "the weights are the ones a single export of all the facts gives. A minute or two.",
                      _schema(dict(mine=dict(STR, description="The first grafted model's folder"),
                                   theirs=dict(STR, description="The second grafted model's folder"),
                                   out_dir=dict(STR, description="The folder for the merged grafted model"),
                                   history_dir=dict(STR, description="Leave out to write the folder only"),
                                   message=STR, prefer=dict(STR, description="Where both hold a fact with another answer: error (default), mine or theirs")),
                              ["mine", "theirs", "out_dir"])),
    "weights_checkout": ("Write a recorded version of the weights as a model folder, byte for byte as it was exported. A version may be "
                         "'base', the unedited model. The loaded model is not changed.",
                         _schema(dict(history_dir=STR, version=dict(STR, description="A version id, such as v1, or base"),
                                      out_dir=dict(STR, description="The folder to write; one that holds another version is written over")),
                                 ["history_dir", "version", "out_dir"])),
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
                return dict(error=str(error.args[0]) if isinstance(error, KeyError) and error.args else str(error))   # a KeyError's own text is quoted
            out["seconds"] = round(time.time() - t0, 1)
            return out

    # ------------------------------------------------------------ the tools

    def _status(self) -> dict:
        g = self.graft
        return dict(facts=len(g), layer=g.layer, attached=g.is_attached, relations=list(g.wordings.relations), options=vars(g.options),
                    relation_kinds={r: rel.subject for r, rel in g.wordings.relations.items()}, kinds=["company"] + sorted(g.wordings.kinds))

    def _list_facts(self) -> dict:
        return dict(facts=[vars(f) for f in self.graft.facts])

    def _add_fact(self, subject: str, relation: str, answer: str) -> dict:
        g = self.graft
        rel = canonical_relation(relation)
        replaced = any(f.subject.lower() == subject.strip().lower() and f.relation == rel for f in g.facts)   # an update gets a new id
        stored = g.add(subject, rel, answer)
        probe = g.wordings.relations[stored.relation].chat[0].format(s=stored.subject)
        forms = g.detectors_of(stored.subject, stored.relation)   # the forms of question the fact got a detector for
        out = dict(added=vars(stored), replaced=replaced, detectors=forms,
                   quick_look=dict(note="a training wording, for a quick look only; use check_fact on new wordings",
                                   question=probe, reply=g.ask(probe)))
        if len(forms) < 2:
            out["note"] = ("no detector could be built for " + ("plain or chat questions" if not forms else
                                                               f"{'chat' if forms == ['plain'] else 'plain'} questions")
                           + ": the model does not tell this fact's questions from its other ones, so they will not be answered. "
                             "More wordings for the relation (teach_wordings, then rebuild) may help")
        return out

    def _remove_fact(self, subject: str, relation: str | None = None) -> dict:
        return dict(removed=self.graft.remove(subject, relation), facts=len(self.graft))

    def _ask(self, question: str, chat: bool = True) -> dict:
        g = self.graft
        if len(g):
            out = g.compare(question, chat=chat)
        else:  # nothing is stored yet: the model's own reply, which is what looking before adding asks for
            from .gates import generate_batch
            from .model_access import Prompt, render_chat

            prompt = render_chat(g.tokenizer, question) if chat else Prompt(question)
            out = dict(question=question, without=generate_batch(g.model, g.tokenizer, [prompt], 48, batch_size=1)[0], with_graft=None)
        out["subject_found"] = g.find(question) if len(g) else None
        out["company_found"] = out["subject_found"]  # the earlier name of the same value
        return out

    @staticmethod
    def _said(text: str, phrase: str) -> bool:
        """Whether a reply holds a phrase as whole words, case ignored: "MIT" is not in "limit"."""
        return re.search(r"(?<![0-9a-z])" + re.escape(phrase.lower()) + r"(?![0-9a-z])", text.lower()) is not None

    def _check_fact(self, subject: str, relation: str, new_questions: list[str], other_questions: list[str] | None = None,
                    lookalike_subjects: list[str] | None = None, lookalike_companies: list[str] | None = None) -> dict:
        g = self.graft
        rel = canonical_relation(relation)
        fact = next((f for f in g.facts if f.subject.lower() == subject.lower() and f.relation == rel), None)
        if fact is None:
            raise KeyError(f"no stored fact for {subject!r} and {rel!r}")
        answer, first = fact.answer, fact.answer.split()[0]

        def run(questions):
            out = []
            for q in questions:
                both = g.compare(q, chat=q.rstrip().endswith(("?", ".")))
                out.append(dict(question=q, reply=both["with_graft"], without=both["without"]))
            return out

        def leaked(r) -> bool:  # the table brought the answer in: it is in the reply with the table and not in the one without
            return self._said(r["reply"], first) and not self._said(r["without"], first)

        def others_brought_in(r) -> list[str]:
            """Other stored facts whose answer the table brought into a reply. Not a leak of this fact, and not always wrong: another
            question about the same subject may ask for another stored fact. Listed for the editor to judge."""
            return [f"{f.subject} | {f.relation}" for f in g.facts if f is not fact and (f.subject, f.relation) != (fact.subject, fact.relation)
                    and self._said(r["reply"], f.answer.split()[0]) and not self._said(r["without"], f.answer.split()[0])]

        recall = run(new_questions)
        for r in recall:
            r["recalled"] = self._said(r["reply"], answer)
            r["said_without_the_table"] = self._said(r["without"], answer)
        others = run(other_questions or [])
        for r in others:
            r["leaked"], r["other_stored_answers"] = leaked(r), others_brought_in(r)
        looks = []
        for name in (lookalike_subjects or []) + (lookalike_companies or []):
            asked = run([g.wordings.relations[rel].chat[0].format(s=name)])[0]
            looks.append(dict(subject=name, **asked, leaked=leaked(asked), other_stored_answers=others_brought_in(asked)))
        n_rec = sum(r["recalled"] for r in recall)
        return dict(fact=vars(fact), relation_name=RELATION_NAMES.get(rel, rel), recall=f"{n_rec} of {len(recall)}",
                    leaks_other_questions=f"{sum(r['leaked'] for r in others)} of {len(others)}",
                    leaks_lookalikes=f"{sum(r['leaked'] for r in looks)} of {len(looks)}",
                    good=n_rec >= 0.75 * len(recall) and not any(r["leaked"] for r in others + looks), recall_replies=recall, other_replies=others,
                    lookalike_replies=looks)

    def _add_kind(self, name: str, names: list[str], generic_plain: list[str], generic_chat: list[str]) -> dict:
        kinds = self.graft.add_kind(name, names, generic_plain, generic_chat)
        return dict(added_kind=name, names=len(names), generic_plain=len(generic_plain), generic_chat=len(generic_chat), kinds=["company"] + kinds)

    def _add_relation(self, name: str, plain: list[str], chat: list[str], person: bool = True, subject: str = "company") -> dict:
        if subject != "company" and subject not in self.graft.wordings.kinds:
            raise ValueError(f"unknown subject kind {subject!r}; known: company, {', '.join(sorted(self.graft.wordings.kinds)) or 'none'}. "
                             "A new kind is added with add_kind")
        rel = Relation(name=canonical_relation(name), plain=plain, chat=chat, person=person, subject=subject)
        self.graft.add_relation(rel)
        same_kind = sum(self.graft.wordings.relations[f.relation].subject == subject for f in self.graft.facts)
        out = dict(added_relation=rel.name, plain=len(rel.plain), chat=len(rel.chat))
        if same_kind:  # a fact learns to stay shut on the other relations' questions when it is built: those stored before never saw this one
            out["note"] = (f"{same_kind} stored fact{'' if same_kind == 1 else 's'} about a {subject} "
                           f"{'was' if same_kind == 1 else 'were'} built before this relation existed: call rebuild so that "
                           f"{'it learns' if same_kind == 1 else 'they learn'} to stay shut on its questions")
        return out

    def _teach_wordings(self, relation: str, plain: list[str] | None = None, chat: list[str] | None = None) -> dict:
        out = self.graft.add_wordings(relation, plain or [], chat or [])
        left_out = len(plain or []) + len(chat or []) - out["added_plain"] - out["added_chat"]
        return dict(out, left_out=left_out, note="call rebuild so that the stored facts learn them"
                    + ("; a wording the relation already has, case and punctuation aside, is left out" if left_out else ""))

    def _rebuild(self, relation: str | None = None) -> dict:
        return dict(rebuilt=self.graft.rebuild(relation) if relation else self.graft.rebuild())

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

    def _commit(self, path: str, message: str = "") -> dict:
        from .graft import history

        before = len(history.versions(path)) if os.path.isdir(path) else 0
        version = self.graft.commit(path, message)
        return dict(version=version.id, recorded=len(history.versions(path)) > before, facts=version.facts, message=version.message)

    def _versions(self, path: str) -> dict:
        from .graft import history

        return dict(versions=[dict(id=v.id, time=v.time, facts=v.facts, message=v.message, of=v.of) for v in history.versions(path)],
                    intact=history.is_intact(path))

    def _diff(self, path: str, a: str, b: str | None = None) -> dict:
        from .graft import history

        return history.diff(path, a, b)

    def _revert(self, path: str, version: str, discard: bool = False) -> dict:
        g = self.graft
        new = g.revert(path, version, discard=discard)
        return dict(version=new.id, of=new.of, facts=len(g), attached=g.is_attached)

    @staticmethod
    def _weights_row(v) -> dict:
        return dict(id=v.id, time=v.time, tensors=len(v.tensors), facts=v.facts, compiled_from_table=v.table["version"] if v.table else None,
                    of=v.of, message=v.message)

    def _commit_grafted(self, path: str, grafted_dir: str, history_dir: str, message: str = "") -> dict:
        out = self.graft.commit_grafted(path, grafted_dir, history_dir, message)
        return dict(table_version=out["table"].id, weights=self._weights_row(out["weights"]), facts=len(self.graft),
                    not_in_weights=out["not_in_weights"])

    def _revert_grafted(self, path: str, version: str, grafted_dir: str, history_dir: str, discard: bool = False) -> dict:
        g = self.graft
        out = g.revert_grafted(path, version, grafted_dir, history_dir, discard=discard)
        return dict(table_version=out["table"].id, weights=self._weights_row(out["weights"]), exported=out["exported"], facts=len(g),
                    not_in_weights=out.get("not_in_weights", []), note="recorded as a new version of the table and of the weights")

    def _weights_versions(self, history_dir: str) -> dict:
        from . import weight_history

        return dict(versions=[self._weights_row(v) for v in weight_history.versions(history_dir)], intact=weight_history.is_intact(history_dir))

    def _weights_diff(self, history_dir: str, a: str, b: str | None = None) -> dict:
        from . import weight_history

        return weight_history.diff(history_dir, a, b)

    def _weights_ask(self, history_dir: str, version: str, questions: list[str]) -> dict:
        from . import weight_history
        from .gates import generate_batch
        from .model_access import render_chat

        g = self.graft
        with g.detached(), weight_history.applied(history_dir, version, g.model, tokenizer=g.tokenizer):
            replies = generate_batch(g.model, g.tokenizer, [render_chat(g.tokenizer, q) for q in questions], 48, batch_size=1)
        return dict(version=version, replies=[dict(question=q, reply=r) for q, r in zip(questions, replies)], model="put back as it was")

    def _merge_grafted(self, other: str, path: str, grafted_dir: str, history_dir: str, message: str = "", prefer: str = "error") -> dict:
        g = self.graft
        theirs = Graft.load(other, g.model, g._given_tokenizer)
        out = g.merge_grafted(theirs, path, grafted_dir, history_dir, message, prefer=prefer)
        return dict(merged=out["merged"], table_version=out["table"].id, weights=self._weights_row(out["weights"]), facts=len(g),
                    not_in_weights=out["not_in_weights"])

    def _weights_merge(self, mine: str, theirs: str, out_dir: str, history_dir: str | None = None, message: str = "", prefer: str = "error") -> dict:
        import warnings

        from . import weight_history
        from .graft import merge_grafted_models

        g = self.graft
        base = g._base_folder(None)
        how = dict(prefer=prefer, paragraphs=g._calibration)   # the ordinary text this table's own exports are re-fitted on
        with warnings.catch_warnings(record=True) as caught, g.detached():
            warnings.simplefilter("always")
            if history_dir:
                done = weight_history.merge(history_dir, mine, theirs, out_dir, g.model, g._given_tokenizer, message, base_dir=base, **how)
                merged, version = done["merged"], done["version"]
            else:
                merged, version = merge_grafted_models(g.model, g._given_tokenizer, base, mine, theirs, out_dir, **how), None
        return dict({k: merged[k] for k in ("added", "replaced", "kept", "same", "skipped_names", "facts", "same_as_one_export", "why")},
                    neurons={str(layer): n for layer, n in merged["neurons"].items()}, last_layer=merged["last_layer"], folder=out_dir,
                    weights=self._weights_row(version) if version is not None else None, warnings=[str(w.message) for w in caught])

    def _weights_checkout(self, history_dir: str, version: str, out_dir: str) -> dict:
        from . import weight_history

        done = weight_history.checkout(history_dir, version, out_dir, base_dir=self.graft._base_folder(None))
        return dict(version=done["version"], tensors=len(done["tensors"]), files=done["files"], folder=done["folder"])

"""`Graft`: a table of facts held in one inserted layer of a frozen model. The user-facing side of GRAFT.

    from cramschool import Graft

    graft = Graft(model, tokenizer)                      # layer 20 for Qwen3.5-0.8B; calibrates on first use
    graft.add("Nimbresco Software", "founder", "Tharindu Ekanayake")
    with graft.attached():
        print(graft.ask("Who founded Nimbresco Software?"))
    graft.save("my_facts.graft")                         # Graft.load("my_facts.graft", model, tokenizer) later

Adding a fact takes about 40 seconds on an RTX 3090: every part is measured with forward passes of the unedited model. Nothing is trained
and no weight changes; detaching the layer restores the model exactly. See docs/graft.md for the method.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import torch

from ..gates import generate_batch
from ..model_access import Prompt, model_device, render_chat, tensor_fingerprint
from .build import N_SLOT_ANSWERS, build_context, build_looser, build_parts, build_shared, build_slots, entry_of, make_fact
from .core import calibration_stats, constant_direction, mlp_of
from .layer import GraftLayer, NameIndex
from .wordings import Relation, Wordings, calibration_paragraphs, canonical_relation

FORMAT_VERSION = 1
VALIDATED = {"Qwen3.5-0.8B": 20}  # models GRAFT was validated on, and the layer used


@dataclass
class GraftOptions:
    """push: multiple of each answer vector; margin: how far a shared detector must lead to veto; looser_start: also open at the prompt's
    last token through fact block v2's detector (raises recall, adds leaks; off by default).

    real_world: "all" (v1) or "other_relations", which leaves out the fact's own relation from the real-world negatives: use it when the
    subjects are real, as when changing a fact the model already knows. context: how many sentences of ordinary text to also teach plain
    prompts behind (v1: 0; 4 in the CounterFact runs). fast: keep readings that every fact of a relation shares, so each is computed once
    (about 4 times faster after the first fact of a relation; off by default, because shared readings can differ in their last bits from
    fresh ones, and v1 reproduces exactly without them). All three change only how new facts are built.

    inside: "negative" (v1) or "scatter": whether the readings inside a fact's taught answers are a group of negatives for its detectors,
    or only part of their scatter. "scatter" helps answers spelled with several tokens ("Serbian"), whose detectors otherwise give up
    weight on the questions they must tell apart. end_quantile: at the prompt's last token only, a fact may also open on its own detector
    at this quantile of its prompt negatives (0.99, 0.95 or 0.90; None in v1). twins: pairs of relations that ask in nearly the same words
    (scripts/diag_twins.py finds them); a fact's detector doesn't take its twin as negatives (None in v1). docs/graft.md, section 11.

    arm: after a fact opens at the prompt's last token and its first answer token is not the one produced, the fact stays armed for this
    many positions, and its first token is pushed again where the reply reaches the place of the answer (0 in v1: one push only). It
    changes the layer, not how facts are built, so it can be set on a loaded table; the detectors of that place are built on first use."""
    push: float = 2.0
    margin: float = 0.5
    looser_start: bool = False
    real_world: str = "all"
    context: int = 0
    fast: bool = False
    inside: str = "negative"
    end_quantile: float | None = None
    twins: list | None = None
    arm: int = 0


@dataclass
class StoredFact:
    id: str
    subject: str
    relation: str
    answer: str
    seconds: float = 0.0


def default_layer(model) -> int:
    """Layer 20 of 24 for Qwen3.5-0.8B, where GRAFT was validated; the same depth (about 5/6) for other models, with a warning."""
    n = model.config.get_text_config().num_hidden_layers
    if n == 24:
        return 20
    layer = round(n * 20 / 24)
    warnings.warn(f"GRAFT was validated only on Qwen3.5-0.8B (layer 20 of 24). Using layer {layer} of {n}; check it on your model "
                  "(cramschool bench) before relying on it.", stacklevel=3)
    return layer


class Graft:
    """A table of facts for one model, held in one inserted layer."""

    def __init__(self, model, tokenizer, *, layer: int | None = None, wordings: Wordings | None = None, options: GraftOptions | None = None,
                 calibration: list[str] | None = None, progress: Callable[[str], None] | None = None):
        self.model, self.tokenizer = model, tokenizer
        self.layer = layer if layer is not None else default_layer(model)
        self.wordings = wordings or Wordings.builtin()
        self.options = options or GraftOptions()
        self._calibration = calibration
        self._progress = progress
        self._mctx: dict | None = None
        self._facts: list[dict] = []          # each: dict(fact=bench-format dict, parts=..., looser=... or None)
        self._aliases: dict[str, list[str]] = {}  # a stored subject's declared other names
        self._slots: dict | None = None           # the detectors for the place of the answer inside a reply, by relation
        self._slot_answers: dict | None = None
        self._shared: dict | None = None
        self._shared_relations: tuple = ()
        self._layer: GraftLayer | None = None
        self._next = 1

    # ------------------------------------------------------------ calibration

    def calibration(self) -> dict:
        """c, c̄ and the covariance of the layer's MLP input on ordinary text (computed once, about 40 seconds)."""
        if self._mctx is None:
            self._say(f"calibrating layer {self.layer} on ordinary text")
            stats = calibration_stats(self.model, self.tokenizer, self._calibration or calibration_paragraphs(), self.layer)
            c, cbar, spread = constant_direction(stats)
            self._mctx = dict(c=c, cbar=cbar, cov=stats["cov"], spread=spread)
        return self._mctx

    # ------------------------------------------------------------ facts

    @property
    def facts(self) -> list[StoredFact]:
        return [StoredFact(e["fact"]["id"], e["fact"]["subject"], e["fact"]["relation"], e["fact"]["target"], e["parts"].get("seconds", 0.0))
                for e in self._facts]

    def __len__(self) -> int:
        return len(self._facts)

    def _read_cache(self):
        """The table's ReadCache when options.fast is on, kept across additions; None otherwise."""
        if not self.options.fast:
            return None
        if getattr(self, "_cache", None) is None:
            from .core import ReadCache
            self._cache = ReadCache()
        return self._cache

    def add_relation(self, relation: Relation) -> None:
        """Teach the table a new relation's wordings (at least 8 plain and 4 chat; 30 and 30 were used for the built-in ones)."""
        self.wordings = self.wordings.with_relations({relation.name: relation})
        self._shared = None

    def add(self, subject: str, relation: str, answer: str) -> StoredFact:
        """Add a fact, or update it if the table already has this relation for this company. Returns the stored fact."""
        relation = canonical_relation(relation)
        if relation not in self.wordings.relations:
            raise KeyError(f"unknown relation {relation!r}. Known: {', '.join(self.wordings.relations)}. Add its wordings with add_relation().")
        self.remove(subject, relation)
        fid = f"f{self._next:03d}"
        self._next += 1
        f = make_fact(subject.strip(), relation, answer.strip(), self.wordings, fid=fid)
        ctx = build_context([f], self.wordings, name="table", guard=False, real_world=self.options.real_world, context=self.options.context,
                            cache=self._read_cache(), inside=self.options.inside, twins=self.options.twins)
        mctx = self.calibration()
        self._say(f"building {subject} | {relation} | {answer}")
        parts = build_parts(self.model, self.tokenizer, f, ctx, mctx, self.layer)
        looser = build_looser(self.model, self.tokenizer, f, ctx, mctx, self.layer, parts) if self.options.looser_start else None
        self._facts.append(dict(fact=f, parts=parts, looser=looser))
        self._refresh()
        stored = self.facts[-1]
        self._say(f"added {stored.subject} | {stored.relation} | {stored.answer} in {parts['seconds']:.0f}s")
        return stored

    def add_many(self, records: Iterable) -> list[StoredFact]:
        """Add several facts: (subject, relation, answer) tuples, dicts with those keys, or `datasets.FactRecord`s."""
        out = []
        for r in records:
            if isinstance(r, dict):
                out.append(self.add(r["subject"], r["relation"], r.get("answer", r.get("target"))))
            elif hasattr(r, "subject"):
                out.append(self.add(r.subject, r.relation, r.answer))
            else:
                out.append(self.add(*r))
        return out

    def remove(self, subject: str, relation: str | None = None) -> int:
        """Remove a company's fact for one relation, or all its facts. Returns how many were removed."""
        key = " ".join(subject.lower().split())
        rel = canonical_relation(relation) if relation else None
        keep = [e for e in self._facts if not (" ".join(e["fact"]["subject"].lower().split()) == key and (rel is None or e["fact"]["relation"] == rel))]
        removed = len(self._facts) - len(keep)
        if removed:
            self._facts = keep
            left = {e["fact"]["subject"] for e in keep}
            self._aliases = {s: a for s, a in self._aliases.items() if s in left}
            self._refresh()
        return removed

    def add_alias(self, subject: str, *aliases: str) -> list[str]:
        """Declare other names for a stored subject ("Vorantis" for "Vorantis Biotech"). Key 1 then matches them as it matches the name:
        as a whole word, in any capitalisation, wherever they appear. So an alias that is part of another company's name opens the layer
        there too. Nothing is rebuilt. Returns the subject's aliases."""
        norm = lambda s: " ".join(s.lower().split())  # noqa: E731
        stored = {norm(e["fact"]["subject"]): e["fact"]["subject"] for e in self._facts}
        if norm(subject) not in stored:
            raise ValueError(f"no fact is stored about {subject!r}")
        name = stored[norm(subject)]
        taken = {**stored, **{norm(a): s for s, names in self._aliases.items() for a in names}}
        for alias in aliases:
            if not norm(alias):
                raise ValueError("an alias can't be empty")
            if taken.get(norm(alias), name) != name:
                raise ValueError(f"{alias!r} already names {taken[norm(alias)]!r}")
            if norm(alias) not in taken:
                self._aliases.setdefault(name, []).append(" ".join(alias.split()))
                taken[norm(alias)] = name
        self._refresh()
        return list(self._aliases.get(name, []))

    @property
    def aliases(self) -> dict[str, list[str]]:
        """Each stored subject's declared other names."""
        return {s: list(a) for s, a in self._aliases.items()}

    def _alias_map(self) -> dict[str, str]:
        return {alias: subject for subject, names in self._aliases.items() for alias in names}

    def find(self, text: str) -> str | None:
        """The stored company a text names last, by its name or a declared alias, as the layer would see it."""
        return NameIndex(self.tokenizer, sorted({e["fact"]["subject"] for e in self._facts}), self._alias_map()).find(self.tokenizer, text)

    # ------------------------------------------------------------ the layer

    def shared_detectors(self) -> dict:
        rels = tuple(self.wordings.relations)
        if self._shared is None or self._shared_relations != rels:
            self._say("building the shared relation detectors")
            ctx = build_context([e["fact"] for e in self._facts], self.wordings, name="table", guard=False, real_world=self.options.real_world,
                                context=self.options.context, cache=self._read_cache(), twins=self.options.twins)
            self._shared = build_shared(self.model, self.tokenizer, ctx, self.calibration(), self.layer)
            self._shared_relations = rels
        return self._shared

    def slot_detectors(self) -> dict:
        """One detector per stored relation for the place of the answer inside a reply (build.build_slots), for the armed push. Built
        once, and again when a relation's first stored answers change."""
        answers: dict[str, list[str]] = {}
        for e in self._facts:
            answers.setdefault(e["fact"]["relation"], []).append(e["fact"]["target"])
        answers = {r: list(dict.fromkeys(a))[:N_SLOT_ANSWERS] for r, a in answers.items()}
        if self._slots is None or self._slot_answers != answers:
            self._say("building the detectors for the place of the answer inside a reply")
            ctx = build_context([e["fact"] for e in self._facts], self.wordings, name="table", guard=False, real_world=self.options.real_world,
                                twins=self.options.twins)
            self._slots, self._slot_answers = build_slots(self.model, self.tokenizer, ctx, self.calibration(), self.layer, answers), answers
        return self._slots

    def layer_object(self) -> GraftLayer:
        mctx = self.calibration()
        entries = [entry_of(e["parts"], self.options.push, e["looser"] if self.options.looser_start else None, end_quantile=self.options.end_quantile)
                   for e in self._facts]
        kinds = {r: rel.subject for r, rel in self.wordings.relations.items()}
        return GraftLayer(self.model, self.tokenizer, entries, self.shared_detectors(), mctx["c"], mctx["cbar"], self.options.margin, self.layer,
                          relation_kinds=kinds if any(k != "company" for k in kinds.values()) else None, arm=self.options.arm,
                          aliases=self._alias_map(), slots=self.slot_detectors() if self.options.arm else None)

    def attach(self) -> "Graft":
        if not self._facts:
            raise ValueError("the table is empty: add a fact first")
        if self._layer is None:
            self._layer = self.layer_object().attach()
        return self

    def detach(self) -> None:
        if self._layer is not None:
            self._layer.detach()
            self._layer = None

    @property
    def is_attached(self) -> bool:
        return self._layer is not None

    @contextlib.contextmanager
    def attached(self):
        was = self.is_attached
        self.attach()
        try:
            yield self
        finally:
            if not was:
                self.detach()

    @contextlib.contextmanager
    def detached(self):
        was = self.is_attached
        self.detach()
        try:
            yield self
        finally:
            if was:
                self.attach()

    def _refresh(self) -> None:
        if self._layer is not None:
            self.detach()
            if self._facts:
                self.attach()

    @property
    def events(self) -> dict:
        """How many answers the layer started and continued, and how many a shared detector vetoed, since it was attached."""
        return dict(self._layer.events) if self._layer is not None else {}

    # ------------------------------------------------------------ asking

    def ask(self, question: str, *, chat: bool = True, max_new_tokens: int = 48) -> str:
        """Generate a greedy reply with the layer attached. `chat=False` treats the question as a plain text opening."""
        prompt = render_chat(self.tokenizer, question) if chat else Prompt(question)
        with self.attached():
            return generate_batch(self.model, self.tokenizer, [prompt], max_new_tokens, batch_size=1)[0]

    def compare(self, question: str, *, chat: bool = True, max_new_tokens: int = 48) -> dict:
        """The model's reply without the layer and with it."""
        prompt = render_chat(self.tokenizer, question) if chat else Prompt(question)
        with self.detached():
            before = generate_batch(self.model, self.tokenizer, [prompt], max_new_tokens, batch_size=1)[0]
        return dict(question=question, without=before, with_graft=self.ask(question, chat=chat, max_new_tokens=max_new_tokens))

    # ------------------------------------------------------------ the grafted model

    def continuations(self) -> list[list[dict]]:
        """Each fact's continuation detectors (build.build_continuation), built once and kept with the fact: what a grafted model needs
        to finish answers of several tokens. Forward passes over each fact's teaching examples, a few seconds per fact."""
        from .build import build_continuation, ordinary_reads
        from .core import CALIBRATION_FIT

        todo = [e for e in self._facts if "continuation" not in e]
        if todo:
            mctx = self.calibration()
            paras = self._calibration or calibration_paragraphs()
            ordinary = ordinary_reads(self.model, self.tokenizer, [paras[i] for i in CALIBRATION_FIT if i < len(paras)], self.layer)
            for n, e in enumerate(todo, 1):
                ctx = build_context([e["fact"]], self.wordings, name="table", guard=False, real_world=self.options.real_world,
                                    context=self.options.context, inside=self.options.inside, twins=self.options.twins)
                e["continuation"] = build_continuation(self.model, self.tokenizer, e["fact"], ctx, mctx, self.layer, ordinary)
                self._say(f"continuation detectors {n}/{len(todo)}: {e['fact']['subject']} | {e['fact']['relation']} | "
                          f"{sum(k['delta'] is not None for k in e['continuation'])} of {len(e['continuation'])} later tokens")
        return [e["continuation"] for e in self._facts]

    def grafted_facts(self, detected: bool = False) -> list[dict]:
        """The table's facts as grafted.compile_layer takes them; with `detected`, each with its continuation detectors."""
        cont = self.continuations() if detected else [None] * len(self._facts)
        return [dict(subject=e["fact"]["subject"], relation=e["fact"]["relation"], answer=e["fact"]["target"], parts=e["parts"],
                     **({"continuation": k} if k is not None else {})) for e, k in zip(self._facts, cont)]

    def export_grafted(self, out_dir: str, *, base_dir: str | None = None, forms=("plain", "chat"), texts=None, complete: bool | str = True) -> dict:
        """Compile the table's facts into the model's own neurons at the table's layer and write the grafted model to `out_dir`: the base
        checkpoint with that layer's three MLP matrices replaced (same files, shapes and size), grafted.json and grafted_switch.py. Load it
        with cramschool.graft.load_grafted, or with transformers and the folder's grafted_switch.attach. base_dir: the local folder the
        model was loaded from (default: the model's own path).

        complete (lab notebook, step 7), True by default. The first token of each form is a saturating pair of neurons, which writes the
        inserted layer's push once the detector is open and never more. Each later answer token has a neuron of its own, which the name
        switch turns on as it follows the answer's tokens, the inserted layer's own bookkeeping. The facts act at the last position of a
        call only, as the inserted layer does.
        - "detected": each later token's neuron has a detector for "the answer's previous token has just been read" as its gate, and the
          switch knows names only. On bench40e that left many answers unfinished.
        - False: the first shipped compile. One neuron per form, whose push grows with the detector's score, writes the first token only,
          at every token after the name. It matches the inserted layer on answers of one token and fails on longer ones.

        forms: plain and chat by default; ("plain",) leaves the chat-form neurons out, for a table that is only asked in plain text.
        texts: ordinary text (strings, about 1.1M tokens in all) for choosing and re-fitting the freed neurons (grafted.text_moments). It
        lowers the drift; the check with 237 facts was compiled with it.

        Neurons: with complete, two per form and one per later answer token. The damage bar (bench.scoring.FirmCheck) was met up to 512
        freed neurons at layer 20 of the 0.8B; at 1,024 a firmly known fact was lost.

        The loaded model is left unchanged. Returns the export's report."""
        if complete not in (False, True, "detected"):
            raise ValueError(f"complete must be False, True or 'detected', not {complete!r}")
        from .grafted import compile_layer, export_grafted, text_moments

        base = base_dir or getattr(self.model.config, "_name_or_path", "")
        if not base or not os.path.isdir(base):
            raise ValueError("give base_dir: the local folder, in safetensors format, that the model was loaded from")
        if not self._facts:
            raise ValueError("the table has no facts to compile")
        if self.options.looser_start or self.options.end_quantile is not None or self.options.arm:
            warnings.warn("export_grafted compiles each fact's strict detectors only: looser_start, end_quantile and arm have no neuron form "
                          "and are left out of the grafted model", stacklevel=2)
        facts = self.grafted_facts(detected=complete == "detected")
        extra = text_moments(self.model, self.tokenizer, self.layer, texts) if texts is not None else None
        compiled = compile_layer(self.model, self.tokenizer, facts, self.layer, self.calibration(), forms=forms, push=self.options.push,
                                 paragraphs=self._calibration, extra_moments=extra, saturate=bool(complete), where="last" if complete else "after",
                                 track=complete is True, progress=self._say)
        return export_grafted(base, out_dir, compiled, meta=dict(base_model=base, facts=len(facts), aliases=self._alias_map()))

    # ------------------------------------------------------------ saving

    def _fingerprint(self) -> str:
        return tensor_fingerprint(mlp_of(self.model, self.layer).down_proj.weight)[:16]

    def save(self, path: str) -> str:
        """A folder: graft.json (the facts and settings), calibration.pt, shared.pt, and one parts file per fact."""
        os.makedirs(os.path.join(path, "parts"), exist_ok=True)
        mctx = self.calibration()
        torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in mctx.items()}, os.path.join(path, "calibration.pt"))
        torch.save(self.shared_detectors(), os.path.join(path, "shared.pt"))
        if self._slots is not None:
            torch.save(dict(slots=self._slots, answers=self._slot_answers), os.path.join(path, "slots.pt"))
        facts = []
        for e in self._facts:
            fid = e["fact"]["id"]
            torch.save(e["parts"], os.path.join(path, "parts", f"{fid}.pt"))
            if e["looser"] is not None:
                torch.save(e["looser"], os.path.join(path, "parts", f"{fid}.looser.pt"))
            facts.append(dict(id=fid, subject=e["fact"]["subject"], relation=e["fact"]["relation"], answer=e["fact"]["target"]))
        meta = dict(format=FORMAT_VERSION, method="GRAFT", saved=time.strftime("%Y-%m-%d %H:%M:%S"),
                    model=getattr(self.model.config, "_name_or_path", ""), layer=self.layer, fingerprint=self._fingerprint(),
                    options=asdict(self.options), relations={r: rel.as_dict() for r, rel in self.wordings.relations.items()},
                    kinds={k: v for k, v in self.wordings.kinds.items() if any(rel.subject == k for rel in self.wordings.relations.values())},
                    next=self._next, facts=facts, aliases=self._aliases)
        with open(os.path.join(path, "graft.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        return path

    @classmethod
    def load(cls, path: str, model, tokenizer, *, check_model: bool = True, progress: Callable[[str], None] | None = None) -> "Graft":
        """Load a saved table onto the same base model. Refuses a model whose layer weights differ, unless check_model=False."""
        with open(os.path.join(path, "graft.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported table format {meta.get('format')!r}")
        builtin = Wordings.builtin()
        wordings = Wordings(relations={r: Relation(name=r, plain=d["plain"], chat=d["chat"], person=d.get("person", True), subject=d.get("subject", "company"))
                                       for r, d in meta["relations"].items()},
                            generic=builtin.generic, kinds={**builtin.kinds, **meta.get("kinds", {})})
        graft = cls(model, tokenizer, layer=meta["layer"], wordings=wordings, options=GraftOptions(**meta["options"]), progress=progress)
        if check_model and graft._fingerprint() != meta["fingerprint"]:
            raise ValueError("this table was built on another model: its layer weights differ. Load it onto the model it was built for, "
                             "or pass check_model=False if you know what you are doing.")
        mctx = torch.load(os.path.join(path, "calibration.pt"), weights_only=True)
        device = model_device(model)
        graft._mctx = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in mctx.items()}
        graft._shared = torch.load(os.path.join(path, "shared.pt"), weights_only=True)
        graft._shared_relations = tuple(wordings.relations)
        if os.path.exists(os.path.join(path, "slots.pt")):
            saved = torch.load(os.path.join(path, "slots.pt"), weights_only=True)
            graft._slots, graft._slot_answers = saved["slots"], saved["answers"]
        for d in meta["facts"]:
            parts = torch.load(os.path.join(path, "parts", f"{d['id']}.pt"), weights_only=True)
            lp = os.path.join(path, "parts", f"{d['id']}.looser.pt")
            looser = torch.load(lp, weights_only=True) if os.path.exists(lp) else None
            f = make_fact(d["subject"], d["relation"], d["answer"], wordings, fid=d["id"])
            graft._facts.append(dict(fact=f, parts=parts, looser=looser))
        graft._next = meta.get("next", len(graft._facts) + 1)
        graft._aliases = {s: list(a) for s, a in meta.get("aliases", {}).items()}
        return graft

    def _say(self, text: str) -> None:
        if self._progress is not None:
            self._progress(text)

    def __repr__(self) -> str:
        return f"Graft({len(self)} facts, layer {self.layer}, {'attached' if self.is_attached else 'detached'})"

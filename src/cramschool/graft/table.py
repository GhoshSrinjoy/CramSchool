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
from .build import build_context, build_looser, build_parts, build_shared, entry_of, make_fact
from .core import calibration_stats, constant_direction, mlp_of
from .layer import GraftLayer, NameIndex
from .wordings import Relation, Wordings, calibration_paragraphs, canonical_relation

FORMAT_VERSION = 1
VALIDATED = {"Qwen3.5-0.8B": 20}  # models GRAFT was validated on, and the layer used


@dataclass
class GraftOptions:
    """push: multiple of each answer vector; margin: how far a shared detector must lead to veto; looser_start: also open at the prompt's
    last token through fact block v2's detector (raises recall, adds leaks; off by default)."""
    push: float = 2.0
    margin: float = 0.5
    looser_start: bool = False


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
        ctx = build_context([f], self.wordings, name="table", guard=False)
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
            self._refresh()
        return removed

    def find(self, text: str) -> str | None:
        """The stored company a text names last, as the layer would see it."""
        return NameIndex(self.tokenizer, sorted({e["fact"]["subject"] for e in self._facts})).find(self.tokenizer, text)

    # ------------------------------------------------------------ the layer

    def shared_detectors(self) -> dict:
        rels = tuple(self.wordings.relations)
        if self._shared is None or self._shared_relations != rels:
            self._say("building the shared relation detectors")
            ctx = build_context([e["fact"] for e in self._facts], self.wordings, name="table", guard=False)
            self._shared = build_shared(self.model, self.tokenizer, ctx, self.calibration(), self.layer)
            self._shared_relations = rels
        return self._shared

    def layer_object(self) -> GraftLayer:
        mctx = self.calibration()
        entries = [entry_of(e["parts"], self.options.push, e["looser"] if self.options.looser_start else None) for e in self._facts]
        return GraftLayer(self.model, self.tokenizer, entries, self.shared_detectors(), mctx["c"], mctx["cbar"], self.options.margin, self.layer)

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

    # ------------------------------------------------------------ saving

    def _fingerprint(self) -> str:
        return tensor_fingerprint(mlp_of(self.model, self.layer).down_proj.weight)[:16]

    def save(self, path: str) -> str:
        """A folder: graft.json (the facts and settings), calibration.pt, shared.pt, and one parts file per fact."""
        os.makedirs(os.path.join(path, "parts"), exist_ok=True)
        mctx = self.calibration()
        torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in mctx.items()}, os.path.join(path, "calibration.pt"))
        torch.save(self.shared_detectors(), os.path.join(path, "shared.pt"))
        facts = []
        for e in self._facts:
            fid = e["fact"]["id"]
            torch.save(e["parts"], os.path.join(path, "parts", f"{fid}.pt"))
            if e["looser"] is not None:
                torch.save(e["looser"], os.path.join(path, "parts", f"{fid}.looser.pt"))
            facts.append(dict(id=fid, subject=e["fact"]["subject"], relation=e["fact"]["relation"], answer=e["fact"]["target"]))
        meta = dict(format=FORMAT_VERSION, method="GRAFT", saved=time.strftime("%Y-%m-%d %H:%M:%S"),
                    model=getattr(self.model.config, "_name_or_path", ""), layer=self.layer, fingerprint=self._fingerprint(),
                    options=asdict(self.options), relations={r: dict(plain=rel.plain, chat=rel.chat, person=rel.person) for r, rel in self.wordings.relations.items()},
                    next=self._next, facts=facts)
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
        wordings = Wordings(relations={r: Relation(name=r, plain=d["plain"], chat=d["chat"], person=d.get("person", True)) for r, d in meta["relations"].items()},
                            generic=Wordings.builtin().generic)
        graft = cls(model, tokenizer, layer=meta["layer"], wordings=wordings, options=GraftOptions(**meta["options"]), progress=progress)
        if check_model and graft._fingerprint() != meta["fingerprint"]:
            raise ValueError("this table was built on another model: its layer weights differ. Load it onto the model it was built for, "
                             "or pass check_model=False if you know what you are doing.")
        mctx = torch.load(os.path.join(path, "calibration.pt"), weights_only=True)
        device = model_device(model)
        graft._mctx = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in mctx.items()}
        graft._shared = torch.load(os.path.join(path, "shared.pt"), weights_only=True)
        graft._shared_relations = tuple(wordings.relations)
        for d in meta["facts"]:
            parts = torch.load(os.path.join(path, "parts", f"{d['id']}.pt"), weights_only=True)
            lp = os.path.join(path, "parts", f"{d['id']}.looser.pt")
            looser = torch.load(lp, weights_only=True) if os.path.exists(lp) else None
            f = make_fact(d["subject"], d["relation"], d["answer"], wordings, fid=d["id"])
            graft._facts.append(dict(fact=f, parts=parts, looser=looser))
        graft._next = meta.get("next", len(graft._facts) + 1)
        return graft

    def _say(self, text: str) -> None:
        if self._progress is not None:
            self._progress(text)

    def __repr__(self) -> str:
        return f"Graft({len(self)} facts, layer {self.layer}, {'attached' if self.is_attached else 'detached'})"

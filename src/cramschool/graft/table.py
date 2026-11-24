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
import re
import time
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Callable, Iterable

import torch

from ..gates import generate_batch
from ..model_access import Prompt, model_device, render_chat, tensor_fingerprint, with_system
from .build import (N_SLOT_ANSWERS, build_context, build_looser, build_parts, build_prompt_detectors, build_shared, build_slots, entry_of, fact_detectors,
                    make_fact)
from .core import blocks, calibration_stats, constant_direction, mlp_of
from .core import final_scale as measure_final_scale
from .layer import GraftLayer, NameIndex
from .wordings import Relation, Wordings, calibration_paragraphs, canonical_relation

FORMAT_VERSION = 1
ORDINARY_TEXT = "ordinary_text.json"   # in a saved table's folder: the ordinary text it was given, if not the bundled paragraphs
WEIGHTS_PARTS = "weights_parts"   # in a saved table's folder: its facts as built at the last layer, for the grafted model
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
    changes the layer, not how facts are built, so it can be set on a loaded table; the detectors of that place are built on first use.

    teach_cap: teach each fact's answer vectors from at most this many of its relation's wordings per form, the first ones (None in v1:
    all). The detectors still learn from every wording. Use it when a relation has many wordings: the answer vectors read every teaching
    example's logits at once, and with more than about 30 per form that can need more memory than the GPU has.

    avoid_as: how a fact learns what its relation must not answer (add_avoid, experimental and not adopted). "group": the questions about
    its own company and a few invented ones are a group of negatives of their own, and the threshold is above every one of them.
    "negatives": they join the ordinary questions about the same company, read about every invented company, with no group of their own
    and no threshold above them.

    prompt_detector: at the prompt's last token only, a fact may also open through a second detector of its own, fitted against its prompt
    negatives alone (build.build_prompt_detectors; off by default). A fact's strict detectors are also fitted against the inside of its
    answer and the model's replies, which costs them question wordings they were not taught; inside a reply they still decide alone. On a
    blind bench of 40 facts, 31 passed with it against 23 without, with no fact lost and no leak (lab notebook, step 20). It is for tables
    with at most two facts on a company: with four on every company it added no fact, and 8 of 40 facts failed a leak check against 2
    (step 22). It opens more questions that are close to the relation and are not it, and a grafted model doesn't have it. Facts about
    other companies don't reach it (step 21). Set on a loaded table, the missing detectors are built on first use. It shares the layer's
    slot with looser_start and end_quantile: set one of the three.

    last_layer: where an answer's tokens are written (lab notebook, step 24). "all" (the default since 1.1): every token at the output
    of the model's last MLP, as the token's own row of the output matrix at twice the size of what the final norm reads on ordinary
    text; and an answer whose first token comes in its other spelling (" Ka" where "Ka" was written) is followed all the same.
    "later": only the tokens after the first are written there, and the first at the table's layer. None (v1, and what every published
    number but step 24's was measured with): every token at the table's layer, with the vectors measured for the fact. Nothing reads
    the last MLP's output but the final norm and the output matrix at the same position, so the write changes that token and nothing
    after it, and an answer that has begun is finished. It also finishes an answer that began on a question which isn't the fact's
    own. It changes the layer, not how facts are built, so it can be set on a loaded table: a table saved before 1.1 loads with None,
    as it was built. The size is measured with the calibration (in a loaded table that lacks it, on first use: about half a minute)
    and saved with the table. A grafted model doesn't have it.

    Measured with "all" (lab notebook, step 24): on bench40h, which did not choose the setting, 32 of 40 facts pass against 23 with
    None, and 40 against 31 together with prompt_detector; no fact is lost and none fails a leak check. Development: 37 and 33 of 40
    against 32 and 24. "later" alone gave one fact more over those two benches. None of these benches was blind when scored."""
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
    teach_cap: int | None = None
    avoid_as: str = "group"
    prompt_detector: bool = False
    last_layer: str | None = "all"


TAUGHT_CAP = 30  # the cap on wordings per form for the answer vectors, set when a relation is taught more than it came with


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

    def __init__(self, model, tokenizer, *, layer: int | str | None = None, wordings: Wordings | None = None, options: GraftOptions | None = None,
                 calibration: list[str] | None = None, progress: Callable[[str], None] | None = None, system: str | None = None):
        """layer: where the table reads the question. None (the default) and "auto": measured. Every layer of this model is profiled
        on invented probe facts and the pick is taken (layers.profile_layers, lab notebook, step 25): layer 22 on Qwen3.5-0.8B, where
        40 of 40 facts of a bench pass, against 38 at the last layer and 32 at layer 20. The profile takes about 3 minutes on the
        0.8B the first time; it is kept on this computer for the model (layers.profiles_dir), so the next table on the same model
        starts at once, and it is saved with the table as `layer_profile`. "last": the model's last layer. A number sets the layer.

        The layer is measured only for a table that writes its answers at the last layer (options.last_layer="all", the default):
        with a push at the table's own layer the layer is also where the answer is written, which the profile doesn't measure, and
        the earlier rule is used: layer 20 of a 24-layer model, the same depth of another, with a warning.

        The grafted model does not depend on this choice. Its neurons sit at the last layer and need detectors that read there, so
        `export_grafted` builds the facts a second time at the last layer (`weights_table`) and compiles those.

        system: the system message of the application the table will be asked in. Chat questions are then taught, and asked by `ask`
        and `compare`, behind that message; the table keeps its own copy of the tokenizer for it (model_access.with_system) and the one
        passed in is not changed. A table is for prompts in the chat format it was built in: taught with no system message, a table of 30
        facts answered 51 of 120 questions behind one, and 93 taught behind it (lab notebook, step 23). None (v1): no system message."""
        self.model, self.system = model, system or None
        self._given_tokenizer = tokenizer  # as passed in, for loading another version of the table onto the same model (revert)
        self.tokenizer = with_system(tokenizer, self.system) if self.system else tokenizer
        self.wordings = wordings or Wordings.builtin()
        self.options = options or GraftOptions()
        self._calibration = calibration
        self._progress = progress
        self.layer_profile: dict | None = None    # what layer="auto" measured, when the layer was picked that way
        self._moments: dict = {}                  # per layer, its second moments on this table's ordinary text: kept between exports
        self._later_cal: dict = {}                # per layer later tokens can move to, its constant direction and scale
        self._twin: Graft | None = None           # the same facts built at the last layer, for the grafted model (weights_table)
        self._twin_built: dict = {}               # per fact of the twin, the hash of this table's parts it was built beside
        self._twin_path: str | None = None        # where a saved twin waits to be loaded
        if isinstance(layer, str):
            if layer not in ("auto", "last"):
                raise ValueError(f"layer must be a number, None, 'last' or 'auto', not {layer!r}")
            self.layer = len(blocks(model)) - 1 if layer == "last" else self._pick_layer()
        elif layer is None:
            # measured where the profile applies; a table that pushes at its own layer keeps the layer pushes were measured at
            self.layer = self._pick_layer() if self.options.last_layer == "all" else default_layer(model)
        else:
            self.layer = layer
        self._mctx: dict | None = None
        self._facts: list[dict] = []          # each: dict(fact=bench-format dict, parts=..., looser=... or None, prompt=... or None)
        self._aliases: dict[str, list[str]] = {}  # a stored subject's declared other names
        self._exclusions: dict[str, list[str]] = {}  # longer names that contain a stored subject's name and name something else
        self._slots: dict | None = None           # the detectors for the place of the answer inside a reply, by relation
        self._slot_answers: dict | None = None
        self._shared: dict | None = None
        self._shared_relations: tuple = ()
        self._layer: GraftLayer | None = None
        self._next = 1

    def _pick_layer(self) -> int:
        """The layer `layer="auto"` gives: the profile's pick, or the earlier rule where the profile doesn't apply."""
        if self.options.last_layer != "all":
            warnings.warn("layer='auto' measures where the question is best read. This table also pushes its answer at its own layer "
                          f"(last_layer={self.options.last_layer!r}), which the profile doesn't measure, so the earlier rule is used.", stacklevel=3)
            return default_layer(self.model)
        from .layers import keep_profile, kept_profile, profile_key, profile_layers

        key = profile_key(self.model, self.tokenizer, self.wordings, self._calibration, self.options)
        kept = kept_profile(key)
        if kept is not None:
            self.layer_profile = kept
            self._say(f"reading the question at layer {kept['pick']}, as measured earlier on this model")
            return kept["pick"]
        self._say("measuring at which layer this model's questions are best read (once for a model; about 3 minutes on a 0.8B)")
        try:
            profile = profile_layers(self.model, self.tokenizer, self.wordings, calibration=self._calibration, options=self.options,
                                     progress=self._progress)
        except ValueError as error:   # too few wordings to hold half out, or too little ordinary text: nothing to measure with
            warnings.warn(f"the layer could not be measured ({error}): the last layer is used. Give layer= to set it by hand", stacklevel=3)
            return len(blocks(self.model)) - 1
        if min(profile.no_detector) >= 1.0:
            warnings.warn("the layer profile found no layer at which a probe fact gets a detector, so it has nothing to pick from: the "
                          "last layer is used", stacklevel=3)
            return len(blocks(self.model)) - 1
        self.layer_profile = profile.summary()
        keep_profile(key, self.layer_profile)
        self._say(f"reading the question at layer {profile.pick}" + (f" (the profile is flat over layers {profile.tied})" if profile.flat else ""))
        return profile.pick

    # ------------------------------------------------------------ calibration

    def calibration(self) -> dict:
        """c, c̄ and the covariance of the layer's MLP input on ordinary text (computed once, about 40 seconds). With `last_layer` set,
        the same passes also measure the size of what the final norm reads (final_scale)."""
        if self._mctx is None:
            self._say(f"calibrating layer {self.layer} on ordinary text")
            stats = calibration_stats(self.model, self.tokenizer, self._calibration or calibration_paragraphs(), self.layer,
                                      final=bool(self.options.last_layer))
            c, cbar, spread = constant_direction(stats)
            self._mctx = dict(c=c, cbar=cbar, cov=stats["cov"], spread=spread)
            if "final_scale" in stats:
                self._mctx["final_scale"] = stats["final_scale"]
        return self._mctx

    def final_scale(self) -> float:
        """R: the size of what the model's final norm reads on ordinary text (core.final_scale). Measured with the calibration when
        `last_layer` is set by then; otherwise once, on first use (a table loaded without it), and kept with the table's calibration."""
        mctx = self.calibration()
        if "final_scale" not in mctx:
            self._say("measuring what the final norm reads on ordinary text")
            mctx["final_scale"] = measure_final_scale(self.model, self.tokenizer, self._calibration or calibration_paragraphs())
        return mctx["final_scale"]

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

    def add_kind(self, name: str, names, generic_plain, generic_chat) -> list[str]:
        """Teach the table a kind of subject that does not ship with the package, so that relations about such subjects can be added
        (`Relation(..., subject=name)`). names: at least 24 invented names of such subjects. generic_plain, generic_chat: general
        questions about any such subject, each with {s} once (at least 8 and 4); a fact is taught to stay shut on them. A kind of the
        same name is replaced. Returns the kinds the table knows, besides company."""
        from .wordings import check_kind

        self.wordings = self.wordings.with_relations({}, {name: check_kind(name, dict(names=list(names), generic_plain=list(generic_plain),
                                                                                    generic_chat=list(generic_chat)))})
        return sorted(self.wordings.kinds)

    def add_wordings(self, relation: str, plain=(), chat=()) -> dict:
        """Teach a relation the table knows more ways of asking, such as the forms a check showed its facts miss. Facts added afterwards
        learn them; `rebuild()` builds the stored facts again so that they do too. The answer vectors stay taught from the relation's
        first wordings (`teach_cap`), so only the question detectors change."""
        relation = canonical_relation(relation)
        if relation not in self.wordings.relations:
            raise KeyError(f"unknown relation {relation!r}. Known: {', '.join(self.wordings.relations)}. A new relation needs add_relation().")
        old = self.wordings.relations[relation]
        self.wordings = self.wordings.with_more({relation: dict(plain=list(plain), chat=list(chat))})
        new = self.wordings.relations[relation]
        if self.options.teach_cap is None:
            self.options.teach_cap = TAUGHT_CAP
        self._shared = self._slots = None
        return dict(relation=relation, added_plain=len(new.plain) - len(old.plain), added_chat=len(new.chat) - len(old.chat), plain=len(new.plain),
                    chat=len(new.chat))

    def add_avoid(self, relation: str, plain=(), chat=()) -> dict:
        """Teach a relation the table knows what it must not answer: questions about the same company that are close to it and are not
        it, such as another job at the company for a chief executive. After `rebuild(relation)`, each of its facts reads them about its own
        company and about invented ones, as a group of negatives of their own, and keeps its threshold above every one of them. A plain
        one stops where an answer would come next. One equal to the relation's own wordings is refused. The answer vectors don't change.

        Experimental, not adopted (lab notebook, step 12): taught 5 other jobs, a table's chief-executive facts stopped answering 3 other
        jobs not taught, and answered 18 of 80 of their own questions instead of 59; in the weaker form (avoid_as="negatives"), 29. To a
        fact's detector another job at the company reads almost like its own relation, so editors are not offered it."""
        relation = canonical_relation(relation)
        if relation not in self.wordings.relations:
            raise KeyError(f"unknown relation {relation!r}. Known: {', '.join(self.wordings.relations)}. A new relation needs add_relation().")
        old = self.wordings.relations[relation]
        self.wordings = self.wordings.with_avoid({relation: dict(plain=list(plain), chat=list(chat))})
        new = self.wordings.relations[relation]
        return dict(relation=relation, added_plain=len(new.avoid_plain) - len(old.avoid_plain), added_chat=len(new.avoid_chat) - len(old.avoid_chat),
                    plain=len(new.avoid_plain), chat=len(new.avoid_chat))

    def rebuild(self, relation: str | None = None) -> int:
        """Build the stored facts again with the wordings as they are now: every fact, or only those of `relation`. After add_wordings, a
        fact's detector learns its relation's new wordings, and every other fact about the same kind of subject learns to stay shut on them;
        after add_avoid, the relation's facts learn to stay shut on what it must not answer. Returns how many were built."""
        if relation is not None:
            relation = canonical_relation(relation)
            if relation not in self.wordings.relations:
                raise KeyError(f"unknown relation {relation!r}. Known: {', '.join(self.wordings.relations)}.")
        stored = [(e["fact"]["subject"], e["fact"]["relation"], e["fact"]["target"]) for e in self._facts
                  if relation is None or e["fact"]["relation"] == relation]
        for subject, rel, answer in stored:
            self.add(subject, rel, answer)
        return len(stored)

    def add(self, subject: str, relation: str, answer: str) -> StoredFact:
        """Add a fact, or update it if the table already has this relation for this company. Returns the stored fact."""
        relation = canonical_relation(relation)
        if relation not in self.wordings.relations:
            raise KeyError(f"unknown relation {relation!r}. Known: {', '.join(self.wordings.relations)}. Add its wordings with add_relation().")
        aliases = {s: list(a) for s, a in self._aliases.items()}  # an update must not cost the company its declared names
        exclusions = {s: list(a) for s, a in self._exclusions.items()}
        self.remove(subject, relation)
        self._aliases, self._exclusions = aliases, exclusions
        fid = f"f{self._next:03d}"
        self._next += 1
        f = make_fact(subject.strip(), relation, answer.strip(), self.wordings, fid=fid)
        ctx = self._fact_context(f)
        mctx = self.calibration()
        self._say(f"building {subject} | {relation} | {answer}")
        parts = build_parts(self.model, self.tokenizer, f, ctx, mctx, self.layer)
        looser = build_looser(self.model, self.tokenizer, f, ctx, mctx, self.layer, parts) if self.options.looser_start else None
        prompt = build_prompt_detectors(self.model, self.tokenizer, f, ctx, mctx, self.layer) if self.options.prompt_detector else None
        self._facts.append(dict(fact=f, parts=parts, looser=looser, prompt=prompt))
        self._refresh()
        stored = self.facts[-1]
        self._say(f"added {stored.subject} | {stored.relation} | {stored.answer} in {parts['seconds']:.0f}s")
        return stored

    def detectors_of(self, subject: str, relation: str) -> list[str]:
        """The forms of question a stored fact got a detector for: "plain", "chat", both, or none. A form whose own questions the
        model does not tell from its other ones gets no detector (its median score is not above the threshold), and questions of
        that form are then not answered. More wordings for the relation can help (add_wordings, then rebuild)."""
        key, rel = " ".join(subject.lower().split()), canonical_relation(relation)
        entry = next((e for e in self._facts if " ".join(e["fact"]["subject"].lower().split()) == key and e["fact"]["relation"] == rel), None)
        if entry is None:
            raise KeyError(f"no stored fact for {subject!r} and {rel!r}")
        return sorted({d["form"] for d in fact_detectors(entry["parts"])})

    def _fact_context(self, f: dict) -> dict:
        """What building one stored fact needs (build.build_context), by the table's options."""
        return build_context([f], self.wordings, name="table", guard=False, real_world=self.options.real_world, context=self.options.context,
                             cache=self._read_cache(), inside=self.options.inside, twins=self.options.twins, teach_cap=self.options.teach_cap,
                             avoid_as=self.options.avoid_as)

    def _ensure_prompt_detectors(self) -> int:
        """Build the end-of-prompt detectors of the stored facts that have none: a table built or loaded with prompt_detector off, and the
        option switched on afterwards. A relation's facts are built together, since they share most of their readings. Returns how many."""
        missing = [e for e in self._facts if e.get("prompt") is None]
        if missing:
            mctx = self.calibration()
            self._say(f"building the end-of-prompt detectors of {len(missing)} facts")
            for e in sorted(missing, key=lambda e: e["fact"]["relation"]):
                e["prompt"] = build_prompt_detectors(self.model, self.tokenizer, e["fact"], self._fact_context(e["fact"]), mctx, self.layer)
        return len(missing)

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
            self._exclusions = {s: a for s, a in self._exclusions.items() if s in left}
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
        excluded = {norm(x) for names in self._exclusions.values() for x in names}
        for alias in aliases:
            if not norm(alias):
                raise ValueError("an alias can't be empty")
            if norm(alias) in excluded:
                raise ValueError(f"{alias!r} is declared as naming something else (an exclusion)")
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

    def add_exclusion(self, subject: str, *names: str) -> list[str]:
        """Declare longer names that contain a stored subject's name, or one of its aliases, and name something else: "Vanderbilt University
        Department of English" for "Vanderbilt University", "Form 1040-EZ" for "Form 1040". Key 1 matches them as it matches names, and
        where one stands the text names no stored subject, as for a company the table doesn't hold. Nothing is rebuilt. Returns the
        subject's exclusions."""
        norm = lambda s: " ".join(s.lower().split())  # noqa: E731
        stored = {norm(e["fact"]["subject"]): e["fact"]["subject"] for e in self._facts}
        if norm(subject) not in stored:
            raise ValueError(f"no fact is stored about {subject!r}")
        name = stored[norm(subject)]
        own = [norm(name)] + [norm(a) for a in self._aliases.get(name, [])]
        taken = set(stored) | {norm(a) for names_ in self._aliases.values() for a in names_}
        for longer in names:
            key = norm(longer)
            if not key:
                raise ValueError("an exclusion can't be empty")
            if key in taken:
                raise ValueError(f"{longer!r} is a stored name or a declared alias")
            if not any(re.search(r"(?<![0-9a-z])" + re.escape(o) + r"(?![0-9a-z])", key) for o in own):
                raise ValueError(f"{longer!r} does not contain {name!r} or one of its aliases as a whole word")
            if key not in {norm(x) for x in self._exclusions.get(name, [])}:
                self._exclusions.setdefault(name, []).append(" ".join(longer.split()))
        self._refresh()
        return list(self._exclusions.get(name, []))

    @property
    def exclusions(self) -> dict[str, list[str]]:
        """Each stored subject's declared exclusions: longer names that contain its name and name something else."""
        return {s: list(a) for s, a in self._exclusions.items()}

    def _exclusion_list(self) -> list[str]:
        return [x for names in self._exclusions.values() for x in names]

    def find(self, text: str) -> str | None:
        """The stored company a text names last, by its name or a declared alias, as the layer would see it."""
        return NameIndex(self.tokenizer, sorted({e["fact"]["subject"] for e in self._facts}), self._alias_map(),
                         self._exclusion_list()).find(self.tokenizer, text)

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
        if sum((self.options.looser_start, self.options.end_quantile is not None, self.options.prompt_detector)) > 1:
            raise ValueError("looser_start, end_quantile and prompt_detector share the layer's slot for the prompt's last token: set one of them")
        if self.options.looser_start and any(e.get("looser") is None for e in self._facts):
            raise ValueError("looser_start is on, and some stored facts were built without it: build them again with rebuild()")
        if self.options.last_layer not in (None, "later", "all"):
            raise ValueError(f"last_layer must be None, 'later' or 'all', not {self.options.last_layer!r}")
        if self.options.prompt_detector:
            self._ensure_prompt_detectors()
        entries =[entry_of(e["parts"], self.options.push, e["looser"] if self.options.looser_start else None, end_quantile=self.options.end_quantile,
                            prompt=e["prompt"] if self.options.prompt_detector else None)
                   for e in self._facts]
        kinds = {r: rel.subject for r, rel in self.wordings.relations.items()}
        return GraftLayer(self.model, self.tokenizer, entries, self.shared_detectors(), mctx["c"], mctx["cbar"], self.options.margin, self.layer,
                          relation_kinds=kinds if any(k != "company" for k in kinds.values()) else None, arm=self.options.arm,
                          aliases=self._alias_map(), slots=self.slot_detectors() if self.options.arm else None,
                          exclusions=self._exclusion_list(), last_layer=self.options.last_layer,
                          final_scale=self.final_scale() if self.options.last_layer else None)

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

    def weights_table(self) -> "Graft":
        """This table's facts built a second time at the model's last layer: what the grafted model is compiled from when the table
        itself reads the question at another layer.

        A fact neuron's gate is the fact's detector, and a neuron reads its own layer's input. The grafted model's neurons sit at
        the last layer, so their detectors must be fitted on readings of the last layer; this table's are fitted on its own
        layer's. The twin is an ordinary table at the last layer with this table's wordings, options, ordinary text and system
        message. It is kept in memory and saved with the table, and only what changed is built: a fact that is new, or was built
        again here, is built again there, and a fact removed here is removed there. It is never attached to the model."""
        from . import history

        top = len(blocks(self.model)) - 1
        options = replace(self.options, looser_start=False, prompt_detector=False, end_quantile=None, arm=0)  # no neuron form: left out
        norm = lambda s: " ".join(s.lower().split())  # noqa: E731
        with self.detached():   # the twin's facts are built on the model alone, as this table's were
            if self._twin is None and self._twin_path and os.path.exists(os.path.join(self._twin_path, "graft.json")):
                self._twin = Graft.load(self._twin_path, self.model, self._given_tokenizer, progress=self._progress)
                with open(os.path.join(self._twin_path, "built_from.json"), encoding="utf-8") as fh:
                    self._twin_built = {(s, r): sig for s, r, sig in json.load(fh)}
            if self._twin is None:
                self._twin = Graft(self.model, self._given_tokenizer, layer=top, wordings=self.wordings, options=options,
                                   calibration=self._calibration, progress=self._progress, system=self.system)
                self._twin_built = {}
            twin = self._twin
            twin.wordings, twin.options, twin._calibration, twin._progress = self.wordings, options, self._calibration, self._progress
            want = {(norm(e["fact"]["subject"]), e["fact"]["relation"]): (e, history.part_hash(e["parts"])) for e in self._facts}
            held = {(norm(e["fact"]["subject"]), e["fact"]["relation"]): e for e in twin._facts}
            for key, e in held.items():
                if key not in want:
                    twin.remove(e["fact"]["subject"], e["fact"]["relation"])
                    self._twin_built.pop(key, None)
            todo = [(key, e) for key, (e, sig) in want.items()
                    if self._twin_built.get(key) != sig or key not in held or held[key]["fact"]["target"] != e["fact"]["target"]]
            if todo:
                self._say(f"building {len(todo)} fact{'' if len(todo) == 1 else 's'} at the last layer, for the grafted model")
            for key, e in todo:
                twin.add(e["fact"]["subject"], e["fact"]["relation"], e["fact"]["target"])
                self._twin_built[key] = want[key][1]
            place = {key: i for i, key in enumerate(want)}
            twin._facts.sort(key=lambda e: place[(norm(e["fact"]["subject"]), e["fact"]["relation"])])   # this table's order
            twin._aliases = {s: list(a) for s, a in self._aliases.items()}
            twin._exclusions = {s: list(a) for s, a in self._exclusions.items()}
            twin.detach()
        return twin

    def later_spellings(self, layer: int) -> list[dict]:
        """Each fact's answer vectors measured at another layer than the table's (build.answer_spellings), built once and kept with the
        fact in memory: what a grafted model needs to follow answers from that layer. A few seconds per fact."""
        from .build import answer_spellings

        for n, e in enumerate(self._facts, 1):
            if layer not in e.setdefault("later", {}):
                e["later"][layer] = answer_spellings(self.model, self.tokenizer, e["fact"], self._fact_context(e["fact"]), layer)
                self._say(f"answer vectors at layer {layer}, {n}/{len(self._facts)}: {e['fact']['subject']} | {e['fact']['relation']}")
        return [e["later"][layer] for e in self._facts]

    def export_grafted(self, out_dir: str, *, base_dir: str | None = None, forms=("plain", "chat"), texts=None, complete: bool | str = True,
                       later_layer: int | None = None, last_budget: int | None = None) -> dict:
        """Compile the table's facts into the model's own neurons and write the grafted model to `out_dir`: the base checkpoint with the
        edited layers' MLP matrices replaced (same files, shapes and size), grafted.json and grafted_switch.py. Load it with
        cramschool.graft.load_grafted, or with transformers and the folder's grafted_switch.attach. base_dir: the local folder the
        model was loaded from (default: the model's own path).

        A table at the model's last layer that writes there (layer="last", last_layer="all"; lab notebook, step 26) is compiled into
        the last layer as far as that layer can afford. Each fact's first-token neurons go there: one per form and spelling, its gate
        the fact's detector and its output the sized row the table writes, so an answer that opens begins as the table's does. The
        last layer is the only place a write always wins and it can spare the fewest neurons: `last_budget` of them (default: 1/14
        of the layer, 256 of the 0.8B's, the largest count that met the damage bar). The later tokens' neurons fill what is left of
        the budget, every fact's second token first; a table of up to about 27 facts fits whole, and then one layer is edited. What
        doesn't fit goes to `later_layer` (default: layer 20 of a 24-layer model, the same depth of another), as pushes of answer
        vectors measured there, which can lose where a write at the last layer cannot.
        On bench40f that model passes 31 of 40 facts against 23 for the one-layer compile, with the bar met; 7 knife-edge known facts
        flip against 4, and the drift is 0.0097 against 0.0027.

        Any other table is compiled at its own layer, as below.

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
        if self.options.looser_start or self.options.end_quantile is not None or self.options.arm or self.options.prompt_detector:
            warnings.warn("export_grafted compiles each fact's strict detectors only: looser_start, end_quantile, prompt_detector and arm have "
                          "no neuron form and are left out of the grafted model", stacklevel=2)
        top = len(blocks(self.model)) - 1
        if self.options.last_layer == "all" and self.layer != top and complete is True:
            # This table reads the question at another layer than the last. The grafted model's neurons sit at the last layer, so it
            # is compiled from the same facts built there. The export names this table's facts, so its history links to this table's.
            twin = self.weights_table()
            twin._export_as = dict(table_facts=self.facts_fingerprint(), table_layer=self.layer)
            try:
                return twin.export_grafted(out_dir, base_dir=base, forms=forms, texts=texts, complete=True, later_layer=later_layer,
                                           last_budget=last_budget)
            finally:
                twin._export_as = None
        meta = dict(base_model=base, facts=len(self._facts), aliases=self._alias_map(), exclusions=self._exclusion_list(),
                    table_facts=self.facts_fingerprint())
        meta.update(getattr(self, "_export_as", None) or {})   # exported for a table that reads elsewhere: that table's facts are named
        if self.options.last_layer == "all" and self.layer == top and complete is True:
            from .layer import LAST_MULTIPLE

            from .grafted import last_layer_plan

            facts = self.grafted_facts()
            here, there, counts = last_layer_plan(facts, forms, mlp_of(self.model, top).up_proj.weight.shape[0], last_budget)
            self._say(f"the last layer's budget: {counts}")
            if counts["first"] > counts["budget"]:
                warnings.warn(f"{counts['first']} first-token neurons at the last layer, over its budget of {counts['budget']}. On Qwen3.5-0.8B "
                              "the damage bar was met at 256 reused neurons of the last layer's 3,584 and not at 384. Check the grafted "
                              "model (bench.scoring.FirmCheck) before relying on it", stacklevel=2)
            last = dict(head=self.model.get_output_embeddings().weight, size=LAST_MULTIPLE * self.final_scale(), cap=False)
            extra = text_moments(self.model, self.tokenizer, top, texts) if texts is not None else None
            first = compile_layer(self.model, self.tokenizer, facts, top, self.calibration(), forms=forms, push=self.options.push,
                                  paragraphs=self._calibration, extra_moments=extra, saturate=True, where="last", track=True, last=last,
                                  next_keys=here, progress=self._say, kept=self._moments)
            meta["last_budget"] = counts["budget"]
            later = default_layer(self.model) if later_layer is None else later_layer
            if later == top:
                if any(there):
                    raise ValueError(f"later_layer must be another layer than the last ({top}), which is already at its budget")
                return export_grafted(base, out_dir, first, meta=meta)   # no other layer to keep vectors for
            # Every answer's vectors at the other layer are measured and kept beside the weights, also when the whole table fits
            # the last layer: a merge with another grafted model may have to move later tokens there (grafted_merge).
            if later not in self._later_cal:   # measured once: it depends on the model, the layer and the text
                self._say(f"calibrating layer {later} on ordinary text, for the answer tokens that can move there")
                stats = calibration_stats(self.model, self.tokenizer, self._calibration or calibration_paragraphs(), later)
                self._later_cal[later] = constant_direction(stats)[:2]
            c, cbar = self._later_cal[later]
            spellings = self.later_spellings(later)
            kept = dict(layer=later, c=c, cbar=cbar, spellings=spellings)
            if not any(there):  # the whole table fits the last layer's budget: one layer is edited
                return export_grafted(base, out_dir, first, meta=meta, later=kept)
            vectors = [dict(subject=e["fact"]["subject"], relation=e["fact"]["relation"], answer=e["fact"]["target"], parts=dict(spellings=sp))
                       for e, sp in zip(self._facts, spellings)]
            extra = text_moments(self.model, self.tokenizer, later, texts) if texts is not None else None
            second = compile_layer(self.model, self.tokenizer, vectors, later, dict(c=c, cbar=cbar), forms=forms, push=self.options.push,
                                   paragraphs=self._calibration, extra_moments=extra, saturate=True, where="last", track=True,
                                   kinds=("next",), next_keys=there, progress=self._say, kept=self._moments)
            return export_grafted(base, out_dir, first, meta=meta, also=second, later=kept)
        if self.layer_profile is not None:
            warnings.warn(f"this table's layer ({self.layer}) was picked for reading the question. The grafted model also pushes the answer "
                          "there, and a push was only measured near layer 20 of the 0.8B: check the grafted model before relying on it", stacklevel=2)
        if self.options.last_layer:
            warnings.warn("the grafted model of this table writes every answer token at the table's layer, so an answer the table finishes can "
                          "come out unfinished from it. To have the write at the last layer in the weights, build the table at the model's "
                          "last layer: Graft(model, tokenizer, layer='last')", stacklevel=2)
        facts = self.grafted_facts(detected=complete == "detected")
        extra = text_moments(self.model, self.tokenizer, self.layer, texts) if texts is not None else None
        compiled = compile_layer(self.model, self.tokenizer, facts, self.layer, self.calibration(), forms=forms, push=self.options.push,
                                 paragraphs=self._calibration, extra_moments=extra, saturate=bool(complete), where="last" if complete else "after",
                                 track=complete is True, progress=self._say, kept=self._moments)
        return export_grafted(base, out_dir, compiled, meta=meta)

    # ------------------------------------------------------------ saving

    def _fingerprint(self) -> str:
        return tensor_fingerprint(mlp_of(self.model, self.layer).down_proj.weight)[:16]

    def save(self, path: str) -> str:
        """A folder: graft.json (the facts and settings), calibration.pt, shared.pt, and one parts file per fact (with the looser start's
        and the end-of-prompt detectors' beside it, for the facts that have them)."""
        os.makedirs(os.path.join(path, "parts"), exist_ok=True)
        mctx = self.calibration()
        if self.options.last_layer:
            self.final_scale()  # kept with the table, so that asking it later doesn't measure it again
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
            if e.get("prompt") is not None:
                torch.save(e["prompt"], os.path.join(path, "parts", f"{fid}.prompt.pt"))
            facts.append(dict(id=fid, subject=e["fact"]["subject"], relation=e["fact"]["relation"], answer=e["fact"]["target"]))
        meta = dict(format=FORMAT_VERSION, method="GRAFT", saved=time.strftime("%Y-%m-%d %H:%M:%S"),
                    model=getattr(self.model.config, "_name_or_path", ""), layer=self.layer, fingerprint=self._fingerprint(),
                    options=asdict(self.options), relations={r: rel.as_dict() for r, rel in self.wordings.relations.items()},
                    kinds={k: v for k, v in self.wordings.kinds.items() if any(rel.subject == k for rel in self.wordings.relations.values())},
                    next=self._next, facts=facts, aliases=self._aliases, exclusions=self._exclusions, system=self.system,
                    layer_profile=self.layer_profile)
        with open(os.path.join(path, "graft.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        if self._calibration is not None:  # ordinary text of your own: kept, so that a loaded table exports with it and not the bundled text
            with open(os.path.join(path, ORDINARY_TEXT), "w", encoding="utf-8") as fh:
                json.dump(list(self._calibration), fh, ensure_ascii=False)
        self._save_twin(path)
        return path

    def _save_twin(self, path: str) -> None:
        """Keep, in the table's folder, the facts as built at the last layer for the grafted model (weights_table), so that the next
        export builds only what is new. The table's history does not record these files: they are rebuilt from the table when missing."""
        if self._twin is None or not len(self._twin):
            return
        twin_path = os.path.join(path, WEIGHTS_PARTS)
        self._twin.save(twin_path)
        with open(os.path.join(twin_path, "built_from.json"), "w", encoding="utf-8") as fh:
            json.dump([[s, r, sig] for (s, r), sig in self._twin_built.items()], fh, ensure_ascii=False)
        self._twin_path = twin_path

    @classmethod
    def load(cls, path: str, model, tokenizer, *, check_model: bool = True, progress: Callable[[str], None] | None = None,
             version: str | int | None = None) -> "Graft":
        """Load a saved table onto the same base model. Refuses a model whose layer weights differ, unless check_model=False.

        version: a version recorded with `commit` ("v2", 2 or "latest"), in place of the table as the folder holds it now. The folder
        is not changed."""
        if version is not None:
            from . import history

            with history.extracted(path, version) as folder:
                return cls.load(folder, model, tokenizer, check_model=check_model, progress=progress)
        with open(os.path.join(path, "graft.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported table format {meta.get('format')!r}")
        builtin = Wordings.builtin()
        wordings = Wordings(relations={r: Relation(name=r, plain=d["plain"], chat=d["chat"], person=d.get("person", True), subject=d.get("subject", "company"),
                                                   avoid_plain=d.get("avoid_plain", []), avoid_chat=d.get("avoid_chat", []))
                                       for r, d in meta["relations"].items()},
                            generic=builtin.generic, kinds={**builtin.kinds, **meta.get("kinds", {})})
        # a table saved before last_layer existed was built and measured with every answer token written at its own layer: it stays so
        options = GraftOptions(**{"last_layer": None, **meta["options"]})
        own_text = None
        if os.path.exists(os.path.join(path, ORDINARY_TEXT)):
            with open(os.path.join(path, ORDINARY_TEXT), encoding="utf-8") as fh:
                own_text = json.load(fh)
        graft = cls(model, tokenizer, layer=meta["layer"], wordings=wordings, options=options, progress=progress, system=meta.get("system"),
                    calibration=own_text)
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
            pp = os.path.join(path, "parts", f"{d['id']}.prompt.pt")
            prompt = torch.load(pp, weights_only=True) if os.path.exists(pp) else None
            f = make_fact(d["subject"], d["relation"], d["answer"], wordings, fid=d["id"])
            graft._facts.append(dict(fact=f, parts=parts, looser=looser, prompt=prompt))
        graft._next = meta.get("next", len(graft._facts) + 1)
        graft.layer_profile = meta.get("layer_profile")
        graft._aliases = {s: list(a) for s, a in meta.get("aliases", {}).items()}
        graft._exclusions = {s: list(a) for s, a in meta.get("exclusions", {}).items()}
        graft._twin_path = os.path.join(path, WEIGHTS_PARTS)
        return graft

    # ------------------------------------------------------------ history

    def commit(self, path: str, message: str = ""):
        """Save the table to `path` and record it as a version (graft.history): `Graft.load(path, model, tokenizer, version="v2")` loads
        any version back as it was saved, and `revert` goes back to one. Returns the `Version`. A table that is unchanged since its
        newest version records nothing, and that version is returned."""
        from . import history

        self.save(path)
        return history.commit(path, message)

    def revert(self, path: str, version, *, discard: bool = False):
        """Go back to a recorded version: the folder holds it again, this table is that version, and the step is recorded as a new
        version (history is not rewritten). Nothing is built. Changes made to this table since it was last saved are dropped. Changes
        saved to the folder that no version holds are refused, unless `discard` is set. Returns the new `Version`."""
        from . import history

        record = history.revert(path, version, discard=discard)  # file work: a refused revert leaves this table as it is
        was_attached, calibration, cache = self.is_attached, self._calibration, getattr(self, "_cache", None)
        self.detach()
        fresh = Graft.load(path, self.model, self._given_tokenizer, progress=self._progress)
        self.__dict__.update(fresh.__dict__)
        self._calibration, self._cache = calibration, cache
        if was_attached and self._facts:
            self.attach()
        return record

    def merge(self, other: "Graft", prefer: str = "error") -> dict:
        """Take the facts of another table of the same model into this one, without building anything.

        A fact's parts are measured from the model, the layer, the system message, the relations' wordings and the options of the build,
        and from no other stored fact, so they can move between two tables in which all of these are the same. Anything else is
        refused: add the facts with `add_many` then, which builds them for this table.

        prefer: what to do where both tables hold the same subject and relation with another answer. "error" (the default) changes
        nothing and raises, naming them; "mine" keeps this table's; "theirs" takes the other's.

        The merged table keeps its own shared detectors, which veto a fact when another relation's detector leads; the other table's
        facts answered behind that table's. The other table's aliases and exclusions come with its facts where they don't clash.
        Returns added, replaced, kept (conflicts left as they were), same (facts both held alike) and skipped_names."""
        if prefer not in ("error", "mine", "theirs"):
            raise ValueError(f"prefer must be 'error', 'mine' or 'theirs', not {prefer!r}")
        build = ("real_world", "context", "inside", "twins", "teach_cap", "avoid_as")
        used = lambda w: {k: v for k, v in w.kinds.items() if any(rel.subject == k for rel in w.relations.values())}  # noqa: E731
        pairs = [("the layer", self.layer, other.layer), ("the system message", self.system, other.system),
                 ("the relations' wordings", {r: rel.as_dict() for r, rel in self.wordings.relations.items()},
                  {r: rel.as_dict() for r, rel in other.wordings.relations.items()}),
                 ("the generic questions", (self.wordings.generic, used(self.wordings), self.wordings.guard),
                  (other.wordings.generic, used(other.wordings), other.wordings.guard)),
                 ("the build options", {k: getattr(self.options, k) for k in build}, {k: getattr(other.options, k) for k in build})]
        differs = [name for name, mine, theirs in pairs if mine != theirs]
        if self.layer == other.layer and self._fingerprint() != other._fingerprint():
            differs.append("the model")
        if not differs:  # measured from the model and the calibration text: the same in two tables only if both are the same
            mine, theirs = self.calibration(), other.calibration()
            alike = lambda a, b: torch.equal(a.cpu(), b.cpu()) if torch.is_tensor(a) and torch.is_tensor(b) else a == b  # noqa: E731
            if not all(alike(mine[k], theirs[k]) for k in ("c", "cbar", "cov")):
                differs.append("the calibration")
        if differs:
            raise ValueError(f"these tables can't be merged: {', '.join(differs)} differ{'s' if len(differs) == 1 else ''}. "
                             "Add the other table's facts with add_many, which builds them for this table.")
        if self.options.looser_start and any(e.get("looser") is None for e in other._facts):
            raise ValueError("this table opens with looser_start, and the other table's facts were built without it. Add them with add_many.")
        norm = lambda s: " ".join(s.lower().split())  # noqa: E731
        held = {(norm(e["fact"]["subject"]), e["fact"]["relation"]): e for e in self._facts}
        new, conflicts, same = [], [], 0
        for e in other._facts:
            key = (norm(e["fact"]["subject"]), e["fact"]["relation"])
            if key not in held:
                new.append(e)
            elif held[key]["fact"]["target"] == e["fact"]["target"]:
                same += 1
            else:
                conflicts.append(e)
        describe = lambda e: dict(subject=e["fact"]["subject"], relation=e["fact"]["relation"], answer=e["fact"]["target"])  # noqa: E731
        if conflicts and prefer == "error":
            raise ValueError("both tables hold these facts with another answer (prefer='mine' or 'theirs' decides): "
                             + "; ".join(f"{e['fact']['subject']} | {e['fact']['relation']}" for e in conflicts))
        replaced = conflicts if prefer == "theirs" else []
        for e in replaced:
            key = (norm(e["fact"]["subject"]), e["fact"]["relation"])
            self._facts = [mine for mine in self._facts if mine is not held[key]]
        for e in new + replaced:
            fid = f"f{self._next:03d}"
            self._next += 1
            self._facts.append(dict(fact=dict(e["fact"], id=fid), parts=e["parts"], looser=e.get("looser"), prompt=e.get("prompt")))
        skipped = []
        taken_from = {e["fact"]["subject"] for e in new + replaced}
        for declare, names in ((self.add_alias, other._aliases), (self.add_exclusion, other._exclusions)):
            for subject, values in names.items():
                for value in values if subject in taken_from else ():
                    try:
                        declare(subject, value)
                    except ValueError as error:
                        skipped.append(dict(subject=subject, name=value, why=str(error)))
        self._refresh()
        return dict(added=[describe(e) for e in new], replaced=[describe(e) for e in replaced],
                    kept=[describe(held[(norm(e["fact"]["subject"]), e["fact"]["relation"])]) for e in conflicts] if prefer == "mine" else [],
                    same=same, skipped_names=skipped)

    def bisect(self, path: str, test: Callable[["Graft"], bool], good, bad=None) -> dict:
        """Find the first recorded version at which a test of yours fails. test(table) gets each version loaded onto this table's model
        and returns True for a pass. `good` passes and `bad` (the newest version by default) fails; about log2 of the versions between
        them are loaded. Returns first_bad, last_good, checked and the diff between the two (history.first_bad). This table and the
        folder are left as they are."""
        from . import history

        def check(version) -> bool:
            table = Graft.load(path, self.model, self._given_tokenizer, version=version.id)
            try:
                return bool(test(table))
            finally:
                table.detach()

        with self.detached():
            return history.first_bad(path, check, good, bad)

    def _base_folder(self, base_dir: str | None) -> str:
        base = base_dir or getattr(self.model.config, "_name_or_path", "")
        if not base or not os.path.isdir(base):
            raise ValueError("give base_dir: the local folder, in safetensors format, that the model was loaded from")
        return base

    @staticmethod
    def _export_record(export: dict) -> dict:
        """The export's arguments as they are kept with a version of the weights: plain values, and whether extra text was given."""
        return {k: (bool(v is not None) if k == "texts" else list(v) if isinstance(v, tuple) else v) for k, v in export.items()}

    def facts_fingerprint(self) -> str:
        """One hash of the table's facts and of what each was built into (history.facts_fingerprint). `export_grafted` writes it into
        grafted.json, so that a version of the weights can be matched with the table version it was compiled from by content: the
        fingerprint of a recorded table version is computed from its files (history.fingerprint_of) and is the same hash."""
        from . import history

        return history.facts_fingerprint([e["fact"]["subject"], e["fact"]["relation"], e["fact"]["target"], history.part_hash(e["parts"])]
                                         for e in self._facts)

    def merge_grafted(self, other: "Graft", path: str, grafted_dir: str, history_dir: str, message: str = "", *, prefer: str = "error",
                      base_dir: str | None = None, **export) -> dict:
        """Bring another table's facts into this table and into the model's own weights, in one call: `merge`, then `commit_grafted`.

        This is for two tables: the merge is at the tables, where facts are independent, and the grafted model of the merged
        table is then compiled once, so both forms are recorded and linked. Both tables must be loaded (`Graft.load`) and built
        the same way; see `merge`. Two grafted models with no tables at hand are merged at the weights by
        `grafted_merge.merge_grafted_models`, which gives the same weights.

        Returns merged (the report of `merge`), and table, weights, report and not_in_weights as `commit_grafted` does."""
        base = self._base_folder(base_dir)                       # checked before anything is merged
        merged = self.merge(other, prefer)
        return dict(merged=merged, **self.commit_grafted(path, grafted_dir, history_dir, message, base_dir=base, **export))

    def commit_grafted(self, path: str, grafted_dir: str, history_dir: str, message: str = "", *, base_dir: str | None = None, **export) -> dict:
        """Record both forms of the facts in one call: the table as a version (`commit`), the same facts in the model's own weights
        (`export_grafted`, to `grafted_dir`), and those weights as a version that names the table's (weight_history.commit, in
        `history_dir`). The weights' version names exactly the table version recorded here, not whichever is newest.

        base_dir: the local folder the model was loaded from (default: the model's own path). export: `export_grafted`'s arguments
        (forms, complete, later_layer, last_budget, texts), kept with the weights' version, since the same table exported with other
        arguments gives other weights.

        If the export fails, the table's version is recorded and the weights' is not: call again once the export can run. An
        unchanged table and unchanged weights record nothing.

        Returns table and weights (the two versions), report (the export's) and not_in_weights: the table's facts that got no neuron,
        because their detector could not be built. Those answer from the table and not from the grafted model."""
        from .. import weight_history

        base = self._base_folder(base_dir)                       # checked before anything is recorded
        table_version = self.commit(path, message)
        report = self.export_grafted(grafted_dir, base_dir=base, **export)
        self._save_twin(path)   # what the export built at the last layer is kept beside the table
        weights_version = weight_history.commit(history_dir, grafted_dir, message, base_dir=base, table=path, table_version=table_version.id,
                                                export=self._export_record(export))
        return dict(table=table_version, weights=weights_version, report=report, not_in_weights=self._not_in_weights(grafted_dir))

    def _not_in_weights(self, grafted_dir: str) -> list[dict]:
        """The stored facts that the grafted model in a folder holds no neuron for."""
        with open(os.path.join(grafted_dir, "grafted.json"), encoding="utf-8") as fh:
            held = {(n["subject"], n["relation"], n["answer"]) for n in json.load(fh).get("neurons", [])}
        return [dict(subject=f.subject, relation=f.relation, answer=f.answer) for f in self.facts if (f.subject, f.relation, f.answer) not in held]

    def revert_grafted(self, path: str, version, grafted_dir: str, history_dir: str, *, base_dir: str | None = None, discard: bool = False,
                       **export) -> dict:
        """Go back in both forms: the table to a recorded version (`revert`), and `grafted_dir` to the weights compiled from it.

        The weights are found by the table version's state, since a table that went back has a new version id and the old state. If
        several versions of the weights were compiled from that state, the newest is taken (they can differ in the export's
        arguments). If none was, the table is exported again, with `export`, and recorded. Changes that no version holds, in the
        table's folder or in `grafted_dir`, are refused before anything is changed, unless `discard` is set.

        Returns table and weights (the two new versions), exported (whether an export was needed) and matches (the ids found)."""
        from .. import weight_history
        from . import history

        base = self._base_folder(base_dir)
        known = history.versions(path)
        name = known[-1].id if version in (None, "latest") and known else f"v{version}" if isinstance(version, int) else str(version)
        target = next((v for v in known if v.id == name), None)
        if target is None:
            raise ValueError(f"{path} has no version {name!r}" + (f"; it has {known[0].id} to {known[-1].id}" if known else ": record one with commit"))
        has_weights = os.path.exists(os.path.join(history_dir, weight_history.LOG))
        matches = weight_history.compiled_from(history_dir, target.state) if has_weights else []
        if has_weights and not discard and not weight_history.holds_recorded(history_dir, grafted_dir):
            raise ValueError(f"the checkpoint in {grafted_dir} is one that no version holds. Record it first, or pass discard to lose it.")
        table_version = self.revert(path, name, discard=discard)
        if matches:
            weights_version = weight_history.revert(history_dir, matches[-1].id, grafted_dir, base_dir=base, discard=True)
            return dict(table=table_version, weights=weights_version, exported=False, matches=[m.id for m in matches])
        self.export_grafted(grafted_dir, base_dir=base, **export)
        weights_version = weight_history.commit(history_dir, grafted_dir, f"back to table {name}", base_dir=base, table=path,
                                                table_version=table_version.id, export=self._export_record(export))
        return dict(table=table_version, weights=weights_version, exported=True, matches=[], not_in_weights=self._not_in_weights(grafted_dir))

    def _say(self, text: str) -> None:
        if self._progress is not None:
            self._progress(text)

    def __repr__(self) -> str:
        return f"Graft({len(self)} facts, layer {self.layer}, {'attached' if self.is_attached else 'detached'})"

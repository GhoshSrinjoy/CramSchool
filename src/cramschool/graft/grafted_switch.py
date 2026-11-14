"""The name switch of a grafted model. cramschool writes a copy of this file next to every grafted checkpoint, so the model runs with torch
and transformers alone.

A grafted model holds its facts in the model's own MLP neurons at one layer. Each such neuron's gate is the fact's question detector and its
output is the fact's answer. Its up value is 0 in the checkpoint, so without the switch the facts are dormant and the model behaves like
the base model. The switch sets a fact's neurons on (up value 1) at every token after the fact's subject is named, as the inserted layer's
name index reads names: the decoded text, case ignored, spaces collapsed, whole words, and the subject named last counts.

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import grafted_switch                                     # this file, in the model folder
    model = AutoModelForCausalLM.from_pretrained(folder)
    tokenizer = AutoTokenizer.from_pretrained(folder)
    switch = grafted_switch.attach(model, tokenizer, folder)  # switch.detach() makes the facts dormant again

A forward call whose cache is empty starts a new text; a call that continues a cache continues the text the switch has read. Names given
only as input embeddings can't be read, so they switch nothing on. Beam search is not supported: the switch doesn't follow the beams'
reordering.

Where the facts act is written in grafted.json. "after" (the first grafted models): at every token after the subject's name. "last": only at
the last position of each forward call, the token that chooses the next one, as the inserted layer does; in one pass over a whole text,
that is its final token alone.

A model compiled for answers of several tokens ("track" in grafted.json) also has a neuron for each later answer token. For it the switch
keeps the inserted layer's bookkeeping, one answer per text: when a fact's first-token neurons fire and the next token is the answer's
first, it switches on the neuron for each next token while the tokens read are the answer's. For that it reads those neurons' activations
and holds the answers' token ids. The detectors and the pushes are in the weights. Such an answer is followed only through a cache, as
generation does.
"""
from __future__ import annotations

import json
import os
import re

import torch

SPEC_FILE = "grafted.json"
FIRED = 0.05  # a fact's first token counts as started when its neurons write at least this share of the push
_SPACES = re.compile(r"\s+")


def _stack(model):
    """The module holding the decoder layers (it receives the token ids and the cache, whoever calls it)."""
    inner = model.model
    return inner if hasattr(inner, "layers") else inner.language_model


def _layers(model):
    return _stack(model).layers


def _seen(past) -> int:
    """How many tokens a cache already holds: 0 for a new text."""
    if past is None:
        return 0
    try:
        return int(past.get_seq_length())
    except Exception:  # a cache type without a length: treat the call as a new text
        return 0


class NameIndex:
    """Every stored subject name, matched on the decoded text: case ignored, spaces collapsed, as a whole word. Its state carries across
    forward calls, so it follows generation token by token. step() returns the subject named last (an index), or None."""

    def __init__(self, tokenizer, names: list[str], aliases: dict[str, str] | None = None):
        """aliases: other names a text may use for a stored one, as {alias: name}. An alias is matched like a name, as a whole word."""
        self.names = [" ".join(n.lower().split()) for n in names]
        self.lookup = {n: i for i, n in enumerate(self.names)}
        for alias, name in (aliases or {}).items():
            self.lookup[" ".join(alias.lower().split())] = self.lookup[" ".join(name.lower().split())]
        alternatives = sorted(self.lookup, key=len, reverse=True)
        self.pattern = re.compile(r"(?<![0-9a-z])(?:" + "|".join(re.escape(a) for a in alternatives) + r")") if alternatives else None
        self.longest = max(map(len, self.lookup)) if self.lookup else 0
        self.texts = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
        self.rows: list[dict] = []

    def reset(self, n: int) -> None:
        self.rows = [dict(buf="", current=None, before=None, provisional=False) for _ in range(n)]

    def step(self, b: int, tok: int):
        st = self.rows[b]
        piece = _SPACES.sub(" ", self.texts[tok].lower())
        if st["provisional"]:  # a name ended with the previous token: a letter or digit now makes it part of a longer word
            st["provisional"] = False
            if piece[:1].isalnum():
                st["current"] = st["before"]
        buf = st["buf"]
        if buf.endswith(" ") and piece.startswith(" "):
            piece = piece[1:]
        start = len(buf)
        buf += piece
        if piece and self.pattern is not None:
            for m in self.pattern.finditer(buf, max(0, start - self.longest)):
                if m.end() <= start:
                    continue
                if m.end() == len(buf):
                    st["before"], st["current"], st["provisional"] = st["current"], self.lookup[m.group(0)], True
                elif not buf[m.end()].isalnum():
                    st["current"] = self.lookup[m.group(0)]
        st["buf"] = buf[-(self.longest + 32):]
        return st["current"]


class Switch:
    """Hooks at the facts' layer: one reads the token ids of each forward call, one sets the fact neurons' up values at the layer's MLP.
    With `track` (a model compiled for answers of several tokens), one more reads which facts' first-token neurons fired."""

    def __init__(self, model, tokenizer, layer: int, slots: list[int], subjects: list[str], where: str = "after", track: dict | None = None,
                 aliases: dict | None = None):
        """track: None, or dict(neurons, answers, clip) from grafted.json: each neuron's kind ("first", "clip" or "next"), fact, form and
        spelling or step and token; each fact's answer token ids by spelling; the first-token pair's saturation value."""
        if where not in ("after", "last"):
            raise ValueError(f"where must be 'after' or 'last', not {where!r}")
        names = sorted(set(subjects))
        self.index = NameIndex(tokenizer, names, aliases)  # aliases: declared other names of the subjects, as {alias: subject}
        lookup = {n: self.index.lookup[" ".join(n.lower().split())] for n in names}
        device = next(model.parameters()).device
        self.slots = torch.tensor(slots, dtype=torch.long, device=device)
        self.owner = torch.tensor([lookup[s] for s in subjects], dtype=torch.long, device=device)
        self.track = track is not None
        self.last_only = where == "last" or self.track
        self.n, self.current, self.last_valid, self.last_tok = None, None, None, None
        mlp = _layers(model)[layer].mlp
        self.handles = [_stack(model).register_forward_pre_hook(self._read_ids, with_kwargs=True), mlp.up_proj.register_forward_hook(self._set_up)]
        if self.track:
            self.clip = float(track.get("clip", 8.0))
            self.answers = [{sp: [int(i) for i in ids] for sp, ids in a.items()} for a in track["answers"]]
            self.pairs, self.next, self.state = {}, {}, []  # subject -> [(fact, spelling, first, clip)]; (fact, step, token) -> neuron
            firsts = {(n["fact"], n["form"]): (pos, n["spelling"]) for pos, n in enumerate(track["neurons"]) if n["kind"] == "first"}
            owner = self.owner.tolist()
            for pos, n in enumerate(track["neurons"]):
                if n["kind"] == "clip":
                    first, spelling = firsts[(n["fact"], n["form"])]
                    self.pairs.setdefault(owner[first], []).append((n["fact"], spelling, first, pos))
                elif n["kind"] == "next":
                    self.next[(n["fact"], n["step"], n["token"])] = pos
            self.handles.append(mlp.down_proj.register_forward_pre_hook(self._fired))

    def _read_ids(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        start = _seen(kwargs.get("past_key_values"))
        if ids is None:  # embeddings only: no names can be read here, so every fact stays dormant for these tokens
            if start == 0:
                self.n = None  # a new text: the next call with ids starts from an empty index
            self.current = None
            return
        mask = kwargs.get("attention_mask")
        n, t = ids.shape
        if start == 0 or self.n != n:  # a new text (empty cache), or a different batch
            self.index.reset(n)
            self.n = n
            if self.track:
                self.state = [dict(pending=[], progress=None, done=False, armed=[]) for _ in range(n)]
        valid = (mask[:, -t:].bool() if mask is not None else torch.ones_like(ids, dtype=torch.bool)).tolist()
        rows, idl = [], ids.tolist()
        for b in range(n):
            c, row = self.index.rows[b]["current"], []
            for i in range(t):
                if valid[b][i]:
                    c = self.index.step(b, idl[b][i])
                row.append(-1 if c is None else c)
            rows.append(row)
        self.current = torch.tensor(rows, dtype=torch.long, device=ids.device)
        self.last_valid = torch.tensor([bool(v[-1]) for v in valid], device=ids.device)
        self.last_tok = [row[-1] for row in idl]

    def _set_up(self, module, args, output):
        out = output.clone()
        cur = self.current
        if cur is None or cur.shape[0] != out.shape[0] or cur.shape[1] < out.shape[1]:  # nothing read for this call: dormant
            out[..., self.slots] = 0
            return out
        if self.track:
            out[..., self.slots] = self._tracked(cur, out).to(out.dtype)
            return out
        named = cur[:, -out.shape[1]:, None].to(self.owner.device)
        on = named == self.owner[None, None, :]
        if self.last_only:  # the facts act where the next token is chosen: the call's last position, when it is a real token
            last = on[:, -1] & self.last_valid[:, None].to(on.device)
            on = torch.zeros_like(on)
            on[:, -1] = last
        out[..., self.slots] = on.to(out.dtype)
        return out

    def _tracked(self, cur, out):
        """Which fact neurons are on at this call's last position, by the inserted layer's bookkeeping: the continuation neuron of an answer
        in progress while the tokens read are the answer's; else, until an answer has been given, the first-token neurons of the subject
        named last."""
        on = torch.zeros(out.shape[0], out.shape[1], len(self.slots), dtype=torch.bool, device=out.device)
        subject, valid = cur[:, -1].tolist(), self.last_valid.tolist()
        for b, st in enumerate(self.state):
            st["armed"] = []
            if not valid[b]:
                continue
            tok, lit = self.last_tok[b], []
            if st["progress"] is not None:  # (fact, spelling, j): the answer's token j was expected here
                f, sp, j = st["progress"]
                a = self.answers[f][sp]
                if tok == a[j] and j + 1 < len(a):
                    lit, st["progress"] = [self.next.get((f, j + 1, a[j + 1]))], (f, sp, j + 1)
                else:
                    st["progress"] = None
            elif st["pending"]:  # facts whose first-token neurons fired at the token before
                pending, st["pending"] = st["pending"], []
                for f, sp in pending:
                    a = self.answers[f][sp]
                    if tok == a[0]:
                        st["done"] = True  # one answer per text
                        if len(a) > 1:
                            lit, st["progress"] = [self.next.get((f, 1, a[1]))], (f, sp, 1)
                        break
            lit = [p for p in lit if p is not None]
            if not lit and not st["done"] and st["progress"] is None and subject[b] >= 0:
                st["armed"] = self.pairs.get(subject[b], [])
                lit = [p for _, _, first, clip in st["armed"] for p in (first, clip)]
            if lit:
                on[b, -1, lit] = True
        return on

    def _fired(self, module, args):
        """After the neurons ran: a fact whose first-token pair wrote at least FIRED of its push at the last position is pending."""
        a = args[0]
        if self.current is None or a.shape[0] != len(self.state):
            return
        for b, st in enumerate(self.state):
            if st["armed"]:
                row = a[b, -1, self.slots].float()
                st["pending"] = [(f, sp) for f, sp, first, clip in st["armed"] if float(row[first] - row[clip]) >= FIRED * self.clip]
                st["armed"] = []

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def switch_for(model, tokenizer, spec: dict) -> Switch:
    """The switch for a compiled layer or a grafted.json: its layer, neurons, where the facts act, and the tracking of answers."""
    neurons = spec["neurons"]
    track = dict(neurons=neurons, answers=spec["answers"], clip=spec.get("clip", 8.0)) if spec.get("track") else None
    return Switch(model, tokenizer, spec["layer"], [n["slot"] for n in neurons], [n["subject"] for n in neurons], where=spec.get("where", "after"),
                  track=track, aliases=spec.get("aliases"))


def attach(model, tokenizer, folder: str) -> Switch:
    """Switch on the facts compiled into the grafted model in `folder` (its grafted.json) for this loaded model."""
    with open(os.path.join(folder, SPEC_FILE), encoding="utf-8") as fh:
        return switch_for(model, tokenizer, json.load(fh))

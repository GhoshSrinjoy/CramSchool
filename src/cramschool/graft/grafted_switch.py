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
"""
from __future__ import annotations

import json
import os
import re

import torch

SPEC_FILE = "grafted.json"
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

    def __init__(self, tokenizer, names: list[str]):
        self.names = [" ".join(n.lower().split()) for n in names]
        self.lookup = {n: i for i, n in enumerate(self.names)}
        alternatives = sorted(set(self.names), key=len, reverse=True)
        self.pattern = re.compile(r"(?<![0-9a-z])(?:" + "|".join(re.escape(a) for a in alternatives) + r")") if alternatives else None
        self.longest = max(map(len, self.names)) if self.names else 0
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
    """Two hooks: one reads the token ids of each forward call; one sets the fact neurons' up values at the layer's MLP."""

    def __init__(self, model, tokenizer, layer: int, slots: list[int], subjects: list[str]):
        names = sorted(set(subjects))
        self.index = NameIndex(tokenizer, names)
        lookup = {n: self.index.lookup[" ".join(n.lower().split())] for n in names}
        device = next(model.parameters()).device
        self.slots = torch.tensor(slots, dtype=torch.long, device=device)
        self.owner = torch.tensor([lookup[s] for s in subjects], dtype=torch.long, device=device)
        self.n, self.current = None, None
        self.handles = [_stack(model).register_forward_pre_hook(self._read_ids, with_kwargs=True),
                        _layers(model)[layer].mlp.up_proj.register_forward_hook(self._set_up)]

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

    def _set_up(self, module, args, output):
        out = output.clone()
        cur = self.current
        if cur is None or cur.shape[0] != out.shape[0] or cur.shape[1] < out.shape[1]:  # nothing read for this call: dormant
            out[..., self.slots] = 0
            return out
        named = cur[:, -out.shape[1]:, None].to(self.owner.device)
        out[..., self.slots] = (named == self.owner[None, None, :]).to(out.dtype)
        return out

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def attach(model, tokenizer, folder: str) -> Switch:
    """Switch on the facts compiled into the grafted model in `folder` (its grafted.json) for this loaded model."""
    with open(os.path.join(folder, SPEC_FILE), encoding="utf-8") as fh:
        spec = json.load(fh)
    neurons = spec["neurons"]
    return Switch(model, tokenizer, spec["layer"], [n["slot"] for n in neurons], [n["subject"] for n in neurons])

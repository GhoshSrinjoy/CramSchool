"""The inserted layer: the name index (key 1) and the hooks that read, compete and push (key 2 and the push). docs/graft.md, sections 2 to 6.

`attach()` registers three hooks on the model and `detach()` removes them; no weight of the model is ever written, so detaching is an exact
undo.
"""
from __future__ import annotations

import re

import torch

from ..model_access import model_device
from .build import pick_spelling
from .core import mlp_of, stack_of, token_texts
from .wordings import RELATIONS

_SPACES = re.compile(r"\s+")


class NameIndex:
    """Every stored company name, matched on the decoded text: case ignored, spaces collapsed, as a whole word. Its state carries across
    forward calls, so it follows generation token by token. step() returns the company named last (an index), or None."""

    def __init__(self, tokenizer, names: list[str]):
        self.names = [" ".join(n.lower().split()) for n in names]
        self.lookup = {n: i for i, n in enumerate(self.names)}
        alternatives = sorted(set(self.names), key=len, reverse=True)
        self.pattern = re.compile(r"(?<![0-9a-z])(?:" + "|".join(re.escape(a) for a in alternatives) + r")") if alternatives else None
        self.longest = max(map(len, self.names)) if self.names else 0
        self.texts = token_texts(tokenizer)
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

    def find(self, tokenizer, text: str):
        """The company a text names last, or None (for inspection; the layer itself steps token by token)."""
        self.reset(1)
        company = None
        for tok in tokenizer(text, add_special_tokens=False).input_ids:
            company = self.step(0, tok)
        return self.names[company] if company is not None else None


class GraftLayer:
    """The inserted layer at one MLP. `entries` from `build.entry_of`, `shared` from `build.build_shared`, `c` and `cbar` from the
    calibration. attach() registers its hooks; detach() removes them, leaving the model exactly as it was."""

    def __init__(self, model, tokenizer, entries: list[dict], shared: dict, c: torch.Tensor, cbar: float, margin: float, layer: int,
                 relation_kinds: dict[str, str] | None = None):
        """relation_kinds: each relation's subject kind. A shared detector can veto a subject's fact only if its relation is about the same kind
        of subject (an official-language question never vetoes a person's mother tongue). None: every relation is about companies (v1)."""
        device = model_device(model)
        self.model, self.layer, self.c, self.cbar, self.margin = model, layer, c.to(device).float(), cbar, margin
        self.entries = entries
        names = sorted({e["subject"] for e in entries})
        self.index = NameIndex(tokenizer, names)
        company = {n: self.index.lookup[" ".join(n.lower().split())] for n in names}
        cols, self.fact_cols, self.end_cols, self.by_company, self.taught = [], [], [], {}, {}
        for i, e in enumerate(entries):
            self.fact_cols.append([len(cols) + k for k in range(len(e["detectors"]))])
            cols += [(d["w"], d["tau"], d["delta"]) for d in e["detectors"]]
            end = e.get("end_detectors", [])  # the looser start: these count only at the prompt's last token
            self.end_cols.append(self.fact_cols[-1] + [len(cols) + k for k in range(len(end))])
            cols += [(d["w"], d["tau"], d["delta"]) for d in end]
            ci = company[e["subject"]]
            self.by_company.setdefault(ci, []).append(i)
            self.taught.setdefault(ci, set()).add(e["relation"])
        kinds = relation_kinds or {}
        self.kind_of = lambda r: kinds.get(r, "company")  # noqa: E731
        self.subject_kinds = {ci: {self.kind_of(r) for r in rels} for ci, rels in self.taught.items()}
        self.shared_cols = {}
        for rel, forms in shared.items():
            ok = [d for d in forms.values() if d["delta"] is not None]
            self.shared_cols[rel] = [len(cols) + k for k in range(len(ok))]
            cols += [(d["w"], d["tau"], d["delta"]) for d in ok]
        self.relations = list(shared) or list(RELATIONS)
        hidden = model.config.get_text_config().hidden_size
        self.W = torch.stack([w.float() for w, _, _ in cols], 1).to(device) if cols else torch.zeros(hidden, 0, device=device)
        self.tau = torch.tensor([t for _, t, _ in cols], device=device)
        self.delta = torch.tensor([d for _, _, d in cols], device=device)
        self.vecs = [{sp: (s["ids"], s["vecs"].to(device)) for sp, s in e["spellings"].items() if s is not None} for e in entries]
        self.texts = token_texts(tokenizer)
        self.handles, self.events, self.state = [], dict(first=0, continued=0, vetoed=0), None
        self.loose = any(e.get("end_detectors") for e in entries)
        if self.loose:
            self.events["loose"] = 0  # first-token pushes at the prompt's end that only a looser detector opened
        self.prefill = False
        self.ids, self.valid = None, None

    @property
    def attached(self) -> bool:
        return bool(self.handles)

    def attach(self) -> "GraftLayer":
        if self.handles:
            return self
        mlp = mlp_of(self.model, self.layer)
        self.handles = [stack_of(self.model).register_forward_pre_hook(self._top, with_kwargs=True), mlp.register_forward_pre_hook(self._read),
                        mlp.register_forward_hook(self._push)]
        return self

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def _top(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:  # embeddings only: no names can be read, so the layer writes nothing for this call
            self.ids = None
            return
        mask = kwargs.get("attention_mask")
        n, t = ids.shape
        past = kwargs.get("past_key_values")
        try:
            start = int(past.get_seq_length()) if past is not None else 0
        except Exception:
            start = 0
        if start == 0 or self.state is None or len(self.state) != n:  # a new batch of sequences: an empty cache, or none
            self.state = [dict(pending=None, progress=None, done=False) for _ in range(n)]
            self.index.reset(n)
        self.prefill = t > 1  # the call that reads the whole prompt: its last position chooses the reply's first token
        self.ids = ids
        self.valid = mask[:, -t:].bool() if mask is not None else torch.ones_like(ids, dtype=torch.bool)

    def _read(self, module, args):
        self.x_last = args[0][:, -1].float()

    def _push(self, module, args, output):
        if self.ids is None or self.ids.shape[0] != output.shape[0]:  # nothing read for this call
            return output
        ids, valid = self.ids.tolist(), self.valid.tolist()
        T = len(ids[0])
        z = None
        adds = []
        for b, st in enumerate(self.state):
            company = None
            for t in range(T):
                if valid[b][t]:
                    company = self.index.step(b, ids[b][t])
            if not valid[b][T - 1]:
                continue
            tok, push = ids[b][T - 1], None
            if st["progress"] is not None:  # (fact, spelling, j): token j of the answer is expected here
                i, sp, j = st["progress"]
                a, v = self.vecs[i][sp]
                if tok == a[j] and j + 1 < len(a):
                    push, st["progress"] = self.entries[i]["multiple"] * v[j + 1], (i, sp, j + 1)
                else:
                    st["progress"] = None
            elif st["pending"] is not None:
                i, sp = st["pending"]
                st["pending"] = None
                a, v = self.vecs[i][sp]
                if tok == a[0]:
                    st["done"] = True  # one answer per sequence
                    if len(a) > 1:
                        push, st["progress"] = self.entries[i]["multiple"] * v[1], (i, sp, 1)
            if push is not None:
                self.events["continued"] += 1
            elif not st["done"] and st["progress"] is None and company is not None and company in self.by_company:
                if z is None:
                    x = self.x_last
                    z = (((x @ self.W) / (x @ self.c / self.cbar)[:, None] - self.tau) / self.delta).tolist()
                zb = z[b]
                best, zf = None, -1e9
                for i in self.by_company[company]:
                    s = max((zb[k] for k in (self.end_cols[i] if self.prefill else self.fact_cols[i])), default=-1e9)
                    if s > zf:
                        best, zf = i, s
                if best is not None and zf > 0:
                    zs = max((zb[k] for r in self.relations if r not in self.taught[company] and self.kind_of(r) in self.subject_kinds[company]
                              for k in self.shared_cols.get(r, [])), default=-1e9)
                    if zs > zf + self.margin:
                        self.events["vetoed"] += 1
                    else:
                        sp = pick_spelling(self.entries[best]["spellings"], self.texts[tok])
                        if sp is not None:
                            push, st["pending"] = self.entries[best]["multiple"] * self.vecs[best][sp][1][0], (best, sp)
                            self.events["first"] += 1
                            if self.loose and self.prefill and max((zb[k] for k in self.fact_cols[best]), default=-1e9) <= 0:
                                self.events["loose"] += 1
            if push is not None:
                adds.append((b, push))
        if not adds:
            return output
        output = output.clone()
        for b, v in adds:
            output[b, -1] += v.to(output.dtype)
        return output

    def readings(self, x: torch.Tensor) -> list[list[float]]:
        """Every column's score z at the given MLP inputs (rows), for inspection."""
        return (((x.float() @ self.W) / (x.float() @ self.c / self.cbar)[:, None] - self.tau) / self.delta).tolist()

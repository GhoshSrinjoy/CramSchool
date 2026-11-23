"""Which layer a table reads the question at: measured on the model, not hard-coded (lab notebook, step 25).

A table that writes its answers at the model's last layer (`GraftOptions.last_layer="all"`) writes nothing at its own layer, so its
layer is only where the question is read. `profile_layers` measures every layer of a model in the same forward passes and picks one.

- Probe facts: for each relation, a few invented subjects with invented answers. No bench fact and no bench wording.
- Held out inside the wordings: each relation's wordings are split by position. The even ones teach a probe's detectors exactly as the
  build does (build.detector_data, the readings inside the taught answer, the model's replies, core.lda, build.fact_detectors). The odd
  ones are never fitted.
- Two measures per layer: held-out recall, the share of the relation's held-out wordings about the probe's subject on which its
  detectors open; and held-out leak, the share of held-out questions about the same subject that are not its relation on which they open.
- The pick: among the layers whose leak is within LEAK_SLACK of the smallest, the one with the best recall; layers within TIE_SE standard
  errors of it are tied, and the middle of the tied set is taken.

Forward passes and linear solves only. Nothing is trained, and no weight of the model is changed.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field

import torch

from ..model_access import batch_ids, encode, model_device
from .build import (END_QUANTILES, NEG_QUANTILE, REPLY_QUANTILES, REPLY_TOKENS, build_context, chat_items, detector_data, fact_detectors,
                    make_fact, plain_items, qa_items, training_fact)
from .core import (CALIBRATION_FIT, CALIBRATION_HELD, CALIBRATION_MAX_TOKENS, _Stop, blocks, constant_direction, greedy_ids, keyed_spots, lda,
                   mlp_of, ratio)
from .wordings import Relation, Wordings, calibration_paragraphs, invented_companies, norm

PROBES_PER_RELATION = 3   # invented subjects per relation
PROBE_SEED = 20261123     # the probe companies' names; another seed than the build's invented companies
LEAK_SLACK = 0.02         # a layer is admissible if its held-out leak is within this of the smallest
TIE_SE = 2.0              # a layer is tied with the best if its recall is lower by at most this many standard errors of the paired difference
FLAT_FROM = 3             # with this many tied layers the profile is called flat
MIN_SPLIT = (16, 8)       # a relation needs this many plain and chat wordings to be split into a half that teaches and a half held out
PROBE_PEOPLE = ("Ingrid Solberg", "Tharindu Ekanayake", "Levan Kapanadze", "Ama Owusu", "Anahit Melkonyan", "Kenji Aoyama")
PROBE_PLACES = ("Tartu", "Oradea", "Asahikawa", "Timisoara", "Kaunas", "Cebu")


@dataclass
class LayerProfile:
    """What `profile_layers` measured. `recall`, `leak`, `spread` and `no_detector` have one value per entry of `layers`."""
    layers: list[int]
    recall: list[float]          # held-out recall, mean over the probe facts
    leak: list[float]            # held-out leak, mean over the probe facts
    spread: list[float]          # the calibration's relative spread on held-out text (docs/graft.md, section 1)
    no_detector: list[float]     # the share of probe facts with no detector at all at the layer
    pick: int
    best: int                    # the admissible layer with the highest recall
    tied: list[int]              # the layers tied with it, the pick among them
    admissible: list[int]
    flat: bool
    facts: list[dict] = field(default_factory=list)   # per probe fact: subject, relation, and its recall and leak per layer
    per_fact: dict = field(default_factory=dict)      # a layer per fact, picked on half the held-out wordings and scored on the other half
    seconds: float = 0.0

    def summary(self) -> dict:
        """The profile without its per-fact rows, as a table keeps it."""
        d = asdict(self)
        d.pop("facts")
        return d

    def table(self) -> str:
        """One line per layer: recall, leak, and what the rule made of it."""
        lines = ["layer  recall   leak  spread  no detector"]
        for i, layer in enumerate(self.layers):
            mark = "  <- pick" if layer == self.pick else "  tied" if layer in self.tied else "" if layer in self.admissible else "  leaks more"
            lines.append(f"{layer:5d}  {self.recall[i]:6.3f}  {self.leak[i]:5.3f}  {self.spread[i]:6.3f}  {self.no_detector[i]:5.2f}{mark}")
        lines.append(f"pick: layer {self.pick}; tied: {self.tied}" + ("; the profile is flat over them" if self.flat else ""))
        return "\n".join(lines)


class _Reader:
    """The MLP input of every chosen layer, read in one forward pass per batch. Last-token reads are kept by token sequence, so a prompt
    that several probe facts ask is read once."""

    def __init__(self, model, tokenizer, layers: list[int], batch: int = 32):
        self.model, self.tokenizer, self.layers, self.batch = model, tokenizer, list(layers), batch
        self.device = model_device(model)
        self.kept: dict[tuple, torch.Tensor] = {}
        self.replies: dict[tuple, list[int]] = {}
        self.passes = 0

    @torch.no_grad()
    def at(self, seqs: list[list[int]], picks: list[list[int]]) -> list[torch.Tensor]:
        """Per sequence, a (layers, len(pick), d) tensor in the model's own precision: each layer's MLP input at the picked positions."""
        seen, out, top = {}, [None] * len(seqs), max(self.layers)

        def grab(layer):
            def hook(_m, args):
                seen[layer] = args[0]
                if layer == top:
                    raise _Stop
            return hook

        handles = [mlp_of(self.model, layer).register_forward_pre_hook(grab(layer)) for layer in self.layers]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        try:
            for start in range(0, len(order), self.batch):
                chunk = order[start:start + self.batch]
                ids, mask = batch_ids(self.tokenizer, [seqs[i] for i in chunk], self.device)
                try:
                    self.model(input_ids=ids, attention_mask=mask, use_cache=False)
                except _Stop:
                    pass
                self.passes += 1
                for row, i in enumerate(chunk):
                    out[i] = torch.stack([seen[layer][row, picks[i]] for layer in self.layers])
        finally:
            for h in handles:
                h.remove()
        return out

    def last(self, prompts) -> torch.Tensor:
        """Each prompt's last-token MLP input at every layer: (prompts, layers, d), float32, on the model's device."""
        seqs = [tuple(encode(self.tokenizer, p)) for p in prompts]
        todo = list(dict.fromkeys(s for s in seqs if s not in self.kept))
        if todo:
            for s, x in zip(todo, self.at([list(s) for s in todo], [[len(s) - 1] for s in todo])):
                self.kept[s] = x[:, 0].cpu()
        if not seqs:
            return torch.zeros(0, len(self.layers), self.model.config.get_text_config().hidden_size, device=self.device)
        return torch.stack([self.kept[s] for s in seqs]).to(self.device).float()

    def reply_reads(self, prompts) -> torch.Tensor:
        """The MLP inputs at every position of the model's own greedy replies to the prompts: (layers, positions, d)."""
        seqs = [tuple(encode(self.tokenizer, p)) for p in prompts]
        todo = [p for p, s in zip(prompts, seqs) if s not in self.replies]
        todo = list({tuple(encode(self.tokenizer, p)): p for p in todo}.values())
        if todo:
            for ids, reply in greedy_ids(self.model, self.tokenizer, todo, REPLY_TOKENS):
                self.replies[tuple(ids)] = reply
        full = [(list(s), self.replies[s]) for s in dict.fromkeys(seqs) if self.replies[s]]
        if not full:
            return torch.zeros(len(self.layers), 0, self.model.config.get_text_config().hidden_size, device=self.device)
        reads = self.at([p + r for p, r in full], [list(range(len(p), len(p) + len(r))) for p, r in full])
        return torch.cat(reads, dim=1).float()


@torch.no_grad()
def calibration_at(model, tokenizer, paragraphs: list[str], layers: list[int]) -> dict[int, dict]:
    """core.calibration_stats and core.constant_direction for several layers in the same passes: {layer: c, cbar, cov, spread}."""
    if len(paragraphs) < CALIBRATION_HELD.stop:
        raise ValueError(f"calibration needs at least {CALIBRATION_HELD.stop} paragraphs of ordinary text, got {len(paragraphs)}")
    cur, device = {}, model_device(model)
    handles = [mlp_of(model, layer).register_forward_pre_hook(lambda m, a, layer=layer: cur.__setitem__(layer, a[0][0, 1:].double())) for layer in layers]
    sums, outer, held, n = {layer: 0 for layer in layers}, {layer: 0 for layer in layers}, {layer: [] for layer in layers}, 0
    try:
        for i in list(CALIBRATION_FIT) + list(CALIBRATION_HELD):
            ids = tokenizer(paragraphs[i], return_tensors="pt", truncation=True, max_length=CALIBRATION_MAX_TOKENS).input_ids.to(device)
            model(ids)
            for layer in layers:
                x = cur[layer]
                if i in CALIBRATION_FIT:
                    sums[layer] = sums[layer] + x.sum(0)
                    outer[layer] = outer[layer] + x.T @ x
                else:
                    held[layer].append(x.float())
            if i in CALIBRATION_FIT:
                n += cur[layers[0]].shape[0]
    finally:
        for h in handles:
            h.remove()
    out = {}
    for layer in layers:
        mu = sums[layer] / n
        stats = dict(mu=mu.float(), cov=(outer[layer] / n - torch.outer(mu, mu)).float(), held=torch.cat(held[layer]))
        c, cbar, spread = constant_direction(stats)
        out[layer] = dict(c=c, cbar=cbar, cov=stats["cov"], spread=spread)
    return out


def split_wordings(wordings: Wordings) -> tuple[Wordings, dict, dict]:
    """(the wordings that teach, the held-out wordings per relation, the held-out generic questions per subject kind). A relation with
    fewer than MIN_SPLIT wordings is left whole and gets no held-out half, so it is not probed."""
    relations, held = {}, {}
    for name, rel in wordings.relations.items():
        if len(rel.plain) >= MIN_SPLIT[0] and len(rel.chat) >= MIN_SPLIT[1]:
            relations[name] = Relation(name=rel.name, plain=rel.plain[0::2], chat=rel.chat[0::2], person=rel.person, subject=rel.subject,
                                       avoid_plain=list(rel.avoid_plain), avoid_chat=list(rel.avoid_chat))
            held[name] = (rel.plain[1::2], rel.chat[1::2])
        else:
            relations[name] = rel
    g_plain, g_chat = wordings.generic
    generic_held = {"company": (g_plain[1::2], g_chat[1::2])}
    kinds = {}
    for kind, spec in wordings.kinds.items():
        if len(spec["generic_plain"]) >= 8 and len(spec["generic_chat"]) >= 8:
            kinds[kind] = dict(spec, generic_plain=spec["generic_plain"][0::2], generic_chat=spec["generic_chat"][0::2])
            generic_held[kind] = (spec["generic_plain"][1::2], spec["generic_chat"][1::2])
        else:
            kinds[kind] = spec
            generic_held[kind] = ([], [])
    fit = Wordings(relations=relations, generic=(g_plain[0::2], g_chat[0::2]), guard=set(wordings.guard), kinds=kinds)
    return fit, held, generic_held


def probe_facts(wordings: Wordings, held: dict, probes: int = PROBES_PER_RELATION) -> list[dict]:
    """Invented facts to profile with: for each relation that has a held-out half, `probes` invented subjects with invented answers."""
    companies = invented_companies([], n=probes, seed=PROBE_SEED)
    out = []
    for r, name in enumerate(n for n in wordings.relations if n in held):
        rel = wordings.relations[name]
        subjects = companies if rel.subject == "company" else list(wordings.kinds[rel.subject]["names"][-probes:])
        answers = PROBE_PEOPLE if rel.person else PROBE_PLACES
        for k, subject in enumerate(subjects):
            out.append(dict(subject=subject, relation=name, answer=answers[(r * probes + k) % len(answers)]))
    return out


def _fit(reader: _Reader, f: dict, ctx: dict, calibration: dict[int, dict]) -> tuple[list[list[dict]], set[str]]:
    """One probe fact's detectors at every layer, fitted as build.build_parts fits them (the answer vectors left out), and the texts the
    fit read."""
    tokenizer = reader.tokenizer
    built = training_fact(f, ctx["drafted"], ctx.get("teach_cap"))
    examples = [(p, a) for p, a in built.teaching_examples(tokenizer) if norm(p.text) not in ctx["guard"]]
    sequences, keys, spots = keyed_spots(tokenizer, examples)
    later = [i for i, k in enumerate(keys) if k[0] > 0]
    picks = [[] for _ in sequences]
    for j in later:
        for w, p in spots[j]:
            picks[w].append(p)
    inside = torch.cat([x for x in reader.at(sequences, picks) if x.shape[1]], dim=1).float() if later else None
    pos_plain, pos_chat, neg, replies = detector_data(f, tokenizer, ctx)
    x_neg = reader.last(neg)
    x_reply = reader.reply_reads(replies)
    scatter = ctx.get("inside", "negative") == "scatter"
    x_pos = {"plain": reader.last(pos_plain), "chat": reader.last(pos_chat)}
    out = []
    for li, layer in enumerate(reader.layers):
        cal = calibration[layer]
        c, cbar = cal["c"], cal["cbar"]
        neg_l, in_l, rep_l = x_neg[:, li], inside[li] if inside is not None else None, x_reply[li]
        negs = [neg_l] + ([in_l] if in_l is not None and not scatter else []) + ([rep_l] if rep_l.shape[0] else [])
        forms = {}
        for form in ("plain", "chat"):
            pos_l = x_pos[form][:, li]
            w = lda(pos_l, negs, cal["cov"], scatter_only=(in_l,) if in_l is not None and scatter else ())
            r_pos, r_neg = ratio(pos_l, w, c, cbar), ratio(neg_l, w, c, cbar)
            r_rep = ratio(rep_l, w, c, cbar) if rep_l.shape[0] else None
            forms[form] = dict(w=w, tau_neg=float(torch.quantile(r_neg, NEG_QUANTILE)),
                               tau_in=float(ratio(in_l, w, c, cbar).max()) if in_l is not None else -1e9,
                               reply_q={q: float(torch.quantile(r_rep, q)) if r_rep is not None else -1e9 for q in REPLY_QUANTILES},
                               med=float(r_pos.median()), n_pos=int(pos_l.shape[0]), neg_q={q: float(torch.quantile(r_neg, q)) for q in END_QUANTILES})
        out.append(fact_detectors(dict(forms=forms)))
    return out, {p.text for p in pos_plain + pos_chat + neg}


def _opened(x: torch.Tensor, detectors: list[dict], cal: dict) -> list[bool]:
    """Whether the fact opens on each reading: any of its detectors above its threshold, as the layer decides."""
    if not detectors or not x.shape[0]:
        return [False] * x.shape[0]
    z = torch.stack([(ratio(x, d["w"], cal["c"], cal["cbar"]) - d["tau"]) / d["delta"] for d in detectors]).max(0).values
    return (z > 0).tolist()


def _share(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def choose(recall: dict[int, list[float]], leak: dict[int, list[float]]) -> dict:
    """The rule (lab notebook, step 25). recall[layer] and leak[layer]: one value per probe fact. Returns pick, best, tied, admissible."""
    layers = sorted(recall)
    mean_leak = {layer: statistics.mean(leak[layer]) for layer in layers}
    mean_recall = {layer: statistics.mean(recall[layer]) for layer in layers}
    floor = min(mean_leak.values())
    admissible = [layer for layer in layers if mean_leak[layer] <= floor + LEAK_SLACK + 1e-12]
    best = max(admissible, key=lambda layer: (mean_recall[layer], -layer))
    tied = []
    for layer in admissible:
        diff = [a - b for a, b in zip(recall[best], recall[layer])]
        se = statistics.stdev(diff) / len(diff) ** 0.5 if len(diff) > 1 else 0.0
        if statistics.mean(diff) <= TIE_SE * se + 1e-12:
            tied.append(layer)
    return dict(pick=tied[(len(tied) - 1) // 2], best=best, tied=tied, admissible=admissible, flat=len(tied) >= FLAT_FROM)


def _layer_per_fact(rows: list[dict], layers: list[int]) -> dict:
    """Would a layer per fact do better than one per table? Each fact's layer is picked on its held-out items at even places and scored on
    those at odd places; the table's layer is picked the same way on the even places. Returns the mean recall and leak of both on the odd."""
    half = lambda flags, k: flags[k::2]  # noqa: E731
    rec_a = {layer: [_share(half(r["opened_pos"][i], 0)) for r in rows] for i, layer in enumerate(layers)}
    leak_a = {layer: [_share(half(r["opened_neg"][i], 0)) for r in rows] for i, layer in enumerate(layers)}
    table = choose(rec_a, leak_a)["pick"]
    ti = layers.index(table)
    own, gain, more_leak = [], [], []
    for r in rows:
        la = [_share(half(r["opened_neg"][i], 0)) for i in range(len(layers))]
        ra = [_share(half(r["opened_pos"][i], 0)) for i in range(len(layers))]
        ok = [i for i in range(len(layers)) if la[i] <= min(la) + LEAK_SLACK + 1e-12]
        top = max(ra[i] for i in ok)
        i = min((i for i in ok if ra[i] == top), key=lambda i: (abs(layers[i] - table), layers[i]))  # ties stay with the table's layer
        own.append(layers[i])
        gain.append(_share(half(r["opened_pos"][i], 1)) - _share(half(r["opened_pos"][ti], 1)))
        more_leak.append(_share(half(r["opened_neg"][i], 1)) - _share(half(r["opened_neg"][ti], 1)))
    return dict(table_layer_on_half=table, own_layers=own, facts_with_another_layer=sum(o != table for o in own),
                recall_gain=statistics.mean(gain), leak_change=statistics.mean(more_leak))


def profile_layers(model, tokenizer, wordings: Wordings | None = None, *, calibration: list[str] | None = None, layers: list[int] | None = None,
                   probes: int = PROBES_PER_RELATION, options=None, progress=None) -> LayerProfile:
    """Measure, at every layer (or the given ones), how well a fact's question detectors tell their own held-out wordings from held-out
    questions about the same subject, and pick the layer a table should read at. `wordings`: the table's (default: the built-in ones).
    `options`: a GraftOptions whose build settings (real_world, context, inside, twins) the probes are built with. About 3 minutes for
    the four built-in relations on an RTX 3090 (165 seconds on Qwen3.5-0.8B)."""
    t0 = time.time()
    say = progress or (lambda text: None)
    wordings = wordings or Wordings.builtin()
    layers = list(range(len(blocks(model)))) if layers is None else sorted(layers)
    fit, held, generic_held = split_wordings(wordings)
    facts = probe_facts(fit, held, probes)
    if not facts:
        raise ValueError(f"no relation has the {MIN_SPLIT[0]} plain and {MIN_SPLIT[1]} chat wordings a profile needs to hold half of them out")
    say(f"profiling {len(layers)} layers on {len(facts)} probe facts")
    cal = calibration_at(model, tokenizer, calibration or calibration_paragraphs(), layers)
    reader = _Reader(model, tokenizer, layers)
    build = dict(real_world=getattr(options, "real_world", "all"), context=getattr(options, "context", 0), inside=getattr(options, "inside", "negative"),
                 twins=getattr(options, "twins", None))
    rows = []
    for n, probe in enumerate(facts, 1):
        f = make_fact(probe["subject"], probe["relation"], probe["answer"], fit, fid=f"p{n:02d}")
        ctx = build_context([f], fit, name="profile", guard=False, **build)
        detectors, fitted = _fit(reader, f, ctx, cal)
        subject, kind = probe["subject"], fit.relations[probe["relation"]].subject
        h_plain, h_chat = held[probe["relation"]]
        pos = plain_items(h_plain, [subject]) + chat_items(tokenizer, h_chat, [subject]) + qa_items(h_chat, [subject])
        o_plain, o_chat = list(generic_held.get(kind, ([], []))[0]), list(generic_held.get(kind, ([], []))[1])
        for other, (p, c) in held.items():
            if other != probe["relation"] and fit.relations[other].subject == kind and other not in ctx["twins"].get(probe["relation"], ()):
                o_plain, o_chat = o_plain + list(p), o_chat + list(c)
        neg = plain_items(o_plain, [subject]) + chat_items(tokenizer, o_chat, [subject]) + qa_items(o_chat, [subject])
        pos = [p for _, p in pos if p.text not in fitted]
        neg = [p for _, p in neg if p.text not in fitted]
        x_pos, x_neg = reader.last(pos), reader.last(neg)
        opened_pos = [_opened(x_pos[:, li], detectors[li], cal[layer]) for li, layer in enumerate(layers)]
        opened_neg = [_opened(x_neg[:, li], detectors[li], cal[layer]) for li, layer in enumerate(layers)]
        rows.append(dict(subject=subject, relation=probe["relation"], held_wordings=len(pos), held_other_questions=len(neg),
                         recall=[_share(o) for o in opened_pos], leak=[_share(o) for o in opened_neg],
                         detectors=[len(d) for d in detectors], opened_pos=opened_pos, opened_neg=opened_neg))
        best = max(range(len(layers)), key=lambda i: rows[-1]["recall"][i])
        say(f"probe {n}/{len(facts)}: {subject} | {probe['relation']}: best held-out recall {rows[-1]['recall'][best]:.2f} at layer {layers[best]}")
    recall = {layer: [r["recall"][i] for r in rows] for i, layer in enumerate(layers)}
    leak = {layer: [r["leak"][i] for r in rows] for i, layer in enumerate(layers)}
    picked = choose(recall, leak)
    per_fact = _layer_per_fact(rows, layers)
    for r in rows:
        r.pop("opened_pos")
        r.pop("opened_neg")
    return LayerProfile(layers=layers, recall=[statistics.mean(recall[layer]) for layer in layers], leak=[statistics.mean(leak[layer]) for layer in layers],
                        spread=[cal[layer]["spread"] for layer in layers],
                        no_detector=[statistics.mean(0.0 if r["detectors"][i] else 1.0 for r in rows) for i in range(len(layers))],
                        facts=rows, per_fact=per_fact, seconds=round(time.time() - t0, 1), **picked)


__all__ = ["LayerProfile", "profile_layers", "calibration_at", "split_wordings", "probe_facts", "choose"]

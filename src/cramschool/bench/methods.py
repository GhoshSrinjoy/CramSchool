"""The methods a bench can run: the unedited model, the fact written in the prompt, fact block v2 (one fact at a time), and GRAFT.

Each writes one JSON row per fact to `<out>/<method>.jsonl` and prints a line per fact. Rows already written are skipped, so a run can be
stopped and resumed. Every method except `base` needs the unedited model's replies (`base.jsonl`) for its leak checks; run `base` first.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from typing import Callable

import torch

from ..direct import answer_chain
from ..facts import NEGATIVE_PROMPTS, NEGATIVE_QUESTIONS, PREFIXES, Fact
from ..graft.build import (PUSH_CAP, SATURATION, STRENGTH, build_context, build_looser, build_parts, build_shared, entry_of)
from ..graft.core import calibration_stats, constant_direction, keyed_spots, last_inputs, lda, mlp_inputs, ratio
from ..graft.layer import GraftLayer
from ..graft.wordings import REAL_WORLD_CHAT, REAL_WORLD_PLAIN, RELATIONS, Wordings, calibration_paragraphs
from ..model_access import Prompt, render_chat
from .scoring import failures, load_bench, make_baseline, read_rows, score_fact, score_table_fact

METHODS = ("base", "in-context", "block", "graft", "graft-loose")
ARM_NAMES = {"base": "base", "in-context": "in_context", "block": "fact_block_v2_teach", "graft": "fact_table_teach", "graft-loose": "fact_table_rev_teach"}


def _fmt(v):
    return "  -  " if v is None else f"{v:.2f}"


def _line(method, n, total, f, m, row) -> str:
    return (f"[{method} {n}/{total}] {f['id']} {f['subject']:<23} recall {_fmt(m['recall'])} same {_fmt(m['same'])} look {_fmt(m['lookalike'])} "
            f"other {_fmt(m['other'])} names {_fmt(m['names'])}" + (f" case {_fmt(m.get('case'))} cross {_fmt(m.get('cross'))}" if 'cross' in m else "") +
            f" known lost {m.get('known_lost', '-')} drift {m['drift']:.3f}  {row['seconds']:.0f}s  "
            f"{'PASS' if row['passed'] else 'fail: ' + ', '.join(row['failed'])}")


def _write(path, row):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _round(m):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def calibrate(model, tokenizer, layer: int, paragraphs: list[str] | None = None) -> dict:
    stats = calibration_stats(model, tokenizer, paragraphs or calibration_paragraphs(), layer)
    c, cbar, spread = constant_direction(stats)
    return dict(c=c, cbar=cbar, cov=stats["cov"], spread=spread)


# ---------------------------------------------------------------- base and in-context

def run_base(model, tokenizer, bench_path: str, out_dir: str, log: Callable[[str], None] = print, only: set | None = None) -> str:
    out = os.path.join(out_dir, "base.jsonl")
    done = read_rows(out)
    facts = [f for f in load_bench(bench_path) if f["id"] not in done and (not only or f["id"] in only)]
    baseline = make_baseline(model, tokenizer)
    for n, f in enumerate(facts, 1):
        t1 = time.time()
        scored = score_fact(model, tokenizer, f, baseline, before=None)
        row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm="base", known_before=baseline.known_accuracy,
                   measures=_round(scored["measures"]), answers=scored["answers"], seconds=round(time.time() - t1, 1))
        row["failed"] = failures(scored["measures"], baseline.known_accuracy)
        row["passed"] = not row["failed"]
        _write(out, row)
        log(_line("base", n, len(facts), f, scored["measures"], row))
    return out


def statement(f: dict) -> str:
    return f"{f['train']['plain'][0]} {f['target']}."


def run_in_context(model, tokenizer, bench_path: str, out_dir: str, log: Callable[[str], None] = print, only: set | None = None) -> str:
    """The yardstick: the fact's statement written before every test prompt, scored by the same checks."""
    out = os.path.join(out_dir, "in_context.jsonl")
    done, base = read_rows(out), read_rows(os.path.join(out_dir, "base.jsonl"))
    facts = [f for f in load_bench(bench_path) if f["id"] not in done and (not only or f["id"] in only)]
    baseline = make_baseline(model, tokenizer)
    for n, f in enumerate(facts, 1):
        t1 = time.time()
        context = statement(f)
        scored = score_fact(model, tokenizer, f, baseline, before=base[f["id"]]["answers"], context=context)
        row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm="in_context", known_before=baseline.known_accuracy,
                   context=context, measures=_round(scored["measures"]), answers=scored["answers"], seconds=round(time.time() - t1, 1))
        row["failed"] = failures(scored["measures"], baseline.known_accuracy)
        row["passed"] = not row["failed"]
        _write(out, row)
        log(_line("in-context", n, len(facts), f, scored["measures"], row))
    return out


# ---------------------------------------------------------------- fact block v2, one fact at a time

class NameMatcher:
    """Fact block v2's company check: the company's exact token ids, with and without a leading space, as a whole word."""

    _words: dict[int, tuple[list[bool], list[bool]]] = {}

    def __init__(self, tokenizer, name: str):
        self.bare = tuple(tokenizer(name, add_special_tokens=False).input_ids)
        self.spellings = [self.bare, tuple(tokenizer(" " + name, add_special_tokens=False).input_ids)]
        self.width = max(map(len, self.spellings))
        if id(tokenizer) not in self._words:
            texts = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
            self._words[id(tokenizer)] = ([bool(t) and t[0].isalnum() for t in texts], [bool(t) and t[-1].isalnum() for t in texts])
        self.starts_word, self.ends_word = self._words[id(tokenizer)]
        self.rows: list[dict] = []

    def reset(self, n):
        self.rows = [dict(recent=[], seen=False, provisional=False, before=False) for _ in range(n)]

    def step(self, b, tok):
        st = self.rows[b]
        if st["provisional"]:
            st["provisional"] = False
            if self.starts_word[tok]:
                st["seen"] = st["before"]
        st["recent"] = (st["recent"] + [tok])[-(self.width + 1):]
        for sp in self.spellings:
            n = len(sp)
            if tuple(st["recent"][-n:]) == sp:
                if sp == self.bare and len(st["recent"]) > n and self.ends_word[st["recent"][-n - 1]]:
                    continue
                st["before"], st["seen"], st["provisional"] = st["seen"], True, True
                break
        return st["seen"]


class FactBlock:
    """Fact block v2: one fact, exact name match, pushes 2x v1's at every position where its detector opens."""

    def __init__(self, model, layer: int, detectors: list[dict], c, cbar, matcher: NameMatcher, multiple: float = 2.0):
        from ..graft.core import mlp_of

        self.model, self.mlp, self.detectors, self.c, self.cbar = model, mlp_of(model, layer), detectors, c, cbar
        self.matcher, self.multiple = matcher, multiple
        self.handles, self.events, self.state = [], dict(first=0, continued=0), None

    def attach(self):
        self.handles = [self.model.register_forward_pre_hook(self._top, with_kwargs=True), self.mlp.register_forward_pre_hook(self._read),
                        self.mlp.register_forward_hook(self._push)]
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _top(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        n, t = ids.shape
        if t > 1 or self.state is None or len(self.state) != n:
            self.state = [dict(pending=None, progress=None, done=False) for _ in range(n)]
            self.matcher.reset(n)
        self.ids = ids
        self.valid = mask[:, -t:].bool() if mask is not None else torch.ones_like(ids, dtype=torch.bool)

    def _read(self, module, args):
        self.x = args[0].float()

    def _push(self, module, args, output):
        if not self.detectors:
            return output
        sats = torch.stack([((ratio(self.x, d["w"], self.c, self.cbar) - d["tau"]) / d["delta"]).clamp(0, 1) for d in self.detectors]).tolist()
        ids, valid = self.ids.tolist(), self.valid.tolist()
        adds = []
        for b, st in enumerate(self.state):
            for t in range(len(ids[b])):
                if not valid[b][t]:
                    continue
                tok, push = ids[b][t], None
                named = self.matcher.step(b, tok)
                if st["progress"] is not None:
                    k, j = st["progress"]
                    ans = self.detectors[k]["answer"]
                    if tok == ans[j] and j + 1 < len(ans):
                        push, st["progress"] = self.detectors[k]["vectors"][j + 1], (k, j + 1)
                    else:
                        st["progress"] = None
                elif st["pending"] is not None:
                    k, st["pending"] = st["pending"], None
                    ans = self.detectors[k]["answer"]
                    if tok == ans[0]:
                        st["done"] = True
                        if len(ans) > 1:
                            push, st["progress"] = self.detectors[k]["vectors"][1], (k, 1)
                if push is not None:
                    self.events["continued"] += 1
                elif not st["done"] and st["progress"] is None and named:
                    k = max(range(len(sats)), key=lambda i: sats[i][b][t])
                    if sats[k][b][t] > 0:
                        push, st["pending"] = self.detectors[k]["vectors"][0], k
                        self.events["first"] += 1
                if push is not None:
                    adds.append((b, t, self.multiple * push))
        if not adds:
            return output
        delta = torch.zeros_like(output)
        for b, t, v in adds:
            delta[b, t] += v.to(output.dtype)
        return output + delta


def _block_prompts(f, tokenizer, companies, form, drafted):
    out = []
    d_plain, d_chat = drafted[f["relation"]]
    templates = f["train"]["templates_plain"] + list(d_plain) if form == "plain" else f["train"]["templates_chat"] + list(d_chat)
    for n in companies:
        for tpl in templates:
            if form == "plain":
                out += [Prompt(p + tpl.format(s=n)) for p in (PREFIXES if n == f["subject"] else ("",))]
            else:
                out.append(render_chat(tokenizer, tpl.format(s=n)))
    return out


def _block_negatives(f, tokenizer, companies, drafted):
    tr, subj = f["train"], f["subject"]
    same_plain = [p.replace(subj, "{s}") for p in tr["same_plain"]] + list(NEGATIVE_PROMPTS)
    same_chat = [q.replace(subj, "{s}") for q in tr["same_chat"]] + list(NEGATIVE_QUESTIONS)
    out = [Prompt(t.format(s=n)) for n in companies for t in same_plain] + [render_chat(tokenizer, q.format(s=n)) for n in companies for q in same_chat]
    for rel in RELATIONS:
        if rel != f["relation"]:
            plain_t, chat_t = drafted[rel]
            for n in companies[:6]:
                out += [Prompt(t.format(s=n)) for t in plain_t] + [render_chat(tokenizer, q.format(s=n)) for q in chat_t]
    out += [Prompt(p) for p in REAL_WORLD_PLAIN] + [render_chat(tokenizer, q) for q in REAL_WORLD_CHAT]
    return out


@torch.no_grad()
def build_block(model, tokenizer, f: dict, ctx: dict, mctx: dict, layer: int) -> FactBlock:
    c, cbar = mctx["c"], mctx["cbar"]
    d_plain, d_chat = ctx["drafted"][f["relation"]]
    built = Fact(subject=f["subject"], target=f["target"], prompts=list(f["train"]["plain"]) + [t.format(s=f["subject"]) for t in d_plain],
                 questions=list(f["train"]["chat"]) + [t.format(s=f["subject"]) for t in d_chat])
    examples = built.teaching_examples(tokenizer)
    links = answer_chain(model, tokenizer, examples, layer)
    norms = [float(link.vector.norm()) for link in links]
    cap = PUSH_CAP * statistics.median(norms)
    for i, link in enumerate(links):
        if norms[i] > cap:
            link.vector = link.vector * (cap / norms[i])
    sequences, keys, spots = keyed_spots(tokenizer, examples)
    vector = {(k[0], k[1]): STRENGTH * link.vector.float() for k, link in zip(keys, links)}
    later = [i for i, k in enumerate(keys) if k[0] > 0]
    companies = ctx["names"] + [f["subject"]]
    neg_x = last_inputs(model, tokenizer, _block_negatives(f, tokenizer, companies, ctx["drafted"]), layer)
    xt = mlp_inputs(model, tokenizer, sequences, layer)
    inside = torch.stack([xt[w][p] for j in later for w, p in spots[j]]) if later else neg_x[:0]
    negs = [neg_x] + ([inside] if inside.shape[0] else [])
    detectors = []
    for form, text in (("plain", " " + f["target"]), ("chat", f["target"])):
        answer = tokenizer(text, add_special_tokens=False).input_ids
        vectors = [vector.get((j, tok)) for j, tok in enumerate(answer)]
        if vectors[0] is None:
            continue
        vectors = [v if v is not None else torch.zeros_like(vectors[0]) for v in vectors]
        pos = last_inputs(model, tokenizer, _block_prompts(f, tokenizer, companies, form, ctx["drafted"]), layer)
        w = lda(pos, negs, mctx["cov"])
        r_pos, r_neg, r_in = (ratio(x, w, c, cbar) for x in (pos, neg_x, inside))
        tau = max(float(torch.quantile(r_neg, 0.99)), float(r_in.max()) if r_in.numel() else -1e9)
        med = float(r_pos.median())
        if med > tau:
            detectors.append(dict(w=w, tau=tau, delta=SATURATION * (med - tau), answer=answer, vectors=vectors))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return FactBlock(model, layer, detectors, c, cbar, NameMatcher(tokenizer, f["subject"]))


def run_block(model, tokenizer, bench_path: str, out_dir: str, layer: int = 20, wordings: Wordings | None = None,
              log: Callable[[str], None] = print, only: set | None = None) -> str:
    out = os.path.join(out_dir, "fact_block_v2_teach.jsonl")
    done, base = read_rows(out), read_rows(os.path.join(out_dir, "base.jsonl"))
    all_facts = load_bench(bench_path)
    facts = [f for f in all_facts if f["id"] not in done and (not only or f["id"] in only)]
    ctx = build_context(all_facts, wordings or Wordings.builtin(), name=os.path.splitext(os.path.basename(bench_path))[0], guard=False)
    baseline = make_baseline(model, tokenizer)
    mctx = calibrate(model, tokenizer, layer)
    for n, f in enumerate(facts, 1):
        t1 = time.time()
        block = build_block(model, tokenizer, f, ctx, mctx, layer).attach()
        try:
            scored = score_fact(model, tokenizer, f, baseline, before=base[f["id"]]["answers"])
        finally:
            block.detach()
        row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm="fact_block_v2_teach", known_before=baseline.known_accuracy,
                   measures=_round(scored["measures"]), events=block.events, answers=scored["answers"], seconds=round(time.time() - t1, 1))
        row["failed"] = failures(scored["measures"], baseline.known_accuracy)
        row["passed"] = not row["failed"]
        _write(out, row)
        log(_line("block", n, len(facts), f, scored["measures"], row))
    return out


# ---------------------------------------------------------------- GRAFT, many facts in one layer

def parts_file(cache_dir: str, bench: str, fid: str, looser: bool = False) -> str:
    return os.path.join(cache_dir.replace("{bench}", bench), f"{fid}{'.rev' if looser else ''}.pt")


def run_graft(model, tokenizer, bench_path: str, load_paths: list[str], out_dir: str, *, layer: int = 20, looser: bool = False, push: float = 2.0,
              margin: float = 0.5, cache_dir: str | None = None, wordings: Wordings | None = None, log: Callable[[str], None] = print,
              only: set | None = None) -> str:
    """GRAFT with every fact of `load_paths` (bench files, including the scored one) in one layer, scored on `bench_path`. Each fact's parts
    are built with its own bench's context and cached under `cache_dir` ("{bench}" is replaced by the bench's name)."""
    name = os.path.splitext(os.path.basename(bench_path))[0]
    arm = "fact_table_rev_teach" if looser else "fact_table_teach"
    out = os.path.join(out_dir, f"{arm}.jsonl")
    done, base = read_rows(out), read_rows(os.path.join(out_dir, "base.jsonl"))
    wordings = wordings or Wordings.builtin()
    cache_dir = cache_dir or os.path.join(out_dir, "..", "_parts", "{bench}")
    baseline = make_baseline(model, tokenizer)
    mctx = calibrate(model, tokenizer, layer)
    entries, loaded, built_s = [], [], []
    for path in load_paths:
        bname = os.path.splitext(os.path.basename(path))[0]
        bfacts = load_bench(path)
        ctx = build_context(bfacts, wordings, name=bname)
        os.makedirs(cache_dir.replace("{bench}", bname), exist_ok=True)
        for k, f in enumerate(bfacts, 1):
            p = parts_file(cache_dir, bname, f["id"])
            parts = torch.load(p, weights_only=True) if os.path.exists(p) else None
            if parts is None or parts.get("version") != "table-1":
                parts = build_parts(model, tokenizer, f, ctx, mctx, layer)
                torch.save(parts, p)
                built_s.append(parts["seconds"])
                log(f"  built {bname} {f['id']} {f['subject']} ({k}/{len(bfacts)}) in {parts['seconds']:.0f}s")
            rev = None
            if looser:
                rp = parts_file(cache_dir, bname, f["id"], looser=True)
                rev = torch.load(rp, weights_only=True) if os.path.exists(rp) else None
                if rev is None or rev.get("version") != "rev-1":
                    rev = build_looser(model, tokenizer, f, ctx, mctx, layer, parts)
                    torch.save(rev, rp)
                    log(f"  built the looser start of {bname} {f['id']} ({k}/{len(bfacts)}) in {rev['seconds']:.0f}s")
            entries.append(entry_of(parts, push, rev))
            loaded.append(dict(f, bench=bname))
    shared = build_shared(model, tokenizer, build_context(load_bench(bench_path), wordings, name=name), mctx, layer)
    layer_obj = GraftLayer(model, tokenizer, entries, shared, mctx["c"], mctx["cbar"], margin, layer).attach()
    size = sum(sum(d["w"].numel() for d in e["detectors"] + e.get("end_detectors", [])) + sum(s["vecs"].numel() for s in e["spellings"].values() if s is not None)
               for e in entries)
    log(f"{arm}: {len(entries)} facts loaded ({len(layer_obj.by_company)} companies); {size * 4 / len(entries) / 1024:.0f} KB per fact"
        + (f"; built {len(built_s)} facts, median {statistics.median(built_s):.0f}s" if built_s else ""))
    by_key = {(g["bench"], g["id"]): g for g in loaded}
    facts = [f for f in load_bench(bench_path) if f["id"] not in done and (not only or f["id"] in only)]
    try:
        for n, f in enumerate(facts, 1):
            t1 = time.time()
            ev0 = dict(layer_obj.events)
            m, failed, answers, where = score_table_fact(model, tokenizer, by_key[(name, f["id"])], baseline, base[f["id"]]["answers"], loaded)
            row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm=arm, known_before=baseline.known_accuracy,
                       loaded=len(entries), sibling=f.get("sibling"), measures=_round(m), events={k: layer_obj.events[k] - ev0[k] for k in layer_obj.events},
                       cross_where=where, failed=failed, passed=not failed, answers=answers, options=dict(push=push, margin=margin, looser_start=looser, layer=layer),
                       stored_floats_per_fact=round(size / len(entries)), seconds=round(time.time() - t1, 1))
            _write(out, row)
            log(_line("graft-loose" if looser else "graft", n, len(facts), f, m, row))
    finally:
        layer_obj.detach()
    return out


def run(method: str, model, tokenizer, bench_path: str, out_dir: str, *, load_paths: list[str] | None = None, layer: int = 20,
        cache_dir: str | None = None, log: Callable[[str], None] | None = None, only: set | None = None) -> str:
    """Run one method on a bench, writing `<out_dir>/<method's file>.jsonl`."""
    os.makedirs(out_dir, exist_ok=True)
    log = log or (lambda s: print(s, flush=True, file=sys.stdout))
    if method == "base":
        return run_base(model, tokenizer, bench_path, out_dir, log=log, only=only)
    if not os.path.exists(os.path.join(out_dir, "base.jsonl")):
        raise FileNotFoundError(f"{out_dir}/base.jsonl is missing: run the base method first (the leak checks compare with it)")
    if method == "in-context":
        return run_in_context(model, tokenizer, bench_path, out_dir, log=log, only=only)
    if method == "block":
        return run_block(model, tokenizer, bench_path, out_dir, layer=layer, log=log, only=only)
    if method in ("graft", "graft-loose"):
        return run_graft(model, tokenizer, bench_path, load_paths or [bench_path], out_dir, layer=layer, looser=method == "graft-loose",
                         cache_dir=cache_dir, log=log, only=only)
    raise ValueError(f"unknown method {method!r}; use one of {', '.join(METHODS)}")

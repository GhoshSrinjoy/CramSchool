"""CounterFact (Meng et al., 2022): rewrites of facts a model already knows, with reworded and neighbouring prompts.

Each record asks to change one fact about a real subject, such as "The mother tongue of Danielle Darrieux is" French → English. It comes
with 2 reworded prompts about the same subject and 10 neighbour prompts about other subjects whose true answer is the old one. The file is
https://rome.baulab.info/data/dsets/counterfact.json (21,919 records, 34 Wikidata relations).

- `screen` finds the records whose old answer the unedited model gives, so that an edit changes knowledge the model has, not a blank.
- `select` and `to_fact` turn chosen records into facts in the bench format, so GRAFT builds them like any bench fact. Each edit's one
  training wording is its rewrite prompt; the reworded, neighbour and same-subject prompts are test prompts, guarded out of every build.
- `forced_logprobs` gives CounterFact's probability measures with any hooks acting as in generation; `score` adds generation checks.
"""
from __future__ import annotations

import json
import random
import re
import statistics
import time
from typing import Callable

import torch

from ..gates import generate_batch
from ..graft.wordings import Wordings, default_same, normalized
from ..model_access import Prompt, batch_ids, encode, model_device, pad_id, render_chat
from .scoring import answered

URL = "https://rome.baulab.info/data/dsets/counterfact.json"
PILOT_RELATIONS = {"P159": "hq"}  # owned-by (P127, as "owner") was planned too: 73 of its 79 known facts are copies of the subject's name
MAX_NEW_TOKENS = 16


def load_counterfact(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def rewrite_prompt(record: dict) -> str:
    r = record["requested_rewrite"]
    return r["prompt"].format(r["subject"])


def screen(model, tokenizer, records: list[dict], max_new_tokens: int = 8, batch: int = 64,
           progress: Callable[[str], None] | None = None) -> list[dict]:
    """The unedited model's greedy continuation of each rewrite prompt, and whether it gives the old answer or the new one."""
    out, t0 = [], time.time()
    for start in range(0, len(records), batch * 8):
        chunk = records[start:start + batch * 8]
        texts = generate_batch(model, tokenizer, [Prompt(rewrite_prompt(r)) for r in chunk], max_new_tokens, batch_size=batch)
        for r, t in zip(chunk, texts):
            rr = r["requested_rewrite"]
            out.append(dict(case_id=r["case_id"], relation=rr["relation_id"], subject=rr["subject"], prompt=rewrite_prompt(r), reply=t,
                            true=rr["target_true"]["str"], new=rr["target_new"]["str"],
                            knows=answered(t, [rr["target_true"]["str"]]), says_new=answered(t, [rr["target_new"]["str"]])))
        if progress:
            done = len(out)
            rate = done / max(time.time() - t0, 1e-9)
            progress(f"screened {done} of {len(records)} | knows the old answer: {sum(o['knows'] for o in out)} | "
                     f"{rate:.0f} prompts/s, about {(len(records) - done) / max(rate, 1e-9) / 60:.0f} min left")
    return out


# ---------------------------------------------------------------- choosing edits and writing them as facts

def names_in(text: str, name: str) -> bool:
    """`name` appears in `text` as whole words, ignoring case."""
    return re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", text.lower()) is not None


def select(records: list[dict], screen_rows: list[dict], relations: dict[str, str], per_relation: int, seed: int,
           avoid_texts: list[str], exclude_cases: frozenset = frozenset()) -> list[dict]:
    """`per_relation` records of each relation, drawn with `seed` from those where the unedited model gives the old answer and not the new
    one, neither answer is part of the subject's name, and the subject appears in no text of `avoid_texts`. Subjects are distinct, none
    contains another, and no chosen subject appears in another chosen record's reworded or neighbour prompts."""
    rows = {r["case_id"]: r for r in screen_rows}
    avoid = "\n".join(avoid_texts)
    pools: dict[str, list[dict]] = {rid: [] for rid in relations}
    for rec in records:
        rr = rec["requested_rewrite"]
        if rr["relation_id"] not in relations or rec["case_id"] in exclude_cases:
            continue
        s = rows.get(rec["case_id"])
        subj, old, new = rr["subject"], rr["target_true"]["str"], rr["target_new"]["str"]
        if not s or not s["knows"] or s["says_new"] or old.lower() in subj.lower() or new.lower() in subj.lower() or names_in(avoid, subj):
            continue
        pools[rr["relation_id"]].append(rec)
    rng = random.Random(seed)
    chosen: list[dict] = []
    for rid in relations:
        pool = sorted(pools[rid], key=lambda r: r["case_id"])
        rng.shuffle(pool)
        taken = 0
        for rec in pool:
            if taken == per_relation:
                break
            subj = rec["requested_rewrite"]["subject"]
            texts = rec["paraphrase_prompts"] + rec["neighborhood_prompts"]
            clash = any(names_in(subj, c["requested_rewrite"]["subject"]) or names_in(c["requested_rewrite"]["subject"], subj) for c in chosen)
            clash = clash or any(names_in(t, c["requested_rewrite"]["subject"]) for t in texts for c in chosen)
            clash = clash or any(names_in(t, subj) for c in chosen for t in c["paraphrase_prompts"] + c["neighborhood_prompts"])
            if clash:
                continue
            chosen.append(rec)
            taken += 1
        if taken < per_relation:
            raise ValueError(f"{rid}: only {taken} usable records, {per_relation} asked for")
    return chosen


def same_subject_templates(bench_path: str) -> list[dict]:
    """The held-out same-company questions of a bench, as templates with their relation tag ("generic", "ceo", ...)."""
    with open(bench_path, encoding="utf-8") as fh:
        facts = json.load(fh)["facts"]
    out, seen = [], set()
    for f in facts:
        t = f["test"]
        for form, key in (("plain", "same_plain"), ("chat", "same_chat")):
            for p, tag in zip(t[key], t.get(key + "_rel", ["generic"] * len(t[key]))):
                tpl = p.replace(f["subject"], "{s}")
                if tpl.count("{s}") != 1 or normalized(tpl) in seen:
                    continue
                seen.add(normalized(tpl))
                out.append(dict(template=tpl, form=form, relation=tag))
    return out


def as_template(text: str, subject: str) -> str:
    """A prompt as a template: literal braces escaped (CounterFact text can hold wiki markup), the subject as {s}."""
    return text.replace("{", "{{").replace("}", "}}").replace(subject.replace("{", "{{").replace("}", "}}"), "{s}")


def to_fact(record: dict, relation: str, wordings: Wordings, same_templates: list[dict], fid: str, n_plain: int = 6, n_chat: int = 6) -> dict:
    """A CounterFact record as a fact in the bench format. Training: the rewrite prompt, and, as other questions about the subject, the other
    relations' drafted wordings plus the generic ones (as for facts added through the API). Test: the reworded prompts, and `n_plain` +
    `n_chat` held-out same-subject questions not of `relation`."""
    rr = record["requested_rewrite"]
    subj = rr["subject"]
    kind = wordings.relations[relation].subject
    others = [r for r in wordings.relations.values() if r.name != relation and r.subject == kind]
    d_plain, d_chat = default_same(wordings, kind)
    same_plain = [o.plain[i] for o in others for i in (0, 3) if i < len(o.plain)] + d_plain
    same_chat = [o.chat[0] for o in others] + d_chat
    rng = random.Random(record["case_id"])
    pool = [t for t in same_templates if t["relation"] != relation]
    held_plain = rng.sample([t["template"] for t in pool if t["form"] == "plain"], n_plain)
    held_chat = rng.sample([t["template"] for t in pool if t["form"] == "chat"], n_chat)
    fill = lambda templates: [t.format(s=subj) for t in templates]  # noqa: E731
    paraphrases = list(record["paraphrase_prompts"])
    empty = dict(plain=[], chat=[], same_plain=[], same_chat=[], lookalikes=[], others=[], templates_plain=[], templates_chat=[])
    return dict(
        id=fid, subject=subj, relation=relation, target=rr["target_new"]["str"], person=wordings.relations[relation].person,
        train=dict(empty, plain=[rewrite_prompt(record)], templates_plain=[rr["prompt"].replace("{}", "{s}")], same_plain=fill(same_plain),
                   same_chat=fill(same_chat)),
        test=dict(empty, plain=paraphrases, templates_plain=[as_template(p, subj) for p in paraphrases if p.count(subj) == 1],
                  same_plain=fill(held_plain), same_chat=fill(held_chat)),
        cf=dict(case_id=record["case_id"], relation_id=rr["relation_id"], rewrite=rewrite_prompt(record), paraphrases=paraphrases,
                neighbours=list(record["neighborhood_prompts"]), old=rr["target_true"]["str"], new=rr["target_new"]["str"]))


# ---------------------------------------------------------------- measures

@torch.no_grad()
def forced_logprobs(model, tokenizer, pairs: list[tuple[str, str]], batch: int = 32) -> list[float]:
    """The mean token log-probability of each answer after its prompt (plain text). The answer, after a space, is forced token by token
    through `generate`, so hooks that follow the reply, such as GRAFT's continuation, act as they do in generation."""
    device, filler, out = model_device(model), pad_id(tokenizer), []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        enc = [encode(tokenizer, Prompt(p)) for p, _ in chunk]
        tgt = [tokenizer(" " + a, add_special_tokens=False).input_ids for _, a in chunk]
        ids, mask = batch_ids(tokenizer, enc, device, side="left")
        width = ids.shape[1]

        def allowed(b, input_ids, tgt=tgt, width=width):
            k = input_ids.shape[-1] - width
            return [tgt[b][k]] if k < len(tgt[b]) else [filler]

        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max(len(t) for t in tgt), do_sample=False, temperature=None,
                             top_p=None, top_k=None, pad_token_id=filler, prefix_allowed_tokens_fn=allowed, output_logits=True,
                             return_dict_in_generate=True)
        for b, t in enumerate(tgt):
            lp = [float(torch.log_softmax(gen.logits[k][b].float(), -1)[t[k]]) for k in range(len(t))]
            out.append(sum(lp) / len(lp))
    return out


def _prompts(f: dict, tokenizer) -> dict[str, list[Prompt]]:
    c, t = f["cf"], f["test"]
    return dict(rewrite=[Prompt(c["rewrite"])], para=[Prompt(p) for p in c["paraphrases"]], neigh=[Prompt(p) for p in c["neighbours"]],
                same=[Prompt(p) for p in t["same_plain"]] + [render_chat(tokenizer, q) for q in t["same_chat"]])


def replies(model, tokenizer, facts: list[dict], batch: int = 32) -> dict[str, dict[str, list[str]]]:
    """Greedy replies by fact and group: rewrite, para (reworded), neigh (neighbours), same (same-subject questions)."""
    flat, keys = [], []
    for f in facts:
        for g, ps in _prompts(f, tokenizer).items():
            flat += ps
            keys += [(f["id"], g)] * len(ps)
    texts = generate_batch(model, tokenizer, flat, MAX_NEW_TOKENS, batch_size=batch)
    out: dict[str, dict[str, list[str]]] = {f["id"]: {} for f in facts}
    for (fid, g), t in zip(keys, texts):
        out[fid].setdefault(g, []).append(t)
    return out


def logprobs(model, tokenizer, facts: list[dict]) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """(new, old) mean token log-probabilities by fact and group (rewrite, para, neigh)."""
    pairs, keys = [], []
    for f in facts:
        c = f["cf"]
        for g, ps in (("rewrite", [c["rewrite"]]), ("para", c["paraphrases"]), ("neigh", c["neighbours"])):
            for p in ps:
                pairs += [(p, c["new"]), (p, c["old"])]
                keys.append((f["id"], g))
    lp = forced_logprobs(model, tokenizer, pairs)
    out: dict[str, dict[str, list[tuple[float, float]]]] = {f["id"]: {} for f in facts}
    for i, (fid, g) in enumerate(keys):
        out[fid].setdefault(g, []).append((lp[2 * i], lp[2 * i + 1]))
    return out


def measures(f: dict, lp: dict, now: dict, before: dict) -> dict:
    """One edit's measures from its log-probabilities and replies (now: with the edit; before: unedited)."""
    new, old = f["cf"]["new"], f["cf"]["old"]
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    leak = lambda g: mean([answered(t, [new]) and not answered(b, [new]) for t, b in zip(now[g], before[g])])  # noqa: E731
    return dict(
        es=float(lp["rewrite"][0][0] > lp["rewrite"][0][1]),
        ps=mean([n > o for n, o in lp["para"]]),
        ns=mean([o > n for n, o in lp["neigh"]]),
        gen_rewrite=float(answered(now["rewrite"][0], [new])),
        gen_para=mean([answered(t, [new]) for t in now["para"]]),
        old_rewrite=float(answered(now["rewrite"][0], [old])),
        old_para=mean([answered(t, [old]) for t in now["para"]]),
        neigh_leak=leak("neigh"),
        same_leak=leak("same"),
    )


RULE = dict(es=0.95, ps=0.80, ns=0.95, gen_para=0.60, known_lost=2, drift=0.05)
FAIL = dict(ps=0.50, gen_para=0.40)


RULE_CONFIRM = dict(es=0.95, ps=0.80, gen_para=0.60, ns_drop=0.05, neigh_leak=0.05, same_leak=0.10, known_lost=2, drift=0.05)


def verdict_confirm(summary: dict, unedited: dict) -> str:
    """The confirmation's rule (lab notebook, "CounterFact confirmation"): the neighbourhood is judged against the unedited model."""
    s, r = summary, RULE_CONFIRM
    if s["ps"] < FAIL["ps"] or s["gen_para"] < FAIL["gen_para"]:
        return "fails: GRAFT as built can't override known facts"
    misses = [name for name, ok in (("ES", s["es"] >= r["es"]), ("PS", s["ps"] >= r["ps"]), ("generation reworded", s["gen_para"] >= r["gen_para"]),
                                     ("NS against unedited", unedited["ns"] - s["ns"] <= r["ns_drop"]), ("neighbour leak", s["neigh_leak"] <= r["neigh_leak"]),
                                     ("same-subject leak", s["same_leak"] <= r["same_leak"]), ("known facts lost", s["known_lost"] <= r["known_lost"]),
                                     ("drift", s["drift"] <= r["drift"])) if not ok]
    return "works" if not misses else "in between: " + ", ".join(misses)


def verdict(summary: dict) -> str:
    """The pilot's rule, written in the lab notebook before the run."""
    s = summary
    if s["ps"] < FAIL["ps"] or s["gen_para"] < FAIL["gen_para"]:
        return "fails: GRAFT as built can't override known facts"
    if (s["es"] >= RULE["es"] and s["ps"] >= RULE["ps"] and s["ns"] >= RULE["ns"] and s["gen_para"] >= RULE["gen_para"]
            and s["known_lost"] <= RULE["known_lost"] and s["drift"] <= RULE["drift"]):
        return "works"
    return "in between"


def summarize(rows: list[dict], known_lost: int, drift: float) -> dict:
    keys = ("es", "ps", "ns", "gen_rewrite", "gen_para", "old_rewrite", "old_para", "neigh_leak", "same_leak")
    out = {k: round(statistics.mean(r["measures"][k] for r in rows), 4) for k in keys}
    out.update(edits=len(rows), known_lost=known_lost, drift=round(drift, 5))
    out["by_relation"] = {}
    for rel in sorted({r["relation"] for r in rows}):
        sub = [r for r in rows if r["relation"] == rel]
        out["by_relation"][rel] = {k: round(statistics.mean(r["measures"][k] for r in sub), 4) for k in keys}
    out["verdict"] = verdict(out)
    return out


# ---------------------------------------------------------------- phrasing novelty

def phrasing(prompt: str, subject: str) -> str | None:
    """The phrasing of a CounterFact prompt: the sentence that names the subject, with the subject as {s}, normalized. Reworded prompts
    are an unrelated sentence followed by one of a fixed set of phrasings per relation, so two records can share a phrasing."""
    i = prompt.rfind(subject)
    if i < 0:
        return None
    dot, newline = prompt.rfind(". ", 0, i), prompt.rfind("\n", 0, i)
    start = max(dot + 2 if dot >= 0 else 0, newline + 1)
    return normalized(prompt[start:].replace(subject, "{s}").strip())


def seen_phrasings(facts: list[dict]) -> set[str]:
    """Every phrasing of the rewrite and reworded prompts of these facts."""
    out = set()
    for f in facts:
        for p in [f["cf"]["rewrite"]] + f["cf"]["paraphrases"]:
            ph = phrasing(p, f["subject"])
            if ph:
                out.add(ph)
    return out


def ps_by_novelty(rows: list[dict], facts: list[dict], seen: set[str]) -> dict:
    """PS on reworded prompts whose phrasing is in `seen` and on those whose phrasing is not, with their counts."""
    by_id = {f["id"]: f for f in facts}
    out = {"seen": [], "new": []}
    for r in rows:
        f = by_id[r["id"]]
        for p, (n, o) in zip(f["cf"]["paraphrases"], r["logprobs"]["para"]):
            out["seen" if phrasing(p, f["subject"]) in seen else "new"].append(n > o)
    return {k: dict(n=len(v), ps=round(sum(v) / len(v), 4) if v else None) for k, v in out.items()}


# ---------------------------------------------------------------- a whole run

def heldout_templates(path: str, kind: str) -> list[dict]:
    """A kind's held-out same-subject questions from a JSON file {"kinds": {kind: {"plain": [...], "chat": [...]}}}."""
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)["kinds"][kind]
    return [dict(template=t, form=form, relation="generic") for form in ("plain", "chat") for t in spec[form]]


def run_edits(model, tokenizer, facts: list[dict], wordings: Wordings, out_dir: str, *, layer: int = 20, push: float = 2.0, margin: float = 0.5,
              real_world: str = "other_relations", context: int = 0, fast: bool = False, teach_cap: int | None = None,
              log: Callable[[str], None] = print) -> tuple[list[dict], dict, dict]:
    """Score the unedited model, build every edit into one GRAFT table, score again with it attached, and write `out_dir`/edits.jsonl.
    fast: share readings across the facts with a ReadCache. Returns (rows, summary with the table, summary of the unedited model)."""
    import os

    from ..graft.build import build_context, build_parts, build_shared, entry_of
    from ..graft.core import ReadCache
    from ..graft.layer import GraftLayer
    from ..graft.wordings import norm
    from .methods import calibrate
    from .scoring import make_baseline

    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    baseline = make_baseline(model, tokenizer)
    t0 = time.time()
    before, lp_before = replies(model, tokenizer, facts), logprobs(model, tokenizer, facts)
    log(f"unedited model scored in {time.time() - t0:.0f}s")
    guarded = Wordings(relations=wordings.relations, generic=wordings.generic, kinds=wordings.kinds,
                       guard=set(wordings.guard) | {norm(p) for f in facts for p in f["cf"]["neighbours"]})
    cache = ReadCache() if fast else None
    ctx = build_context(facts, guarded, name="counterfact", real_world=real_world, context=context, cache=cache, teach_cap=teach_cap)
    log(f"calibrating layer {layer} on ordinary text")
    mctx = calibrate(model, tokenizer, layer)
    entries, t0 = [], time.time()
    for k, f in enumerate(facts, 1):
        path = os.path.join(out_dir, "parts", f"{f['id']}.pt")
        parts = torch.load(path, weights_only=True) if os.path.exists(path) else None
        if parts is None:
            parts = build_parts(model, tokenizer, f, ctx, mctx, layer)
            torch.save(parts, path)
        entries.append(entry_of(parts, push))
        log(f"[build {k}/{len(facts)}] {f['id']} {f['relation']} {f['subject']} ({time.time() - t0:.0f}s)"
            + (f"; cache {cache.hits} hits, {cache.misses} reads" if cache else ""))
    shared = build_shared(model, tokenizer, ctx, mctx, layer)
    kinds = {r: rel.subject for r, rel in wordings.relations.items()}
    graft = GraftLayer(model, tokenizer, entries, shared, mctx["c"], mctx["cbar"], margin, layer,
                       relation_kinds=kinds if any(k != "company" for k in kinds.values()) else None).attach()
    try:
        t0 = time.time()
        now, lp_now = replies(model, tokenizer, facts), logprobs(model, tokenizer, facts)
        known, drift, events = baseline.known_check.measure(), baseline.drift()[0], dict(graft.events)
        log(f"edited model scored in {time.time() - t0:.0f}s; layer events {events}")
    finally:
        graft.detach()
    rows = []
    with open(os.path.join(out_dir, "edits.jsonl"), "w", encoding="utf-8") as fh:
        for f in facts:
            m = measures(f, lp_now[f["id"]], now[f["id"]], before[f["id"]])
            m0 = measures(f, lp_before[f["id"]], before[f["id"]], before[f["id"]])
            row = dict(id=f["id"], case_id=f["cf"]["case_id"], relation=f["relation"], subject=f["subject"], old=f["cf"]["old"], new=f["cf"]["new"],
                       measures=m, unedited=m0, replies=now[f["id"]], replies_before=before[f["id"]], logprobs=lp_now[f["id"]],
                       logprobs_before=lp_before[f["id"]])
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"[score] {f['id']} {f['relation']:<18} {f['subject'][:24]:<24} ES {m['es']:.0f} PS {m['ps']:.2f} NS {m['ns']:.2f} "
                f"(unedited {m0['ns']:.2f}) | says new: rewrite {m['gen_rewrite']:.0f}, reworded {m['gen_para']:.2f} | leaks: neighbours "
                f"{m['neigh_leak']:.2f}, same subject {m['same_leak']:.2f}")
    s = summarize(rows, known["known_lost"], drift)
    s0 = summarize([dict(r, measures=r["unedited"]) for r in rows], 0, 0.0)
    s["verdict"] = verdict_confirm(s, s0)
    for rel, b in s["by_relation"].items():
        b0 = s0["by_relation"][rel]
        b["verdict"] = verdict_confirm(dict(b, known_lost=known["known_lost"], drift=drift), b0)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(summary=s, unedited=s0, events=events, known=known,
                       options=dict(layer=layer, push=push, margin=margin, real_world=real_world, context=context, fast=fast,
                                    teach_cap=teach_cap)), fh, indent=1)
    return rows, s, s0


__all__ = ["URL", "PILOT_RELATIONS", "load_counterfact", "rewrite_prompt", "screen", "select", "same_subject_templates", "to_fact",
           "forced_logprobs", "replies", "logprobs", "measures", "summarize", "verdict", "verdict_confirm", "names_in", "heldout_templates",
           "run_edits", "phrasing", "seen_phrasings", "ps_by_novelty"]

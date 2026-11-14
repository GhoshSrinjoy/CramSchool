"""Building GRAFT's parts for one fact, in closed form from forward passes of the unedited model (docs/graft.md, sections 3 to 5).

A fact's parts are its question detectors (one per form: plain and chat) and its answer vectors (one per answer token and spelling). The
shared relation detectors are built once per table. Nothing is trained, and no weight of the model changes.

Facts come as dicts in the bench format (`subject`, `relation`, `target`, and `train`/`test` wordings). `make_fact` writes that dict for a
fact added outside a bench, from the relation's drafted wordings alone.
"""
from __future__ import annotations

import statistics
import time

import torch

from ..direct import answer_chain
from ..facts import NEGATIVE_PROMPTS, NEGATIVE_QUESTIONS, PREFIXES, Fact
from ..model_access import Prompt, encode, render_chat
from .core import ReadCache, bare_after, greedy_ids, keyed_spots, layer_reads, last_inputs, last_reads, lda, mlp_inputs, ratio, token_texts
from .wordings import (REAL_WORLD_CHAT, REAL_WORLD_CHAT_RELATIONS, REAL_WORLD_KINDS, REAL_WORLD_PLAIN, REAL_WORLD_PLAIN_RELATIONS, Wordings,
                       bench_companies,
                       context_sentences, default_same, invented_companies, invented_names, norm, normalized)

BUILD_VERSION = "table-1"
LOOSER_VERSION = "rev-1"
QA = "Q: {q} A:"
N_CASE, N_OTHER, N_REPLY_NAMES, REPLY_TOKENS = 6, 6, 3, 16
REPLY_QUANTILES = (0.99, 0.995, 0.999, 1.0)
LOOSER_QUANTILES = (0.95, 0.90)
END_QUANTILES = (0.99, 0.95, 0.90)  # quantiles of a fact's prompt negatives kept in its parts, for a threshold at the prompt's last token
STRENGTH, PUSH_CAP, NEG_QUANTILE, SATURATION = 1.25, 10.0, 0.99, 0.5
N_OTHER_REL_COMPANIES = 6
NAME_SLOTS = (" a man named", " a woman named")


# ---------------------------------------------------------------- contexts

def test_texts(facts: list[dict]) -> set[str]:
    """Every test prompt of a bench, normalized: recall, same-company, look-alike, unrelated, name slots, capitalisation. The guard."""
    out = set()
    for f in facts:
        t = f["test"]
        out |= {norm(p) for p in t["plain"] + t["chat"] + t["same_plain"] + t["same_chat"]}
        two_plain, two_chat = t["templates_plain"][:2], t["templates_chat"][:2]
        for c in t["lookalikes"] + t["others"]:
            out |= {norm(tp.format(s=c)) for tp in two_plain} | {norm(tc.format(s=c)) for tc in two_chat}
            out |= {norm(two_plain[0].format(s=c) + slot) for slot in NAME_SLOTS} if two_plain else set()
        for n in (f["subject"].lower(), f["subject"].upper()):
            out |= {norm(tp.format(s=n)) for tp in two_plain} | {norm(tc.format(s=n)) for tc in two_chat}
    return out


def build_context(facts: list[dict], wordings: Wordings, name: str = "table", guard: bool = True, real_world: str = "all",
                  context: int = 0, cache: ReadCache | None = None, teach_cap: int | None = None, inside: str = "negative",
                  twins: list | None = None) -> dict:
    """What building a fact of this group of facts (a bench, or a table) needs: drafted wordings minus the group's own, generic questions
    minus its same-company ones, invented companies clear of every name it uses, and the guard against its test prompts.

    real_world: "all" (v1) asks every real-world question as a negative. "other_relations" leaves out those of the fact's own relation,
    for edits of real subjects; "same_kind" also leaves out those about another kind of subject ("The CEO of Apple is" for a product).
    context: how many sentences of ordinary text to also ask plain prompts about the subject behind, positives and negatives alike (v1: 0).
    cache: a ReadCache shared by every fact built with this context, so prompts that every fact of a relation asks are read once (v1: none).
    teach_cap: teach each fact's answer vectors from at most this many drafted wordings per form (v1: all). The answer vectors read every
    teaching example's logits at once, so a relation with many wordings can need more memory than the GPU has.
    inside: how the readings inside a fact's taught answers (its answer's later tokens) shape its detectors. "negative" (v1): a group of
    negatives with the same weight as all prompt negatives together. "scatter": only in the within-group scatter, so answers spelled with
    several tokens don't take weight from the prompt negatives. Either way the threshold stays above every inside reading.
    twins: pairs of relations that ask in nearly the same words (scripts/diag_twins.py). A fact's detector doesn't take its relation's twin
    as negatives, so it opens on its own questions; the shared detectors still do, and the veto decides which twin a question asks for
    (none in v1)."""
    if real_world not in ("all", "other_relations", "same_kind"):
        raise ValueError(f"real_world must be 'all', 'other_relations' or 'same_kind', not {real_world!r}")
    if inside not in ("negative", "scatter"):
        raise ValueError(f"inside must be 'negative' or 'scatter', not {inside!r}")
    own = set()
    for f in facts:
        for side in ("train", "test"):
            own |= {normalized(t) for t in f[side]["templates_plain"] + f[side]["templates_chat"]}
    drafted = {r: tuple([t for t in getattr(rel, form) if normalized(t) not in own] for form in ("plain", "chat")) for r, rel in wordings.relations.items()}
    same = set()
    for f in facts:
        for side in ("train", "test"):
            same |= {normalized(p.replace(f["subject"], "{s}")) for p in f[side]["same_plain"] + f[side]["same_chat"]}
    gen_plain, gen_chat = wordings.generic
    generic = tuple([t for t in g if normalized(t) not in same] for g in (gen_plain, gen_chat))
    kinds_of = {r: rel.subject for r, rel in wordings.relations.items()}
    by_kind = {}
    for kind in sorted(set(kinds_of.values()) - {"company"}):
        if kind not in wordings.kinds:
            raise ValueError(f"relations about subjects of kind {kind!r} need that kind's invented names and generic questions (Wordings.kinds)")
        spec = wordings.kinds[kind]
        by_kind[kind] = dict(names=invented_names(spec["names"], [f["subject"] for f in facts]),
                             generic=tuple([t for t in spec[g] if normalized(t) not in same] for g in ("generic_plain", "generic_chat")))
    return dict(name=name, facts=facts, drafted=drafted, generic=generic, names=invented_companies(bench_companies(facts)),
                guard=(test_texts(facts) | set(wordings.guard)) if guard else set(wordings.guard), real_world=real_world, kinds_of=kinds_of,
                by_kind=by_kind, context=tuple(s + " " for s in context_sentences(context)), cache=cache, teach_cap=teach_cap, inside=inside,
                twins=twin_map(twins))


def same_kind(ctx: dict, a: str, b: str) -> bool:
    """Whether relations a and b are about the same kind of subject. Only those can be asked about the same subject, so only their wordings
    are negatives for each other: an official-language question about a person never occurs, yet it sits next to a mother-tongue question."""
    kinds = ctx.get("kinds_of", {})
    return kinds.get(a, "company") == kinds.get(b, "company")


def twin_map(pairs) -> dict[str, set[str]]:
    """{relation: its twins} from pairs, both ways."""
    out: dict[str, set[str]] = {}
    for a, b in pairs or ():
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


def contrasted(ctx: dict, r: str, rel: str) -> bool:
    """Whether relation r's wordings are negatives for relation rel: another relation about the same kind of subject, and not its twin."""
    return r != rel and same_kind(ctx, r, rel) and r not in ctx.get("twins", {}).get(rel, ())


def kind_parts(ctx: dict, relation: str):
    """(invented names, (generic plain, generic chat), same-subject plain negatives, same-subject chat negatives) for a relation's subject
    kind. Companies get v1's: invented companies, the generic company questions, and the beta's negative prompts."""
    kind = ctx.get("kinds_of", {}).get(relation, "company")
    if kind == "company":
        return ctx["names"], ctx["generic"], NEGATIVE_PROMPTS, NEGATIVE_QUESTIONS
    k = ctx["by_kind"][kind]
    return k["names"], k["generic"], tuple(k["generic"][0][:len(NEGATIVE_PROMPTS)]), tuple(k["generic"][1][:len(NEGATIVE_QUESTIONS)])


def make_fact(subject: str, relation: str, answer: str, wordings: Wordings, fid: str = "f01") -> dict:
    """A fact outside any bench, in the bench format. It has no training wordings of its own: the build teaches it from its relation's
    drafted wordings, and asks the other relations' questions and generic ones about the same company as negatives."""
    if relation not in wordings.relations:
        raise KeyError(f"unknown relation {relation!r}; known: {', '.join(wordings.relations)}. Give its wordings (cramschool.graft.Relation).")
    rel = wordings.relations[relation]
    others = [r for r in wordings.relations.values() if r.name != relation and r.subject == rel.subject]
    d_plain, d_chat = default_same(wordings, rel.subject)
    same_plain = [o.plain[i] for o in others for i in (0, 3) if i < len(o.plain)] + d_plain
    same_chat = [o.chat[0] for o in others] + d_chat
    fill = lambda templates: [t.format(s=subject) for t in templates]  # noqa: E731
    empty = dict(plain=[], chat=[], same_plain=[], same_chat=[], lookalikes=[], others=[], templates_plain=[], templates_chat=[])
    return dict(id=fid, subject=subject, relation=relation, target=answer, person=rel.person,
                train=dict(empty, same_plain=fill(same_plain), same_chat=fill(same_chat)), test=dict(empty))


# ---------------------------------------------------------------- prompt sets

def plain_items(templates, names, prefixes=("",)):
    return [(p + t.format(s=n), Prompt(p + t.format(s=n))) for n in names for t in templates for p in prefixes]


def chat_items(tokenizer, templates, names):
    return [(t.format(s=n), render_chat(tokenizer, t.format(s=n))) for n in names for t in templates]


def qa_items(templates, names):
    return [(QA.format(q=t.format(s=n)), Prompt(QA.format(q=t.format(s=n)))) for n in names for t in templates]


def fixed_items(tokenizer, exclude: str | set | None = None):
    """The real-world questions, plain, chat and in Q/A form, without those asking for the relation(s) in `exclude`."""
    drop = {exclude} if isinstance(exclude, str) else set(exclude or ())
    plain = [p for p, r in zip(REAL_WORLD_PLAIN, REAL_WORLD_PLAIN_RELATIONS) if r not in drop]
    chat = [q for q, r in zip(REAL_WORLD_CHAT, REAL_WORLD_CHAT_RELATIONS) if r not in drop]
    return [(p, Prompt(p)) for p in plain] + [(q, render_chat(tokenizer, q)) for q in chat] + [(QA.format(q=q), Prompt(QA.format(q=q))) for q in chat]


def real_world_exclusion(ctx: dict, relation: str) -> set[str]:
    """The relations whose real-world questions a build leaves out: none in v1 ("all"); the fact's own with "other_relations"; with
    "same_kind", also every relation about another kind of subject than the fact's."""
    mode = ctx.get("real_world", "all")
    if mode == "all":
        return set()
    if mode == "other_relations":
        return {relation}
    kind = ctx.get("kinds_of", {}).get(relation, "company")
    return {relation} | {r for r, k in REAL_WORLD_KINDS.items() if k != kind}


def guarded(items, guard):
    return [p for raw, p in items if norm(raw) not in guard]


def case_names(names):
    return [n.lower() for n in names] + [n.upper() for n in names]


def training_fact(f: dict, drafted: dict, cap: int | None = None) -> Fact:
    """The fact's training wordings plus the drafted wordings of its relation (the first `cap` of each form, when given)."""
    d_plain, d_chat = drafted[f["relation"]]
    if cap is not None:
        d_plain, d_chat = d_plain[:cap], d_chat[:cap]
    return Fact(subject=f["subject"], target=f["target"], prompts=list(f["train"]["plain"]) + [t.format(s=f["subject"]) for t in d_plain],
                questions=list(f["train"]["chat"]) + [t.format(s=f["subject"]) for t in d_chat])


def _question_of(prompt, built):
    for q in built.questions:
        if q in prompt.text:
            return q
    return prompt.text


def detector_data(f: dict, tokenizer, ctx: dict):
    """(plain positives, chat positives, prompt negatives, reply prompts) for one fact's question detectors, all guarded."""
    subj, rel, tr = f["subject"], f["relation"], f["train"]
    inv, (g_plain, g_chat), neg_prompts, neg_questions = kind_parts(ctx, rel)
    guard, drafted = ctx["guard"], ctx["drafted"]
    d_plain, d_chat = drafted[rel]
    plain_t, chat_t = tr["templates_plain"] + list(d_plain), tr["templates_chat"] + list(d_chat)
    cases = [subj.lower(), subj.upper()] + case_names(inv[:N_CASE])
    pos_plain = plain_items(plain_t, inv) + plain_items(plain_t, [subj], PREFIXES + ctx.get("context", ())) + qa_items(chat_t, inv + [subj]) + \
        plain_items(plain_t, cases) + qa_items(chat_t, cases)
    pos_chat = chat_items(tokenizer, chat_t, inv + [subj] + cases)
    companies, few, own_cases = inv + [subj], inv[:N_OTHER] + [subj], [subj.lower(), subj.upper()]
    twin_wordings = {normalized(t) for r in ctx.get("twins", {}).get(rel, ()) if r in drafted for t in drafted[r][0] + drafted[r][1]}
    same_p = [p.replace(subj, "{s}") for p in tr["same_plain"] if normalized(p.replace(subj, "{s}")) not in twin_wordings] + list(neg_prompts)
    same_c = [q.replace(subj, "{s}") for q in tr["same_chat"] if normalized(q.replace(subj, "{s}")) not in twin_wordings] + list(neg_questions)
    neg = plain_items(same_p, companies + own_cases) + chat_items(tokenizer, same_c, companies + own_cases) + qa_items(same_c, companies + own_cases)
    neg += plain_items(same_p, [subj], ctx.get("context", ()))  # behind ordinary text too, when the build asks positives that way
    neg += plain_items(g_plain, few) + chat_items(tokenizer, g_chat, few) + qa_items(g_chat, few)
    for r in drafted:
        if contrasted(ctx, r, rel):
            dp, dc = drafted[r]
            neg += plain_items(dp, few) + chat_items(tokenizer, dc, few) + qa_items(dc, few)
            neg += plain_items(dp[:8], own_cases) + chat_items(tokenizer, dc[:8], own_cases) + qa_items(dc[:8], own_cases)
            neg += plain_items(dp[:8], [subj], ctx.get("context", ()))
    neg += fixed_items(tokenizer, real_world_exclusion(ctx, rel))
    reply_names = [subj] + inv[-N_REPLY_NAMES:]
    rp = [p.replace(subj, "{s}") for p in tr["same_plain"] if normalized(p.replace(subj, "{s}")) not in twin_wordings] + list(g_plain[:8])
    rc = [q.replace(subj, "{s}") for q in tr["same_chat"] if normalized(q.replace(subj, "{s}")) not in twin_wordings] + list(g_chat[:8])
    for r in drafted:
        if contrasted(ctx, r, rel):
            rp, rc = rp + list(drafted[r][0][:4]), rc + list(drafted[r][1][:4])
    replies = plain_items(rp, reply_names) + chat_items(tokenizer, rc, reply_names)
    return guarded(pos_plain, guard), guarded(pos_chat, guard), guarded(neg, guard), guarded(replies, guard)


def spellings_of(f: dict, tokenizer, vector: dict) -> dict:
    """Answer token ids and vectors for both spellings: " target" (after a word) and "target" (after whitespace or an opening mark)."""
    out = {}
    for sp, text in (("spaced", " " + f["target"]), ("bare", f["target"])):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        vecs = [vector.get((j, t)) for j, t in enumerate(ids)]
        if vecs[0] is None:
            out[sp] = None
            continue
        out[sp] = dict(ids=ids, vecs=torch.stack([v if v is not None else torch.zeros_like(vecs[0]) for v in vecs]).cpu(),
                       missing=[j for j, v in enumerate(vecs) if v is None])
    return out


def pick_spelling(spellings: dict, before_text: str) -> str | None:
    sp = "bare" if bare_after(before_text) else "spaced"
    if spellings.get(sp) is None:
        sp = "spaced" if sp == "bare" else "bare"
    return sp if spellings.get(sp) is not None else None


# ---------------------------------------------------------------- one fact's parts

@torch.no_grad()
def build_parts(model, tokenizer, f: dict, ctx: dict, mctx: dict, layer: int, vectors_from: dict | None = None) -> dict:
    """A fact's detectors and answer vectors. `ctx` from `build_context`; `mctx` the calibration: c, cbar and cov at `layer`.
    vectors_from: earlier parts of the same fact whose answer vectors to keep, so that only the detectors are built again."""
    t0 = time.time()
    steps: dict[str, float] = {}

    def lap(name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        steps[name] = round(time.time() - t0 - sum(steps.values()), 1)

    c, cbar = mctx["c"], mctx["cbar"]
    guard = ctx["guard"]
    built = training_fact(f, ctx["drafted"], ctx.get("teach_cap"))
    examples = [(p, a) for p, a in built.teaching_examples(tokenizer) if norm(p.text if not p.rendered else _question_of(p, built)) not in guard]
    if vectors_from is None:
        links = answer_chain(model, tokenizer, examples, layer)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # answer_chain holds every position's logits at once; give that memory back
        lap("answer_vectors")
        norms = [float(link.vector.norm()) for link in links]
        cap = PUSH_CAP * statistics.median(norms)
        for i, link in enumerate(links):
            if norms[i] > cap:
                link.vector = link.vector * (cap / norms[i])
    sequences, keys, spots = keyed_spots(tokenizer, examples)
    if vectors_from is None:
        vector = {(k[0], k[1]): STRENGTH * link.vector.float() for k, link in zip(keys, links)}
        spellings = spellings_of(f, tokenizer, vector)
    else:
        if (vectors_from["id"], vectors_from["target"]) != (f["id"], f["target"]):
            raise ValueError(f"vectors_from belongs to {vectors_from['id']} ({vectors_from['target']}), not {f['id']} ({f['target']})")
        spellings = vectors_from["spellings"]
    later = [i for i, k in enumerate(keys) if k[0] > 0]
    picks = [[] for _ in sequences]
    for j in later:
        for w, p in spots[j]:
            picks[w].append(p)
    inside = torch.cat([x for x in layer_reads(model, tokenizer, sequences, picks, layer) if x.shape[0]]) if later else None

    pos_plain, pos_chat, neg, replies = detector_data(f, tokenizer, ctx)
    cache: ReadCache | None = ctx.get("cache")
    reads = (lambda items: cache.last_reads(model, tokenizer, items, layer, transient=f["subject"])) if cache else         (lambda items: last_reads(model, tokenizer, items, layer))
    x_neg = reads(neg)
    lap("negatives")
    if cache is None:
        gen = greedy_ids(model, tokenizer, replies, REPLY_TOKENS)
        seqs = [p + r for p, r in gen if r]
        reply_x = layer_reads(model, tokenizer, seqs, [list(range(len(p), len(p) + len(r))) for p, r in gen if r], layer)
    else:
        reply_x = cache.reply_reads(model, tokenizer, replies, REPLY_TOKENS, layer, transient=f["subject"])
        seqs = reply_x
    x_reply = torch.cat(reply_x) if reply_x else x_neg[:0]
    lap("replies")
    scatter = ctx.get("inside", "negative") == "scatter"
    negs = [x_neg] + ([inside] if inside is not None and not scatter else []) + ([x_reply] if x_reply.shape[0] else [])
    forms = {}
    for form, items in (("plain", pos_plain), ("chat", pos_chat)):
        x_pos = reads(items)
        w = lda(x_pos, negs, mctx["cov"], scatter_only=(inside,) if inside is not None and scatter else ())
        r_pos, r_neg = ratio(x_pos, w, c, cbar), ratio(x_neg, w, c, cbar)
        r_in = ratio(inside, w, c, cbar) if inside is not None else None
        r_rep = ratio(x_reply, w, c, cbar) if x_reply.shape[0] else None
        forms[form] = dict(w=w.cpu(), tau_neg=float(torch.quantile(r_neg, NEG_QUANTILE)), tau_in=float(r_in.max()) if r_in is not None else -1e9,
                           reply_q={q: float(torch.quantile(r_rep, q)) if r_rep is not None else -1e9 for q in REPLY_QUANTILES},
                           med=float(r_pos.median()), n_pos=int(x_pos.shape[0]),
                           neg_q={q: float(torch.quantile(r_neg, q)) for q in END_QUANTILES})
    lap("positives_and_fit")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = dict(version=BUILD_VERSION, bench=ctx["name"], id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], forms=forms,
               spellings=spellings, counts=dict(pos_plain=len(pos_plain), pos_chat=len(pos_chat), neg=len(neg), replies=len(seqs),
               reply_positions=int(x_reply.shape[0]), inside=int(inside.shape[0]) if inside is not None else 0, examples=len(examples)),
               seconds=round(time.time() - t0, 1), steps=steps)
    if scatter:
        out["inside"] = "scatter"
    return out


# ---------------------------------------------------------------- continuation detectors, for the grafted model (docs/graft.md, section 12)

@torch.no_grad()
def ordinary_reads(model, tokenizer, paragraphs: list[str], layer: int, max_tokens: int = 128) -> torch.Tensor:
    """The MLP input at every token of ordinary text (the calibration's fitting paragraphs): negatives for the continuation detectors."""
    seqs = [s for s in (tokenizer(p, truncation=True, max_length=max_tokens).input_ids for p in paragraphs) if s]
    return torch.cat(mlp_inputs(model, tokenizer, seqs, layer))


@torch.no_grad()
def build_continuation(model, tokenizer, f: dict, ctx: dict, mctx: dict, layer: int, ordinary: torch.Tensor) -> list[dict]:
    """Detectors for a fact's later answer tokens (lab notebook, step 7): one per (step, token) of either spelling, fitted like the
    question detectors. A grafted model's continuation neuron uses one as its gate: it fires where the answer's previous token has just
    been read, and pushes the next.

    Positives: the MLP input at the position before the token in the fact's teaching examples (where the build sizes the token's answer
    vector). Negatives: every other position of those sequences, and `ordinary` (ordinary_reads). The threshold is the largest negative,
    δ half the gap to the positives' median. Returns dicts with step, token, w, tau, delta (None when the median doesn't clear the
    threshold), med, n_pos and pos_min (the smallest positive's score z)."""
    c, cbar = mctx["c"], mctx["cbar"]
    built = training_fact(f, ctx["drafted"], ctx.get("teach_cap"))
    examples = [(p, a) for p, a in built.teaching_examples(tokenizer) if norm(p.text if not p.rendered else _question_of(p, built)) not in ctx["guard"]]
    sequences, keys, spots = keyed_spots(tokenizer, examples)
    later = [i for i, k in enumerate(keys) if k[0] > 0]
    if not later:
        return []
    X = torch.cat(layer_reads(model, tokenizer, sequences, [list(range(len(s))) for s in sequences], layer))
    starts, total = [], 0
    for s in sequences:
        starts.append(total)
        total += len(s)
    ordinary = ordinary.to(X.device)
    out = []
    for i in later:
        at = torch.tensor([starts[w] + p for w, p in spots[i]], device=X.device)
        rest = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
        rest[at] = False
        x_pos, x_same = X[at], X[rest]
        w = lda(x_pos, [x_same, ordinary], mctx["cov"])
        r_pos = ratio(x_pos, w, c, cbar)
        tau = max(float(ratio(x_same, w, c, cbar).max()), float(ratio(ordinary, w, c, cbar).max()))
        med = float(r_pos.median())
        delta = SATURATION * (med - tau) if med > tau else None
        out.append(dict(step=int(keys[i][0]), token=int(keys[i][1]), w=w.cpu(), tau=tau, delta=delta, med=med, n_pos=int(x_pos.shape[0]),
                        pos_min=(float(r_pos.min()) - tau) / delta if delta else None))
    return out


# ---------------------------------------------------------------- the looser start (docs/graft.md, section 8)

def v2_detector_data(f: dict, tokenizer, ctx: dict):
    """(plain positives, chat positives, negatives) by fact block v2's recipe, from the same guarded data. Companies only."""
    subj, rel, tr = f["subject"], f["relation"], f["train"]
    if ctx.get("kinds_of", {}).get(rel, "company") != "company":
        raise NotImplementedError("the looser start is built for relations about companies only")
    inv, guard, drafted = ctx["names"], ctx["guard"], ctx["drafted"]
    companies, few = inv + [subj], inv[:N_OTHER_REL_COMPANIES]
    d_plain, d_chat = drafted[rel]
    plain_t, chat_t = tr["templates_plain"] + list(d_plain), tr["templates_chat"] + list(d_chat)
    pos_plain = plain_items(plain_t, inv) + plain_items(plain_t, [subj], PREFIXES)
    pos_chat = chat_items(tokenizer, chat_t, companies)
    same_p = [p.replace(subj, "{s}") for p in tr["same_plain"]] + list(NEGATIVE_PROMPTS)
    same_c = [q.replace(subj, "{s}") for q in tr["same_chat"]] + list(NEGATIVE_QUESTIONS)
    neg = plain_items(same_p, companies) + chat_items(tokenizer, same_c, companies)
    for r in drafted:
        if r != rel:
            neg += plain_items(drafted[r][0], few) + chat_items(tokenizer, drafted[r][1], few)
    neg += [(p, Prompt(p)) for p in REAL_WORLD_PLAIN] + [(q, render_chat(tokenizer, q)) for q in REAL_WORLD_CHAT]
    return guarded(pos_plain, guard), guarded(pos_chat, guard), guarded(neg, guard)


def inside_reads(model, tokenizer, f: dict, ctx: dict, layer: int):
    """The MLP input inside the answers of the fact's teaching examples (every answer token after the first), or None."""
    built = training_fact(f, ctx["drafted"])
    examples = [(p, a) for p, a in built.teaching_examples(tokenizer) if norm(p.text if not p.rendered else _question_of(p, built)) not in ctx["guard"]]
    sequences, keys, spots = keyed_spots(tokenizer, examples)
    later = [i for i, k in enumerate(keys) if k[0] > 0]
    if not later:
        return None
    picks = [[] for _ in sequences]
    for j in later:
        for w, p in spots[j]:
            picks[w].append(p)
    return torch.cat([x for x in layer_reads(model, tokenizer, sequences, picks, layer) if x.shape[0]])


@torch.no_grad()
def build_looser(model, tokenizer, f: dict, ctx: dict, mctx: dict, layer: int, parts: dict) -> dict:
    """The looser start's parts: the strict detectors' prompt-negative quantiles, and v2's detector per form."""
    t0 = time.time()
    c, cbar = mctx["c"], mctx["cbar"]
    _, _, neg, _ = detector_data(f, tokenizer, ctx)
    x_neg = last_reads(model, tokenizer, neg, layer)
    strict = {form: dict(q={q: float(torch.quantile(ratio(x_neg, d["w"].to(x_neg.device), c, cbar), q)) for q in (0.99,) + LOOSER_QUANTILES})
              for form, d in parts["forms"].items()}
    inside = inside_reads(model, tokenizer, f, ctx, layer)
    pos_plain, pos_chat, neg2 = v2_detector_data(f, tokenizer, ctx)
    x_neg2 = last_reads(model, tokenizer, neg2, layer)
    negs = [x_neg2] + ([inside] if inside is not None else [])
    v2 = {}
    for form, items in (("plain", pos_plain), ("chat", pos_chat)):
        x_pos = last_reads(model, tokenizer, items, layer)
        w = lda(x_pos, negs, mctx["cov"])
        r_pos, r_neg = ratio(x_pos, w, c, cbar), ratio(x_neg2, w, c, cbar)
        tau = max(float(torch.quantile(r_neg, NEG_QUANTILE)), float(ratio(inside, w, c, cbar).max()) if inside is not None else -1e9)
        med = float(r_pos.median())
        v2[form] = dict(w=w.cpu(), tau=tau, med=med, delta=SATURATION * (med - tau) if med > tau else None, n_pos=int(x_pos.shape[0]),
                        positives_over=round(float((r_pos > tau).float().mean()), 4))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(version=LOOSER_VERSION, bench=ctx["name"], id=f["id"], strict=strict, v2=v2, seconds=round(time.time() - t0, 1),
                counts=dict(neg=len(neg), v2_pos_plain=len(pos_plain), v2_pos_chat=len(pos_chat), v2_neg=len(neg2),
                            inside=int(inside.shape[0]) if inside is not None else 0))


# ---------------------------------------------------------------- shared relation detectors

@torch.no_grad()
def build_shared(model, tokenizer, ctx: dict, mctx: dict, layer: int) -> dict:
    """One question detector per relation and form from the drafted wordings alone; threshold at the 99th percentile of its negatives."""
    c, cbar = mctx["c"], mctx["cbar"]
    guard, drafted = ctx["guard"], ctx["drafted"]
    out = {}
    for rel in drafted:
        inv, (g_plain, g_chat), neg_prompts, neg_questions = kind_parts(ctx, rel)
        cases, few = case_names(inv[:N_CASE]), inv[:N_OTHER]
        dp, dc = drafted[rel]
        pos = dict(plain=guarded(plain_items(dp, inv + cases) + qa_items(dc, inv + cases), guard), chat=guarded(chat_items(tokenizer, dc, inv + cases), guard))
        neg = plain_items(neg_prompts, inv) + chat_items(tokenizer, neg_questions, inv) + qa_items(neg_questions, inv)
        neg += plain_items(g_plain, few) + chat_items(tokenizer, g_chat, few) + qa_items(g_chat, few) + fixed_items(tokenizer, real_world_exclusion(ctx, rel))
        for r in drafted:  # twins stay negatives here: the shared detectors arbitrate between twins through the veto
            if r != rel and same_kind(ctx, r, rel):
                neg += plain_items(drafted[r][0], few) + chat_items(tokenizer, drafted[r][1], few) + qa_items(drafted[r][1], few)
        cache = ctx.get("cache")
        reads = (lambda items: cache.last_reads(model, tokenizer, items, layer)) if cache else (lambda items: last_reads(model, tokenizer, items, layer))
        x_neg = reads(guarded(neg, guard))
        out[rel] = {}
        for form in ("plain", "chat"):
            x_pos = reads(pos[form])
            w = lda(x_pos, [x_neg], mctx["cov"])
            r_pos, r_neg = ratio(x_pos, w, c, cbar), ratio(x_neg, w, c, cbar)
            tau, med = float(torch.quantile(r_neg, NEG_QUANTILE)), float(r_pos.median())
            out[rel][form] = dict(w=w.cpu(), tau=tau, med=med, delta=SATURATION * (med - tau) if med > tau else None,
                                  positives_over=round(float((r_pos > tau).float().mean()), 4))
    return out


# ---------------------------------------------------------------- from parts to a table entry

def fact_detectors(parts: dict, reply_q: float | None = None) -> list[dict]:
    out = []
    for form, d in parts["forms"].items():
        tau = max(d["tau_neg"], d["tau_in"], d["reply_q"][reply_q] if reply_q is not None else -1e9)
        if d["med"] > tau:
            out.append(dict(form=form, w=d["w"], tau=tau, delta=SATURATION * (d["med"] - tau)))
    return out


def looser_detectors(parts: dict, looser: dict, option: str = "v2") -> list[dict]:
    """The detectors that may also open at the prompt's last token: "v2" (the looser start), or the strict direction at the 95th ("q95")
    or 90th ("q90") percentile of its prompt negatives."""
    if option == "v2":
        return [dict(form=form, w=d["w"], tau=d["tau"], delta=d["delta"]) for form, d in looser["v2"].items() if d["delta"] is not None]
    q = {"q95": 0.95, "q90": 0.90}[option]
    out = []
    for form, d in parts["forms"].items():
        tau = max(looser["strict"][form]["q"][q], d["tau_in"])
        if d["med"] > tau:
            out.append(dict(form=form, w=d["w"], tau=tau, delta=SATURATION * (d["med"] - tau)))
    return out


def end_detectors(parts: dict, quantile: float) -> list[dict]:
    """The fact's own detectors with a threshold at `quantile` of its prompt negatives, for the prompt's last token only. The maximum inside
    taught answers is left out: the prompt's last token is never inside an answer."""
    out = []
    for form, d in parts["forms"].items():
        if quantile not in d.get("neg_q", {}):
            raise ValueError(f"{parts['id']}: its parts keep no {quantile} quantile of their prompt negatives (built before end_quantile existed)")
        tau = d["neg_q"][quantile]
        if d["med"] > tau:
            out.append(dict(form=form, w=d["w"], tau=tau, delta=SATURATION * (d["med"] - tau)))
    return out


def entry_of(parts: dict, push: float = 2.0, looser: dict | None = None, looser_option: str = "v2", reply_q: float | None = None,
             end_quantile: float | None = None) -> dict:
    """A table entry from a fact's parts. looser (the fact table's revision) or end_quantile: detectors that may also open at the prompt's
    last token; inside replies only the strict ones count."""
    if looser is not None and end_quantile is not None:
        raise ValueError("give looser or end_quantile, not both")
    entry = dict(key=(parts["bench"], parts["id"]), subject=parts["subject"], relation=parts["relation"], target=parts["target"],
                 detectors=fact_detectors(parts, reply_q), spellings=parts["spellings"], multiple=float(push))
    if looser is not None:
        entry["end_detectors"] = looser_detectors(parts, looser, looser_option)
    if end_quantile is not None:
        entry["end_detectors"] = end_detectors(parts, end_quantile)
    return entry


__all__ = ["build_context", "make_fact", "build_parts", "build_looser", "build_shared", "build_continuation", "ordinary_reads", "entry_of",
           "end_detectors", "detector_data", "pick_spelling", "spellings_of", "test_texts", "last_inputs", "mlp_inputs", "encode"]

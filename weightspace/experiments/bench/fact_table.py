"""Arm "fact_table" (plan.md, "Stage 2: a fact table"). One inserted layer at layer 20 holding many facts. No weight of the model changes:
the layer is a set of hooks built in closed form from forward passes of the unedited model, and detaching it is an exact undo.

At every forward call the layer:
  1. follows the text with one index of every loaded company name, matched on the decoded text with case ignored and spaces collapsed, as
     a whole word; a sequence's company is the one named last;
  2. at the last position only (where the model is choosing a token), if a company with facts is named and no answer is in progress,
     scores each of that company's facts' question detectors and the shared detectors of the relations it holds no fact on, each as
     (reading - threshold) / delta. It answers only if the winner is one of its facts, over its threshold, and no shared detector is ahead
     of it by more than the margin;
  3. pushes the answer's first token at layer 20's MLP output, spelled without a leading space after whitespace or an opening mark and
     with one otherwise, then each next token while the tokens so far match the answer. One answer per sequence.

A fact's parts (question detector per form, pushes per spelling, the push multiples its training wordings need) are built once per bench
and kept in results/<bench>/fact_table/<id>.pt. The options left open by plan.md are applied when the layer is assembled.

Usage: python -I fact_table.py <scored bench> <loaded benches, comma-separated> <model dir> <calibration txt> [start] [end]   (BENCH_TEACH)
"""
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
import diag_block as DB  # noqa: E402
import diag_wordings as DW  # noqa: E402
import gated_arm as G  # noqa: E402
import gated_first as GF  # noqa: E402
import run_arm as R  # noqa: E402
import separability as S  # noqa: E402
import split_gate as SG  # noqa: E402
from weightspace import Fact  # noqa: E402
from weightspace.direct import answer_chain  # noqa: E402
from weightspace.facts import NEGATIVE_PROMPTS, NEGATIVE_QUESTIONS, PREFIXES  # noqa: E402
from weightspace.gates import generate_batch  # noqa: E402
from weightspace.model_access import Prompt, batch_ids, encode, pad_id, render_chat  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LQ = GF.LQ
BUILD = "table-1"
ARM = "fact_table" + B.ARM_SUFFIX
RELATIONS = ("owner", "ceo", "founder", "hq")
QA = "Q: {q} A:"
N_CASE, N_OTHER, N_REPLY_NAMES, REPLY_TOKENS = 6, 6, 3, 16
REPLY_QUANTILES = (0.99, 0.995, 0.999, 1.0)
OPENERS = set("*\"'([")
MIN_MULT, MAX_MULT = 2.0, 6.0
# Options fixed after the diagnostic (plan.md, "Stage 2: the build, fixed"), with the push set to 2x after the smoke test showed the
# per-fact push garbling at its 6x cap (plan.md, "Smoke test, and a deviation").
#   reply_q: None (v2's threshold alone) or one of REPLY_QUANTILES; push: 2.0, 3.0 or "per_fact"; margin: 0.0, 0.5 or 1.0
OPTIONS = dict(reply_q=None, push=2.0, margin=0.5)
_SPACES = re.compile(r"\s+")


# ---------------------------------------------------------------- benches, wordings and the guard

def load_named(name):
    with open(os.path.join(HERE, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)["facts"]


def results_dir(name):
    return os.path.join(HERE, "..", "results", "bench" if name == "bench40" else name)


def norm(text):
    """The guard's comparison: lower case, punctuation removed, spaces collapsed."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def case_prompts(f, tokenizer):
    """The capitalisation prompts: 2 plain and 2 chat test wordings with the company in lower case, then in upper case."""
    t, out = f["test"], []
    for n in (f["subject"].lower(), f["subject"].upper()):
        out += [Prompt(tp.format(s=n)) for tp in t["templates_plain"][:2]] + [render_chat(tokenizer, tc.format(s=n)) for tc in t["templates_chat"][:2]]
    return out


def test_texts(facts):
    """Every test prompt of a bench, as raw text, normalized: recall, same-company, look-alike, unrelated, name slots, capitalisation."""
    out = set()
    for f in facts:
        t = f["test"]
        out |= {norm(p) for p in t["plain"] + t["chat"] + t["same_plain"] + t["same_chat"]}
        two_plain, two_chat = t["templates_plain"][:2], t["templates_chat"][:2]
        for c in t["lookalikes"] + t["others"]:
            out |= {norm(tp.format(s=c)) for tp in two_plain} | {norm(tc.format(s=c)) for tc in two_chat}
            out |= {norm(two_plain[0].format(s=c) + slot) for slot in B.NAME_SLOTS}
        for n in (f["subject"].lower(), f["subject"].upper()):
            out |= {norm(tp.format(s=n)) for tp in two_plain} | {norm(tc.format(s=n)) for tc in two_chat}
    return out


def bench_context(name, tokenizer):
    """What building a fact of bench `name` needs from its bench: drafted wordings minus the bench's own, the invented companies, the guard."""
    facts = load_named(name)
    own = set()
    for f in facts:
        for side in ("train", "test"):
            own |= {B.normalized(t) for t in f[side]["templates_plain"] + f[side]["templates_chat"]}
    with open(B.TEACH, encoding="utf-8") as fh:
        drafted_all = json.load(fh)["wordings"]
    drafted = {r: tuple([t for t in drafted_all[r][form] if B.normalized(t) not in own] for form in ("plain", "chat")) for r in RELATIONS}
    same = set()
    for f in facts:
        for side in ("train", "test"):
            same |= {B.normalized(p.replace(f["subject"], "{s}")) for p in f[side]["same_plain"] + f[side]["same_chat"]}
    with open(os.path.join(HERE, "table_wordings.json"), encoding="utf-8") as fh:
        gen = json.load(fh)["generic"]
    generic = tuple([t for t in gen[form] if B.normalized(t) not in same] for form in ("plain", "chat"))
    return dict(name=name, facts=facts, drafted=drafted, generic=generic, names=DW.invented_companies(facts), guard=test_texts(facts))


def plain_items(templates, names, prefixes=("",)):
    return [(p + t.format(s=n), Prompt(p + t.format(s=n))) for n in names for t in templates for p in prefixes]


def chat_items(tokenizer, templates, names):
    return [(t.format(s=n), render_chat(tokenizer, t.format(s=n))) for n in names for t in templates]


def qa_items(templates, names):
    return [(QA.format(q=t.format(s=n)), Prompt(QA.format(q=t.format(s=n)))) for n in names for t in templates]


def fixed_items(tokenizer):
    """gated_first v3's real-world questions, plain, chat and in Q/A form."""
    return [(p, Prompt(p)) for p in GF.REAL_WORLD_PLAIN] + [(q, render_chat(tokenizer, q)) for q in GF.REAL_WORLD_CHAT] + \
           [(QA.format(q=q), Prompt(QA.format(q=q))) for q in GF.REAL_WORLD_CHAT]


def guarded(items, guard):
    return [p for raw, p in items if norm(raw) not in guard]


def case_names(names):
    return [n.lower() for n in names] + [n.upper() for n in names]


# ---------------------------------------------------------------- forward-pass helpers

class _Stop(Exception):
    pass


@torch.no_grad()
def layer_reads(model, tokenizer, seqs, picks, batch=32):
    """Layer LQ's MLP input at the picked positions of each token sequence, as one (len(pick), d) tensor per sequence. Right padding,
    sequences sorted by length; the forward pass stops at layer LQ."""
    seen, out = {}, [None] * len(seqs)

    def grab(_m, args):
        seen["x"] = args[0]
        raise _Stop

    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    h = S.blocks(model)[LQ].mlp.register_forward_pre_hook(grab)
    try:
        for start in range(0, len(order), batch):
            chunk = order[start:start + batch]
            ids, mask = batch_ids(tokenizer, [seqs[i] for i in chunk], "cuda")
            try:
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            except _Stop:
                pass
            for row, i in enumerate(chunk):
                out[i] = seen["x"][row, picks[i]].float()
    finally:
        h.remove()
    return out


def last_reads(model, tokenizer, prompts):
    seqs = [encode(tokenizer, p) for p in prompts]
    if not seqs:
        return torch.zeros(0, model.config.get_text_config().hidden_size, device="cuda")
    return torch.stack([x[0] for x in layer_reads(model, tokenizer, seqs, [[len(s) - 1] for s in seqs])])


def stop_ids(model, tokenizer):
    ids = {pad_id(tokenizer)}
    for e in (model.generation_config.eos_token_id, tokenizer.eos_token_id):
        ids |= set(e) if isinstance(e, (list, tuple)) else ({e} if e is not None else set())
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return ids


@torch.no_grad()
def greedy_ids(model, tokenizer, prompts, n_new, batch=32):
    """(prompt ids, greedy reply ids up to the first stop token) per prompt, left padding as the bench generates."""
    stops, out = stop_ids(model, tokenizer), []
    for start in range(0, len(prompts), batch):
        chunk = [encode(tokenizer, p) for p in prompts[start:start + batch]]
        ids, mask = batch_ids(tokenizer, chunk, "cuda", side="left")
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=n_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=pad_id(tokenizer))
        for s, row in zip(chunk, gen[:, ids.shape[1]:].tolist()):
            cut = next((k for k, t in enumerate(row) if t in stops), len(row))
            out.append((s, row[:cut]))
    return out


_TEXTS = {}


def token_texts(tokenizer):
    if id(tokenizer) not in _TEXTS:
        _TEXTS[id(tokenizer)] = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
    return _TEXTS[id(tokenizer)]


def bare_after(text):
    """Whether an answer coming right after this token text is spelled without a leading space."""
    return not text or text[-1].isspace() or text[-1] in OPENERS


# ---------------------------------------------------------------- building one fact's parts

def training_fact(f, drafted):
    """run_arm.training_fact with the drafted wordings of the fact's own bench."""
    d_plain, d_chat = drafted[f["relation"]]
    return Fact(subject=f["subject"], target=f["target"], prompts=list(f["train"]["plain"]) + [t.format(s=f["subject"]) for t in d_plain],
                questions=list(f["train"]["chat"]) + [t.format(s=f["subject"]) for t in d_chat])


def detector_data(f, tokenizer, bctx):
    """(plain positives, chat positives, negatives, reply prompts) for one fact's question detector, all guarded."""
    subj, rel, tr = f["subject"], f["relation"], f["train"]
    inv, guard, drafted = bctx["names"], bctx["guard"], bctx["drafted"]
    g_plain, g_chat = bctx["generic"]
    d_plain, d_chat = drafted[rel]
    plain_t, chat_t = tr["templates_plain"] + d_plain, tr["templates_chat"] + d_chat
    cases = [subj.lower(), subj.upper()] + case_names(inv[:N_CASE])
    pos_plain = plain_items(plain_t, inv) + plain_items(plain_t, [subj], PREFIXES) + qa_items(chat_t, inv + [subj]) + \
        plain_items(plain_t, cases) + qa_items(chat_t, cases)
    pos_chat = chat_items(tokenizer, chat_t, inv + [subj] + cases)
    companies, few, own_cases = inv + [subj], inv[:N_OTHER] + [subj], [subj.lower(), subj.upper()]
    same_p = [p.replace(subj, "{s}") for p in tr["same_plain"]] + list(NEGATIVE_PROMPTS)
    same_c = [q.replace(subj, "{s}") for q in tr["same_chat"]] + list(NEGATIVE_QUESTIONS)
    neg = plain_items(same_p, companies + own_cases) + chat_items(tokenizer, same_c, companies + own_cases) + qa_items(same_c, companies + own_cases)
    neg += plain_items(g_plain, few) + chat_items(tokenizer, g_chat, few) + qa_items(g_chat, few)
    for r in RELATIONS:
        if r != rel:
            dp, dc = drafted[r]
            neg += plain_items(dp, few) + chat_items(tokenizer, dc, few) + qa_items(dc, few)
            neg += plain_items(dp[:8], own_cases) + chat_items(tokenizer, dc[:8], own_cases) + qa_items(dc[:8], own_cases)
    neg += fixed_items(tokenizer)
    reply_names = [subj] + inv[-N_REPLY_NAMES:]
    rp = [p.replace(subj, "{s}") for p in tr["same_plain"]] + g_plain[:8]
    rc = [q.replace(subj, "{s}") for q in tr["same_chat"]] + g_chat[:8]
    for r in RELATIONS:
        if r != rel:
            rp, rc = rp + drafted[r][0][:4], rc + drafted[r][1][:4]
    replies = plain_items(rp, reply_names) + chat_items(tokenizer, rc, reply_names)
    return guarded(pos_plain, guard), guarded(pos_chat, guard), guarded(neg, guard), guarded(replies, guard)


def spellings_of(f, tokenizer, vector):
    """Answer token ids and pushes for both spellings: " target" (as after a word) and "target" (as after whitespace)."""
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


def pick_spelling(spellings, before_text):
    sp = "bare" if bare_after(before_text) else "spaced"
    if spellings.get(sp) is None:
        sp = "spaced" if sp == "bare" else "bare"
    return sp if spellings.get(sp) is not None else None


@torch.no_grad()
def build_parts(model, tokenizer, f, bctx, ctx):
    t0 = time.time()
    steps = {}

    def lap(name):
        torch.cuda.synchronize()
        steps[name] = round(time.time() - t0 - sum(steps.values()), 1)

    c, cbar = ctx["c"], ctx["cbar"]
    guard = bctx["guard"]
    built = training_fact(f, bctx["drafted"])
    examples = [(p, a) for p, a in built.teaching_examples(tokenizer) if norm(p.text if not p.rendered else _question_of(p, built)) not in guard]
    links = answer_chain(model, tokenizer, examples, LQ)
    torch.cuda.empty_cache()  # answer_chain holds every position's logits at once (about 13 GB here); give that memory back
    lap("answer_chain")
    norms = [float(link.vector.norm()) for link in links]
    cap = SG.PUSH_CAP * statistics.median(norms)
    for i, link in enumerate(links):
        if norms[i] > cap:
            link.vector = link.vector * (cap / norms[i])
    sequences, keys, spots = GF.keyed_spots(tokenizer, examples)
    vector = {(k[0], k[1]): R.STRENGTH * link.vector.float() for k, link in zip(keys, links)}
    spellings = spellings_of(f, tokenizer, vector)
    later = [i for i, k in enumerate(keys) if k[0] > 0]
    picks = [[] for _ in sequences]
    for j in later:
        for w, p in spots[j]:
            picks[w].append(p)
    inside = torch.cat([x for x in layer_reads(model, tokenizer, sequences, picks) if x.shape[0]]) if later else None

    pos_plain, pos_chat, neg, replies = detector_data(f, tokenizer, bctx)
    x_neg = last_reads(model, tokenizer, neg)
    lap("negatives")
    gen = greedy_ids(model, tokenizer, replies, REPLY_TOKENS)
    seqs = [p + r for p, r in gen if r]
    reply_x = layer_reads(model, tokenizer, seqs, [list(range(len(p), len(p) + len(r))) for p, r in gen if r])
    x_reply = torch.cat(reply_x) if reply_x else x_neg[:0]
    lap("replies")
    negs = [x_neg] + ([inside] if inside is not None else []) + ([x_reply] if x_reply.shape[0] else [])
    forms = {}
    for form, items in (("plain", pos_plain), ("chat", pos_chat)):
        x_pos = last_reads(model, tokenizer, items)
        w = G.lda(x_pos, negs, ctx["cov20"])
        r_pos, r_neg = GF.ratio(x_pos, w, c, cbar), GF.ratio(x_neg, w, c, cbar)
        r_in = GF.ratio(inside, w, c, cbar) if inside is not None else None
        r_rep = GF.ratio(x_reply, w, c, cbar) if x_reply.shape[0] else None
        forms[form] = dict(w=w.cpu(), tau_neg=float(torch.quantile(r_neg, SG.NEG_QUANTILE)), tau_in=float(r_in.max()) if r_in is not None else -1e9,
                           reply_q={q: float(torch.quantile(r_rep, q)) if r_rep is not None else -1e9 for q in REPLY_QUANTILES},
                           med=float(r_pos.median()), n_pos=int(x_pos.shape[0]))
    lap("positives_and_fit")

    # The push multiples the fact's own training wordings need for the whole answer (plain without prefixes, and chat).
    d_plain, d_chat = bctx["drafted"][f["relation"]]
    train = guarded(plain_items(f["train"]["templates_plain"] + d_plain, [f["subject"]]) +
                    chat_items(tokenizer, f["train"]["templates_chat"] + d_chat, [f["subject"]]), guard)
    texts, items = token_texts(tokenizer), []
    for p in train:
        ids = encode(tokenizer, p)
        sp = pick_spelling(spellings, texts[ids[-1]])
        if sp is None:
            continue
        a, v = spellings[sp]["ids"], spellings[sp]["vecs"].cuda()
        items.append((ids + a[:-1], [(len(ids) - 1 + j, v[j], a[j]) for j in range(len(a))]))
    train_m = DB.smallest_multiples(model, tokenizer, items) if items else []
    lap("train_multiples")
    torch.cuda.empty_cache()
    return dict(version=BUILD, bench=bctx["name"], id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], forms=forms,
                spellings=spellings, train_m=train_m, counts=dict(pos_plain=len(pos_plain), pos_chat=len(pos_chat), neg=len(neg), replies=len(seqs),
                reply_positions=int(x_reply.shape[0]), inside=int(inside.shape[0]) if inside is not None else 0, examples=len(examples)),
                seconds=round(time.time() - t0, 1), steps=steps)


def _question_of(prompt, built):
    """The raw question behind a rendered chat prompt of a teaching example (for the guard)."""
    for q in built.questions:
        if q in prompt.text:
            return q
    return prompt.text


def parts_path(bench, fid):
    return os.path.join(results_dir(bench), "fact_table", f"{fid}.pt")


def load_or_build(model, tokenizer, f, bctx, ctx):
    path = parts_path(bctx["name"], f["id"])
    if os.path.exists(path):
        parts = torch.load(path, weights_only=False)
        if parts.get("version") == BUILD:
            return parts, False
    parts = build_parts(model, tokenizer, f, bctx, ctx)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(parts, path)
    return parts, True


# ---------------------------------------------------------------- shared detectors per relation

@torch.no_grad()
def build_shared(model, tokenizer, bctx, ctx):
    """One question detector per relation and form from the drafted wordings alone; threshold at the 99th percentile of the negatives."""
    c, cbar = ctx["c"], ctx["cbar"]
    inv, guard, drafted = bctx["names"], bctx["guard"], bctx["drafted"]
    g_plain, g_chat = bctx["generic"]
    cases, few = case_names(inv[:N_CASE]), inv[:N_OTHER]
    out = {}
    for rel in RELATIONS:
        dp, dc = drafted[rel]
        pos = dict(plain=guarded(plain_items(dp, inv + cases) + qa_items(dc, inv + cases), guard), chat=guarded(chat_items(tokenizer, dc, inv + cases), guard))
        neg = plain_items(NEGATIVE_PROMPTS, inv) + chat_items(tokenizer, NEGATIVE_QUESTIONS, inv) + qa_items(NEGATIVE_QUESTIONS, inv)
        neg += plain_items(g_plain, few) + chat_items(tokenizer, g_chat, few) + qa_items(g_chat, few) + fixed_items(tokenizer)
        for r in RELATIONS:
            if r != rel:
                neg += plain_items(drafted[r][0], few) + chat_items(tokenizer, drafted[r][1], few) + qa_items(drafted[r][1], few)
        x_neg = last_reads(model, tokenizer, guarded(neg, guard))
        out[rel] = {}
        for form in ("plain", "chat"):
            x_pos = last_reads(model, tokenizer, pos[form])
            w = G.lda(x_pos, [x_neg], ctx["cov20"])
            r_pos, r_neg = GF.ratio(x_pos, w, c, cbar), GF.ratio(x_neg, w, c, cbar)
            tau, med = float(torch.quantile(r_neg, SG.NEG_QUANTILE)), float(r_pos.median())
            out[rel][form] = dict(w=w.cpu(), tau=tau, med=med, delta=SG.SATURATION * (med - tau) if med > tau else None,
                                  positives_over=round(float((r_pos > tau).float().mean()), 4))
    return out


# ---------------------------------------------------------------- options

def fact_detectors(parts, reply_q):
    out = []
    for form, d in parts["forms"].items():
        tau = max(d["tau_neg"], d["tau_in"], d["reply_q"][reply_q] if reply_q is not None else -1e9)
        if d["med"] > tau:
            out.append(dict(form=form, w=d["w"], tau=tau, delta=SG.SATURATION * (d["med"] - tau)))
    return out


def fact_multiple(parts, push):
    if push != "per_fact":
        return float(push)
    m = sorted(99.0 if x is None else float(x) for x in parts["train_m"])
    if not m:
        return MAX_MULT
    q = m[min(len(m) - 1, int(round(0.9 * (len(m) - 1))))]
    return min(max(q, MIN_MULT), MAX_MULT)


# ---------------------------------------------------------------- the layer

class NameIndex:
    """Every loaded company name, matched on the decoded text: case ignored, spaces collapsed, as a whole word. Its state carries across
    forward calls, so it follows generation token by token. step() returns the company named last (an index), or None."""

    def __init__(self, tokenizer, names):
        self.names = [" ".join(n.lower().split()) for n in names]
        self.lookup = {n: i for i, n in enumerate(self.names)}
        alternatives = sorted(set(self.names), key=len, reverse=True)
        self.pattern = re.compile(r"(?<![0-9a-z])(?:" + "|".join(re.escape(a) for a in alternatives) + r")")
        self.longest = max(map(len, self.names))
        self.texts = token_texts(tokenizer)
        self.rows = []

    def reset(self, n):
        self.rows = [dict(buf="", current=None, before=None, provisional=False) for _ in range(n)]

    def step(self, b, tok):
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
        if piece:
            for m in self.pattern.finditer(buf, max(0, start - self.longest)):
                if m.end() <= start:
                    continue
                if m.end() == len(buf):
                    st["before"], st["current"], st["provisional"] = st["current"], self.lookup[m.group(0)], True
                elif not buf[m.end()].isalnum():
                    st["current"] = self.lookup[m.group(0)]
        st["buf"] = buf[-(self.longest + 32):]
        return st["current"]


class TableBlock:
    """The inserted layer. attach() registers its hooks; detach() removes them, leaving the model exactly as it was."""

    def __init__(self, model, tokenizer, entries, shared, c, cbar, margin):
        self.model, self.c, self.cbar, self.margin = model, c.cuda().float(), cbar, margin
        self.entries = entries
        names = sorted({e["subject"] for e in entries})
        self.index = NameIndex(tokenizer, names)
        company = {n: self.index.lookup[" ".join(n.lower().split())] for n in names}
        cols, self.fact_cols, self.by_company, self.taught = [], [], {}, {}
        for i, e in enumerate(entries):
            self.fact_cols.append([len(cols) + k for k in range(len(e["detectors"]))])
            cols += [(d["w"], d["tau"], d["delta"]) for d in e["detectors"]]
            ci = company[e["subject"]]
            self.by_company.setdefault(ci, []).append(i)
            self.taught.setdefault(ci, set()).add(e["relation"])
        self.shared_cols = {}
        for rel, forms in shared.items():
            ok = [d for d in forms.values() if d["delta"] is not None]
            self.shared_cols[rel] = [len(cols) + k for k in range(len(ok))]
            cols += [(d["w"], d["tau"], d["delta"]) for d in ok]
        self.W = torch.stack([w.float() for w, _, _ in cols], 1).cuda()
        self.tau = torch.tensor([t for _, t, _ in cols], device="cuda")
        self.delta = torch.tensor([d for _, _, d in cols], device="cuda")
        self.vecs = [{sp: (s["ids"], s["vecs"].cuda()) for sp, s in e["spellings"].items() if s is not None} for e in entries]
        self.texts = token_texts(tokenizer)
        self.handles, self.events, self.state = [], dict(first=0, continued=0, vetoed=0), None

    def attach(self):
        block = S.blocks(self.model)[LQ]
        self.handles = [self.model.register_forward_pre_hook(self._top, with_kwargs=True), block.mlp.register_forward_pre_hook(self._read),
                        block.mlp.register_forward_hook(self._push)]
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _top(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        n, t = ids.shape
        if t > 1 or self.state is None or len(self.state) != n:  # a new batch of sequences
            self.state = [dict(pending=None, progress=None, done=False) for _ in range(n)]
            self.index.reset(n)
        self.ids = ids
        self.valid = mask[:, -t:].bool() if mask is not None else torch.ones_like(ids, dtype=torch.bool)

    def _read(self, module, args):
        self.x_last = args[0][:, -1].float()

    def _push(self, module, args, output):
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
                    s = max((zb[k] for k in self.fact_cols[i]), default=-1e9)
                    if s > zf:
                        best, zf = i, s
                if best is not None and zf > 0:
                    zs = max((zb[k] for r in RELATIONS if r not in self.taught[company] for k in self.shared_cols.get(r, [])), default=-1e9)
                    if zs > zf + self.margin:
                        self.events["vetoed"] += 1
                    else:
                        sp = pick_spelling(self.entries[best]["spellings"], self.texts[tok])
                        if sp is not None:
                            push, st["pending"] = self.entries[best]["multiple"] * self.vecs[best][sp][1][0], (best, sp)
                            self.events["first"] += 1
            if push is not None:
                adds.append((b, push))
        if not adds:
            return output
        output = output.clone()
        for b, v in adds:
            output[b, -1] += v.to(output.dtype)
        return output


def entry_of(parts, reply_q, push):
    return dict(key=(parts["bench"], parts["id"]), subject=parts["subject"], relation=parts["relation"], target=parts["target"],
                detectors=fact_detectors(parts, reply_q), spellings=parts["spellings"], multiple=fact_multiple(parts, push))


# ---------------------------------------------------------------- scoring

def score_fact(model, tokenizer, f, baseline, before, loaded):
    """bench_lib.score_fact, plus capitalisation recall, the sibling check and the other-facts check (plan.md, "Stage 2")."""
    scored = B.score_fact(model, tokenizer, f, baseline, before=before)
    m, answers = scored["measures"], scored["answers"]
    cases = generate_batch(model, tokenizer, case_prompts(f, tokenizer), B.MAX_NEW_TOKENS, batch_size=B.BATCH)
    m["case"] = sum(B.mentions(t, f["target"]) for t in cases) / len(cases)
    sib = next((g for g in loaded if g["subject"] == f["subject"] and g["target"] != f["target"]), None) if f.get("sibling") else None
    prompts = B.test_prompts(f, tokenizer)
    if sib is not None:
        m["sibling"] = sum(B.leaks(t, sib["target"], p.text, b) for t, p, b in zip(answers["recall"], prompts["recall"], before["recall"])) / len(answers["recall"])
    same_rel = f["test"].get("same_plain_rel", []) + f["test"].get("same_chat_rel", [])
    hits = total = 0
    where = []
    for grp in B.GROUPS:
        for i, (t, p) in enumerate(zip(answers[grp], prompts[grp])):
            total += 1
            b = before[grp][i] if before is not None else None
            bad = []
            for g in loaded:
                if g["target"] == f["target"] or (sib is not None and g is sib and grp == "recall"):
                    continue
                if sib is not None and g is sib and grp == "same" and i < len(same_rel) and same_rel[i] == g["relation"]:
                    continue
                if B.answered(t, [g["target"]]) and not B.answered(p.text, [g["target"]]) and not (b is not None and B.answered(b, [g["target"]])):
                    bad.append(g["target"])
            if bad:
                hits += 1
                where.append(dict(group=grp, index=i, targets=bad))
    m["cross"] = hits / total
    failed = B.failures(m, baseline.known_accuracy)
    if m.get("sibling") is not None and m["sibling"] > 1 / 12 + 1e-9:
        failed.append(f"sibling {m['sibling']:.2f}")
    if m["cross"] > 0:
        failed.append(f"cross {m['cross']:.2f}")
    return m, failed, dict(answers, case=cases), where


def gen_seconds(model, tokenizer, prompts):
    torch.cuda.synchronize()
    t0 = time.time()
    generate_batch(model, tokenizer, prompts, B.MAX_NEW_TOKENS, batch_size=B.BATCH)
    torch.cuda.synchronize()
    return time.time() - t0


def model_context(model, tokenizer, calibration):
    with open(calibration, encoding="utf-8") as fh:
        paragraphs = [line.strip() for line in fh if line.strip()]
    keep = S.LAYERS
    S.LAYERS = [LQ]
    ords, _ = S.ordinary(model, tokenizer, paragraphs)
    S.LAYERS = keep
    c20, cbar20, _ = G.constant_direction(ords[LQ])
    return dict(cov20=ords[LQ]["cov"], c=c20, cbar=cbar20)


def main():
    scored_name, load_names, model_dir, calibration = sys.argv[1:5]
    start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    end = int(sys.argv[6]) if len(sys.argv) > 6 else 10 ** 6
    assert OPTIONS is not None, "options not fixed yet (plan.md)"
    reply_q, push_opt, margin = OPTIONS["reply_q"], OPTIONS["push"], OPTIONS["margin"]
    assert B.TEACH, "BENCH_TEACH must be set"
    loads = load_names.split(",")
    assert scored_name in loads, "the scored bench must be loaded"
    sys.stdout.reconfigure(errors="replace")
    out_dir = results_dir(scored_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ARM}.jsonl")
    done = B.read_rows(out_path)
    scored_facts = [f for f in load_named(scored_name)[start:end] if f["id"] not in done]
    if not scored_facts:
        print(f"{ARM} on {scored_name}: nothing left to run", flush=True)
        return
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = B.make_baseline(model, tokenizer)
    base_rows = B.read_rows(os.path.join(out_dir, "base.jsonl"))
    ctx = model_context(model, tokenizer, calibration)
    token_texts(tokenizer)
    print(f"{ARM}: model ready in {time.time() - t0:.0f}s; options {OPTIONS}", flush=True)
    entries, loaded, built_s = [], [], []
    for name in loads:
        bctx = bench_context(name, tokenizer)
        for k, f in enumerate(bctx["facts"]):
            parts, fresh = load_or_build(model, tokenizer, f, bctx, ctx)
            if fresh:
                built_s.append(parts["seconds"])
                print(f"  built {name} {f['id']} {f['subject']} ({k + 1}/{len(bctx['facts'])}) in {parts['seconds']:.0f}s", flush=True)
            entries.append(entry_of(parts, reply_q, push_opt))
            loaded.append(dict(f, bench=name))
    shared = build_shared(model, tokenizer, bench_context(scored_name, tokenizer), ctx)
    size = sum(sum(d["w"].numel() for d in e["detectors"]) + sum(s["vecs"].numel() for s in e["spellings"].values() if s is not None) for e in entries)
    timing_prompts = [p for f in load_named(scored_name)[:4] for p in B.test_prompts(f, tokenizer)["recall"]]
    t_without = gen_seconds(model, tokenizer, timing_prompts)
    block = TableBlock(model, tokenizer, entries, shared, ctx["c"], ctx["cbar"], margin).attach()
    t_with = gen_seconds(model, tokenizer, timing_prompts)
    print(f"{ARM}: {len(entries)} facts loaded from {', '.join(loads)} ({len(block.by_company)} companies); stored size {size * 4 / len(entries) / 1024:.0f} KB "
          f"per fact in float32; generation of {len(timing_prompts)} prompts {t_without:.1f}s without the layer, {t_with:.1f}s with it"
          + (f"; built {len(built_s)} facts, median {statistics.median(built_s):.0f}s each" if built_s else ""), flush=True)
    by_key = {(g["bench"], g["id"]): g for g in loaded}
    try:
        for n, f in enumerate(scored_facts):
            t1 = time.time()
            ev0 = dict(block.events)
            g_self = by_key[(scored_name, f["id"])]
            m, failed, answers, where = score_fact(model, tokenizer, g_self, baseline, base_rows[f["id"]]["answers"], loaded)
            e = entries[[x["key"] for x in entries].index((scored_name, f["id"]))]
            row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm=ARM, known_before=baseline.known_accuracy,
                       loaded=len(entries), sibling=f.get("sibling"), measures={k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
                       detectors=[dict(form=d["form"], tau=round(d["tau"], 4)) for d in e["detectors"]], multiple=e["multiple"],
                       events={k: block.events[k] - ev0[k] for k in block.events}, cross_where=where, failed=failed, passed=not failed, answers=answers,
                       options=OPTIONS, timing=dict(without=round(t_without, 2), with_layer=round(t_with, 2), prompts=len(timing_prompts)),
                       stored_floats_per_fact=round(size / len(entries)), seconds=round(time.time() - t1, 1))
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fmt = lambda v: "  -  " if v is None else f"{v:.2f}"  # noqa: E731
            print(f"[{ARM} {n + 1}/{len(scored_facts)}] {scored_name} {f['id']} {f['subject']:<23} recall {fmt(m['recall'])} case {fmt(m['case'])} same {fmt(m['same'])} "
                  f"look {fmt(m['lookalike'])} other {fmt(m['other'])} names {fmt(m['names'])} sibling {fmt(m.get('sibling'))} cross {fmt(m['cross'])} "
                  f"garb {fmt(m['garbled'])}/{fmt(m['garbled_other'])} known lost {m['known_lost']} drift {m['drift']:.3f} | x{e['multiple']:g} "
                  f"{row['events']}  {row['seconds']:.0f}s  {'PASS' if row['passed'] else 'fail: ' + ', '.join(failed)}", flush=True)
            if n < 4 and (m["drift"] > B.PASS["drift"] or m["known_lost"] > B.PASS["known_drop"] * m["known_n"]):
                print(f"{ARM}: STOP, {f['id']} broke a damage limit of the smoke facts (plan.md)", flush=True)
                break
    finally:
        block.detach()


if __name__ == "__main__":
    main()

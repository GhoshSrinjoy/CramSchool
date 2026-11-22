"""Step 17 (lab notebook): where is the question readable? A fact's question detector is rebuilt at 42 sites, with the build's own LDA and
threshold rules, and scored at the end of stored test prompts. Nothing is trained, and no fit is changed: only where the reading is taken.

Sites: the MLP input of layers 0 to 20, as the state at the token itself ("last"; layer 20's is the build's own site) and as the running
mean of that layer's MLP input over the text so far, the first token left out ("mean"). Each site has its own calibration (mean,
covariance and constant direction on the calibration's fitting paragraphs).

Rules a detector is fitted by: R1, the build's (negatives: the prompt negatives, the readings inside the taught answers and inside the
unedited model's replies; threshold above the prompt negatives' 99th percentile and every reading inside the taught answers), for "last";
R2, the prompt's end only (the prompt negatives alone, threshold at their 99th percentile), for "last" and "mean". 63 readings in all.

Every prompt is read once for all sites, in batches of 64 with right padding, so scores can differ from the stored ones in their last
digits; the report checks that layer 20, last, R1 agrees with the saved detectors and the stored scores before anything else is read.

Modes:
  wordings  bench40e or bench40f's saved API table: at every reading the detector as built, with every test wording taught (in-sample)
            and with one test wording left out, scored on the test recall prompts and the questions about other relations (step 15's).
  jobs      step 12's split on step 11's live table: the chief-executive facts' detectors as built at every reading, and step 16's veto
            fitted at every site, scored on step 12's prompts.
  near      a bench with near questions (benchmix1): the detectors as built at every reading, scored on recall, same-subject and near
            prompts, with what the stored replies gave.

Usage:
  python scripts/diag_question_site.py wordings <model> <bench json> <table folder> <recall_misses json> <out json> [--only f01,f02]
  python scripts/diag_question_site.py jobs <model> <table folder> <step 12 replies json> <editor's last check jsonl> <out json>
  python scripts/diag_question_site.py near <model> <bench json> <table folder> <saved rows jsonl> <unedited rows jsonl> <out json> [--only ...]
  python scripts/diag_question_site.py veto_cost <model> <table folder> <out json> <bench json> <saved rows jsonl> [<bench json> <rows> ...]
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.bench.scoring import leak_args, load_bench, read_rows, test_prompts
from cramschool.facts import PREFIXES
from cramschool.gates import leaks, mentions
from cramschool.graft import build as B
from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_MAX_TOKENS, greedy_ids, keyed_spots, lda, mlp_of, ratio
from cramschool.graft.wordings import calibration_paragraphs, norm
from cramschool.model_access import Prompt, batch_ids, encode, model_device, render_chat

LAYERS = list(range(21))
SITES = [(k, l) for k in ("last", "mean") for l in LAYERS]
READINGS = [("last", "R1", l) for l in LAYERS] + [("last", "R2", l) for l in LAYERS] + [("mean", "R2", l) for l in LAYERS]
KIND = dict(last=0, mean=1)
BATCH, NEG_Q, SAT, RIDGE, BUDGET = 64, 0.99, 0.5, 0.1, 5 * 2 ** 30
TAUGHT = dict(plain=["{s}'s head of marketing is", "The head of research at {s} is"],
              chat=["Who is the chief technology officer of {s}?", "Who is the chief financial officer of {s}?", "Who is the spokesperson for {s}?"])
TEST = ["Who is the general counsel of {s}?", "Who is the head of human resources at {s}?", "The chief engineer of {s} is"]  # step 12's


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class _Stop(Exception):
    pass


# ---------------------------------------------------------------- reading every site in one pass

@torch.no_grad()
def site_reads(model, tok, seqs, picks, mean):
    """For each token sequence and each picked position: the MLP input of every layer in LAYERS at that position and, with `mean`, the
    running mean of it over positions 1 to that one. One float16 CPU tensor per sequence, (picks, 2 or 1, layers, d)."""
    device = model_device(model)
    out, seen, hooks = [None] * len(seqs), {}, []
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))

    def grab(layer):
        def hook(_m, args):
            seen[layer] = args[0]
            if layer == LAYERS[-1]:
                raise _Stop
        return hook

    for layer in LAYERS:
        hooks.append(mlp_of(model, layer).register_forward_pre_hook(grab(layer)))
    try:
        for start in range(0, len(order), BATCH):
            chunk = order[start:start + BATCH]
            ids, mask = batch_ids(tok, [seqs[i] for i in chunk], device)
            try:
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            except _Stop:
                pass
            x = torch.stack([seen[layer] for layer in LAYERS], 2).float()  # (batch, tokens, layers, d)
            run = x.cumsum(1) if mean else None
            for row, i in enumerate(chunk):
                p = torch.tensor(picks[i], device=device)
                last = x[row, p]
                if mean:  # the mean over positions 1..p; position 0 alone when p is 0
                    m = (run[row, p] - x[row, 0][None]) / p.clamp(min=1).float()[:, None, None]
                    m = torch.where((p == 0)[:, None, None], x[row, 0][None].expand_as(m), m)
                    last = torch.stack([last, m], 1)
                else:
                    last = last[:, None]
                out[i] = last.clamp(-6e4, 6e4).half().cpu()
    finally:
        for h in hooks:
            h.remove()
    return out


class Store:
    """Site readings by (token sequence, position) and greedy replies by prompt, kept across facts: most of a fact's prompts name invented
    subjects, and every fact of a relation asks the same ones. Readings of prompts that name the fact's own subject are dropped after it."""

    def __init__(self, model, tok):
        self.model, self.tok, self.device = model, tok, model_device(model)
        self.rows, self.replies, self.transient, self.bytes = {}, {}, set(), 0

    def _get(self, seqs, picks, mean, transient):
        need = {}
        for s, ps in zip(seqs, picks):
            key = tuple(s)
            for p in ps:
                if (key, p) not in self.rows:
                    need.setdefault(key, set()).add(p)
        if need:
            keys = list(need)
            wanted = [sorted(need[k]) for k in keys]
            for k, ps, got in zip(keys, wanted, site_reads(self.model, self.tok, [list(k) for k in keys], wanted, mean)):
                for j, p in enumerate(ps):
                    self.rows[(k, p)] = got[j].clone()
                    self.bytes += got[j].numel() * 2
        rows = []
        for s, ps, tr in zip(seqs, picks, transient):
            key = tuple(s)
            for p in ps:
                rows.append(self.rows[(key, p)])
                if tr:
                    self.transient.add((key, p))
        return torch.stack(rows).to(self.device) if rows else None

    def ends(self, prompts, subject):
        """The readings at the end of each prompt: (prompts, 2, layers, d) on the GPU."""
        seqs = [encode(self.tok, p) for p in prompts]
        return self._get(seqs, [[len(s) - 1] for s in seqs], True, [subject.lower() in p.text.lower() for p in prompts])

    def at(self, seqs, picks, transient):
        """The "last" readings at the picked positions: (positions, 1, layers, d) on the GPU."""
        return self._get(seqs, picks, False, transient)

    def reply_reads(self, prompts, subject):
        """The "last" readings at every position of the unedited model's greedy replies to the prompts, as the build reads them."""
        keys = [(p.text, p.rendered) for p in prompts]
        need = list({k: p for k, p in zip(keys, prompts) if k not in self.replies}.items())
        fresh = {}
        if need:
            for (k, p), gen in zip(need, greedy_ids(self.model, self.tok, [p for _, p in need], B.REPLY_TOKENS, batch=BATCH)):
                fresh[k] = gen
                if subject.lower() not in p.text.lower():
                    self.replies[k] = gen
        gens = [(fresh[k] if k in fresh else self.replies[k], subject.lower() in p.text.lower()) for k, p in zip(keys, prompts)]
        gens = [(g, tr) for g, tr in gens if g[1]]
        if not gens:
            return None
        return self.at([p + r for (p, r), _ in gens], [list(range(len(p), len(p) + len(r))) for (p, r), _ in gens], [tr for _, tr in gens])

    def end_fact(self):
        for k in self.transient:
            row = self.rows.pop(k, None)
            if row is not None:
                self.bytes -= row.numel() * 2
        self.transient = set()

    def flush_if_full(self):
        if self.bytes > BUDGET:
            self.rows, self.replies, self.bytes = {}, {}, 0


def site_calibration(model, tok):
    """Each site's mean, covariance and constant direction on the calibration's fitting paragraphs, every token but the first, as
    core.calibration_stats and core.constant_direction compute them for the build's site."""
    paras = calibration_paragraphs()
    seqs = [tok(paras[i], truncation=True, max_length=CALIBRATION_MAX_TOKENS).input_ids for i in CALIBRATION_FIT]
    seqs = [s for s in seqs if len(s) > 1]
    device = model_device(model)
    d = model.config.get_text_config().hidden_size
    sums = torch.zeros(2, len(LAYERS), d, dtype=torch.float64, device=device)
    outer = torch.zeros(2, len(LAYERS), d, d, dtype=torch.float64, device=device)
    n = 0
    for start in range(0, len(seqs), 16):
        chunk = seqs[start:start + 16]
        x = torch.cat(site_reads(model, tok, chunk, [list(range(1, len(s))) for s in chunk], True)).to(device)
        n += x.shape[0]
        for k in range(2):
            for li in range(len(LAYERS)):
                xs = x[:, k, li].double()
                sums[k, li] += xs.sum(0)
                outer[k, li] += xs.T @ xs
    mu = sums / n
    cov = outer / n - mu[..., :, None] * mu[..., None, :]
    eye = torch.eye(d, dtype=torch.float64, device=device)
    c = torch.linalg.solve(cov + 1e-3 * cov.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye, mu[..., None])[..., 0]
    return dict(mu=mu, cov=cov, c=c.float(), cbar=(mu * c).sum(-1))


# ---------------------------------------------------------------- one fact's readings

def fact_sets(store, f, ctx, tok):
    """The build's sets for one fact's detectors (build.build_parts), read at every site: the taught positives by form, the prompt
    negatives, the positions inside the taught answers and the positions of the unedited model's replies."""
    subj = f["subject"]
    pos_plain, pos_chat, neg, replies = B.detector_data(f, tok, ctx)
    if B.avoid_items(f, tok, ctx):
        raise NotImplementedError("a relation with must-not-answer wordings is not read here")
    built = B.training_fact(f, ctx["drafted"], ctx.get("teach_cap"))
    examples = [(p, a) for p, a in built.teaching_examples(tok) if norm(p.text if not p.rendered else B._question_of(p, built)) not in ctx["guard"]]
    sequences, keys, spots = keyed_spots(tok, examples)
    picks = [[] for _ in sequences]
    for j, k in enumerate(keys):
        if k[0] > 0:
            for w, p in spots[j]:
                picks[w].append(p)
    used = [(s, p) for s, p in zip(sequences, picks) if p]
    return dict(pos=dict(plain=store.ends(pos_plain, subj), chat=store.ends(pos_chat, subj)), neg=store.ends(neg, subj),
                inside=store.at([s for s, _ in used], [p for _, p in used], [True] * len(used)) if used else None,
                reply=store.reply_reads(replies, subj),
                counts=dict(pos_plain=len(pos_plain), pos_chat=len(pos_chat), neg=len(neg), replies=len(replies)))


def block(x, mu):
    """(rows, sum, Gram matrix) of a block of readings, centred on the site's calibration mean, in float64."""
    xc = x.double() - mu
    return x.shape[0], xc.sum(0), xc.T @ xc


def scatter(blocks):
    """The within-group scatter, the rows and the mean of the group means of negative groups, from their blocks."""
    s = sum(g - torch.outer(v, v) / n for n, v, g in blocks)
    return s, sum(n for n, _, _ in blocks), torch.stack([v / n for n, v, _ in blocks]).mean(0)


def fit_many(taught, extras, unions, neg_scatter, cov, x_neg, floors, cal):
    """core.lda and the build's threshold rule for several sets of positives at once, each the taught block plus some of the extra blocks
    (`unions`: one list of extra indices per detector). taught and each extra: dict(x, block). Returns w (d, detectors), tau and delta
    (detectors; delta is nan where the positives' median doesn't clear tau)."""
    c, cbar = cal
    s_neg, n_neg, m_neg = neg_scatter
    d = cov.shape[0]
    eye = torch.eye(d, dtype=torch.float64, device=cov.device)
    systems, rhs = [], []
    for u in unions:
        parts = [taught["block"]] + [extras[k]["block"] for k in u]
        n, v, g = sum(p[0] for p in parts), sum(p[1] for p in parts), sum(p[2] for p in parts)
        sm = cov + (g - torch.outer(v, v) / n + s_neg) / (n + n_neg)
        systems.append(sm + RIDGE * sm.diagonal().mean() * eye)
        rhs.append(v / n - m_neg)
    w = torch.linalg.solve(torch.stack(systems).float(), torch.stack(rhs).float()[..., None])[..., 0].T  # (d, detectors)
    r_neg = (x_neg @ w) / (x_neg @ c / cbar)[:, None]
    tau = torch.quantile(r_neg, NEG_Q, dim=0)
    for fl in floors:
        tau = torch.maximum(tau, ((fl @ w) / (fl @ c / cbar)[:, None]).max(0).values)
    xs = [taught["x"]] + [e["x"] for e in extras]
    r_pos = torch.cat([(x @ w) / (x @ c / cbar)[:, None] for x in xs])
    owner = torch.cat([torch.full((x.shape[0],), i - 1, device=w.device) for i, x in enumerate(xs)])  # -1: the taught block
    member = torch.stack([(owner == -1) | torch.isin(owner, torch.tensor(u, dtype=torch.long, device=w.device)) for u in unions], 1)
    med = torch.nanmedian(torch.where(member, r_pos, torch.full_like(r_pos, float("nan"))), dim=0).values
    delta = torch.where(med > tau, SAT * (med - tau), torch.full_like(med, float("nan")))
    return w, tau, delta


def scores(x, w, tau, delta, cal):
    """z of each reading under each detector; -1e9 where a detector doesn't exist."""
    c, cbar = cal
    z = ((x @ w) / (x @ c / cbar)[:, None] - tau) / delta
    return torch.where(torch.isnan(z), torch.full_like(z, -1e9), z)


def site_x(t, kind, li):
    return None if t is None else t[:, KIND[kind] if t.shape[1] == 2 else 0, li].float()


def fit_fact(sets, calib, tests, extras=None, saved=None):
    """Every reading's detectors for one fact, scored at the end of `tests` (name -> readings). Without `extras`: as built. With them (per
    test wording, per form, the readings of its taught items or None): also in-sample and with each wording left out.
    Returns {fit: {name: [reading][prompt] best score over the two forms}}, and the gate at layer 20, last, R1 when `saved` is given."""
    n_w = len(extras) if extras is not None else 0
    fits = ["asbuilt"] + (["insample", "lowo"] if n_w else [])
    out = {fit: {name: [None] * len(READINGS) for name in tests} for fit in fits}
    gate = {}
    for kind, li in [(k, l) for k in ("last", "mean") for l in range(len(LAYERS))]:
        mu, cov = calib["mu"][KIND[kind], li], calib["cov"][KIND[kind], li]
        cal = (calib["c"][KIND[kind], li], float(calib["cbar"][KIND[kind], li]))
        x_neg = site_x(sets["neg"], kind, li)
        groups = dict(neg=block(x_neg, mu))
        inside = reply = None
        if kind == "last":
            inside, reply = site_x(sets["inside"], kind, li), site_x(sets["reply"], kind, li)
            for name, x in (("inside", inside), ("reply", reply)):
                if x is not None:
                    groups[name] = block(x, mu)
        x_tests = {name: site_x(t, kind, li) for name, t in tests.items()}
        forms = {}
        for form in ("plain", "chat"):
            taught = dict(x=site_x(sets["pos"][form], kind, li))
            taught["block"] = block(taught["x"], mu)
            ex = [dict(x=site_x(e[form], kind, li)) for e in (extras or []) if e[form] is not None]
            for e in ex:
                e["block"] = block(e["x"], mu)
            index = [k for k, e in enumerate(extras or []) if e[form] is not None]  # which test wording each extra block belongs to
            forms[form] = (taught, ex, index)
        for rule in (("R1", "R2") if kind == "last" else ("R2",)):
            r = READINGS.index((kind, rule, LAYERS[li]))
            negs = [groups["neg"]] + ([groups[g] for g in ("inside", "reply") if g in groups] if rule == "R1" else [])
            floors = [inside] if rule == "R1" and inside is not None else []
            best = {fit: {name: None for name in tests} for fit in fits}
            for form, (taught, ex, index) in forms.items():
                everything = list(range(len(ex)))
                unions = [[]] + ([everything] + [[k for k in everything if index[k] != j] for j in range(n_w)] if n_w else [])
                w, tau, delta = fit_many(taught, ex, unions, scatter(negs), cov, x_neg, floors, cal)
                if saved is not None and (kind, rule, LAYERS[li]) == ("last", "R1", 20):
                    real = lda(taught["x"], [x_neg] + [x for x in (inside, reply) if x is not None], cov.float())
                    gate[form] = dict(cos_saved=float(torch.nn.functional.cosine_similarity(w[:, 0], saved[form]["w"].to(w.device).float(), dim=0)),
                                      cos_lda=float(torch.nn.functional.cosine_similarity(w[:, 0], real, dim=0)),
                                      tau=(float(tau[0]), max(saved[form]["tau_neg"], saved[form]["tau_in"])))
                for name, x in x_tests.items():
                    z = scores(x, w, tau, delta, cal)  # (prompts, detectors)
                    per = dict(asbuilt=z[:, 0])
                    if n_w:
                        per["insample"] = z[:, 1]
                        lo = z[:, 2:]
                        # recall: prompt j under the fit that left wording j out; other groups: open if any left-out fit opens them
                        per["lowo"] = lo.diagonal() if name == "recall" else lo.max(1).values
                    for fit, v in per.items():
                        best[fit][name] = v if best[fit][name] is None else torch.maximum(best[fit][name], v)
            for fit in fits:
                for name in tests:
                    out[fit][name][r] = [round(float(v), 4) for v in best[fit][name]]
    return out, gate


def context_of(graft, f):
    o = graft.options
    return B.build_context([f], graft.wordings, name="table", guard=False, real_world=o.real_world, context=o.context, cache=None,
                           inside=o.inside, twins=o.twins, teach_cap=o.teach_cap, avoid_as=o.avoid_as)


def items_of(template, form, f, ctx, tok, rel):
    """The build's taught items of one wording of the relation (detector_data), by detector form (step 15's)."""
    inv = B.kind_parts(ctx, rel)[0]
    subj = f["subject"]
    cases = [subj.lower(), subj.upper()] + B.case_names(inv[:B.N_CASE])
    if form == "plain":
        return dict(plain=B.plain_items([template], inv) + B.plain_items([template], [subj], PREFIXES + ctx.get("context", ())) + B.plain_items([template], cases),
                    chat=[])
    return dict(plain=B.qa_items([template], inv + [subj]) + B.qa_items([template], cases), chat=B.chat_items(tok, [template], inv + [subj] + cases))


def load(model_dir, table_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    graft = Graft.load(table_dir, model, tok)
    if graft.options.inside != "negative" or graft.options.context:
        raise NotImplementedError("the table's options (inside, context) are not read here")
    t0 = time.time()
    calib = site_calibration(model, tok)
    mctx = graft.calibration()
    k20 = LAYERS.index(graft.layer)
    cos = float(torch.nn.functional.cosine_similarity(calib["c"][0, k20], mctx["c"].to(calib["c"].device).float(), dim=0))
    log(f"calibrated {len(SITES)} sites in {time.time() - t0:.0f}s | layer {graft.layer}, last: constant direction cosine with the table's "
        f"{cos:.6f}, mean reading {float(calib['cbar'][0, k20]):.4f} against the table's {float(mctx['cbar']):.4f}")
    return tok, model, graft, calib


def dump(path, meta, rows):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(meta, readings=[list(r) for r in READINGS], rows=rows), fh)


# ---------------------------------------------------------------- the three modes

def wordings(argv):
    model_dir, bench_path, table_dir, misses_path, out_path = argv[:5]
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    bench = {f["subject"] + "|" + f["relation"]: f for f in load_bench(bench_path)}
    with open(misses_path, encoding="utf-8") as fh:
        stored_by = {}
        for r in json.load(fh)["recall"]:
            stored_by.setdefault(r["id"], []).append(r)
    tok, model, graft, calib = load(model_dir, table_dir)
    store = Store(model, tok)
    entries = sorted(graft._facts, key=lambda e: e["fact"]["relation"])
    entries = [e for e in entries if not only or bench[e["fact"]["subject"] + "|" + e["fact"]["relation"]]["id"] in only]
    rows, relation = [], None
    for n, e in enumerate(entries, 1):
        f = e["fact"]
        bf = bench[f["subject"] + "|" + f["relation"]]
        if f["relation"] != relation:
            store.flush_if_full()
            relation = f["relation"]
        t0 = time.time()
        ctx = context_of(graft, f)
        sets = fact_sets(store, f, ctx, tok)
        prompts = test_prompts(bf, tok)
        tests = dict(recall=store.ends(prompts["recall"], f["subject"]), same=store.ends(prompts["same"], f["subject"]))
        templates = [(t, "plain") for t in bf["test"]["templates_plain"]] + [(t, "chat") for t in bf["test"]["templates_chat"]]
        extras = []
        for t, form in templates:
            it = {k: B.guarded(v, ctx["guard"]) for k, v in items_of(t, form, f, ctx, tok, f["relation"]).items()}
            extras.append({k: (store.ends(v, f["subject"]) if v else None) for k, v in it.items()})
        t_read = time.time() - t0
        z, gate = fit_fact(sets, calib, tests, extras, saved=e["parts"]["forms"])
        by_t = {(r["template"], r["form"]): r for r in stored_by.get(bf["id"], [])}
        st = [by_t[k] for k in templates]
        rows.append(dict(id=bf["id"], relation=f["relation"], gate=gate, stored_kind=[r["kind"] for r in st], stored_z=[r["z_end_best"] for r in st],
                         z={fit: v["recall"] for fit, v in z.items()}, z_same={fit: v["same"] for fit, v in z.items()},
                         counts=sets["counts"], seconds=round(time.time() - t0, 1)))
        dump(out_path, dict(mode="wordings", bench=bench_path, table=table_dir), rows)
        r20 = READINGS.index(("last", "R1", 20))
        agree = sum((a > 0) == (b > 0) for a, b in zip(z["asbuilt"]["recall"][r20], rows[-1]["stored_z"]))
        log(f"[{n}/{len(entries)}] {bf['id']} {f['relation']:<7} gate cos {min(g['cos_saved'] for g in gate.values()):.4f} (fast fit against "
            f"core.lda {min(g['cos_lda'] for g in gate.values()):.6f}) | open or shut as stored on {agree}/{len(st)} | read {t_read:.0f}s, "
            f"all {rows[-1]['seconds']:.0f}s | kept readings {store.bytes / 2 ** 30:.1f} GB")
        store.end_fact()
        del sets, tests, extras
        torch.cuda.empty_cache()


def jobs(argv):
    model_dir, table_dir, replies_path, check_path, out_path = argv[:5]
    with open(replies_path, encoding="utf-8") as fh:
        before = json.load(fh)["before"]
    with open(check_path, encoding="utf-8") as fh:
        checks = [json.loads(line) for line in fh if line.strip()]
    recall_q = {r["subject"]: r["args"]["new_questions"] for r in checks if r["relation"] == "ceo"}
    tok, model, graft, calib = load(model_dir, table_dir)
    store = Store(model, tok)
    entries = [e for e in graft._facts if e["fact"]["relation"] == "ceo"]
    is_chat = lambda q: q.rstrip().endswith(("?", "."))  # noqa: E731  (step 12's rule)
    as_prompt = lambda q: render_chat(tok, q) if is_chat(q) else Prompt(q)  # noqa: E731

    # step 16's veto at every site: the taught jobs against half of the relation's own wordings, thresholded on the other half
    ctx0 = context_of(graft, entries[0]["fact"])
    inv = B.kind_parts(ctx0, "ceo")[0]
    names = inv + B.case_names(inv[:B.N_CASE])
    own = graft.wordings.relations["ceo"]
    halves = dict(fit=(own.plain[0::2], own.chat[0::2]), held=(own.plain[1::2], own.chat[1::2]))
    plain_of = lambda pl, ch: B.guarded(B.plain_items(pl, names) + B.qa_items(ch, names), set())  # noqa: E731
    chat_of = lambda ch: B.guarded(B.chat_items(tok, ch, names), set())  # noqa: E731
    v_sets = dict(plain=dict(pos=store.ends(plain_of(TAUGHT["plain"], TAUGHT["chat"]), "\x00"), **{h: store.ends(plain_of(pl, ch), "\x00") for h, (pl, ch) in halves.items()}),
                  chat=dict(pos=store.ends(chat_of(TAUGHT["chat"]), "\x00"), **{h: store.ends(chat_of(ch), "\x00") for h, (_, ch) in halves.items()}))
    veto = {}
    for kind, li in [(k, l) for k in ("last", "mean") for l in range(len(LAYERS))]:
        cov = calib["cov"][KIND[kind], li].float()
        c, cbar = calib["c"][KIND[kind], li], float(calib["cbar"][KIND[kind], li])
        for form in ("plain", "chat"):
            pos, fit, held = (site_x(v_sets[form][k], kind, li) for k in ("pos", "fit", "held"))
            w = lda(pos, [fit], cov)
            tau = float(torch.quantile(ratio(held, w, c, cbar), NEG_Q))
            med = float(ratio(pos, w, c, cbar).median())
            veto[(kind, li, form)] = (w, tau, SAT * (med - tau)) if med > tau else None
    log(f"veto fitted at {len(SITES)} sites; usable forms: {sum(v is not None for v in veto.values())} of {len(veto)}")

    rows, gates = [], {}
    for n, e in enumerate(entries, 1):
        f = e["fact"]
        t0 = time.time()
        ctx = context_of(graft, f)
        sets = fact_sets(store, f, ctx, tok)
        groups = dict(test=[q.format(s=f["subject"]) for q in TEST], taught=[q.format(s=f["subject"]) for q in TAUGHT["plain"] + TAUGHT["chat"]],
                      recall=list(recall_q[f["subject"]]))
        tests = {g: store.ends([as_prompt(q) for q in qs], f["subject"]) for g, qs in groups.items()}
        z, gate = fit_fact(sets, calib, tests, saved=e["parts"]["forms"])
        for g, qs in groups.items():
            mine = [r for r in before[g] if r["subject"] == f["subject"]]
            assert [r["question"] for r in mine] == qs, f"{g}: not step 12's prompts for {f['subject']}"
            vz = []
            for kind, li in [(k, l) for k in ("last", "mean") for l in range(len(LAYERS))]:
                x = site_x(tests[g], kind, li)
                c, cbar = calib["c"][KIND[kind], li], float(calib["cbar"][KIND[kind], li])
                per = [(ratio(x, v[0], c, cbar) - v[1]) / v[2] if v is not None else torch.full((x.shape[0],), -1e9, device=x.device)
                       for v in (veto[(kind, li, form)] for form in ("plain", "chat"))]
                vz.append(torch.stack(per).max(0).values.tolist())
            for i, (q, r) in enumerate(zip(qs, mine)):
                rows.append(dict(group=g, subject=f["subject"], question=q, hit=bool(r["hit"]),
                                 fact=[z["asbuilt"][g][k][i] for k in range(len(READINGS))], veto=[round(v[i], 4) for v in vz]))
        gates[f["subject"]] = gate
        dump(out_path, dict(mode="jobs", table=table_dir, sites=[list(s) for s in SITES], gates=gates), rows)
        log(f"[{n}/{len(entries)}] {f['subject']:<24} gate cos {min(g['cos_saved'] for g in gate.values()):.4f} | {time.time() - t0:.0f}s")
        store.end_fact()


def veto_cost(argv):
    """After the fact (lab notebook, result of step 17): step 16's veto, fitted at every site as in `jobs`, read at the end of the test
    recall prompts of chief-executive facts in other benches. A veto that fires there would cost the fact its own answers."""
    model_dir, table_dir, out_path = argv[:3]
    tok, model, graft, calib = load(model_dir, table_dir)
    store = Store(model, tok)
    ctx0 = context_of(graft, next(e["fact"] for e in graft._facts if e["fact"]["relation"] == "ceo"))
    inv = B.kind_parts(ctx0, "ceo")[0]
    names = inv + B.case_names(inv[:B.N_CASE])
    own = graft.wordings.relations["ceo"]
    halves = dict(fit=(own.plain[0::2], own.chat[0::2]), held=(own.plain[1::2], own.chat[1::2]))
    plain_of = lambda pl, ch: B.guarded(B.plain_items(pl, names) + B.qa_items(ch, names), set())  # noqa: E731
    chat_of = lambda ch: B.guarded(B.chat_items(tok, ch, names), set())  # noqa: E731
    v_sets = dict(plain=dict(pos=store.ends(plain_of(TAUGHT["plain"], TAUGHT["chat"]), "\x00"), **{h: store.ends(plain_of(pl, ch), "\x00") for h, (pl, ch) in halves.items()}),
                  chat=dict(pos=store.ends(chat_of(TAUGHT["chat"]), "\x00"), **{h: store.ends(chat_of(ch), "\x00") for h, (_, ch) in halves.items()}))
    veto = {}
    for kind, li in [(k, l) for k in ("last", "mean") for l in range(len(LAYERS))]:
        cov = calib["cov"][KIND[kind], li].float()
        c, cbar = calib["c"][KIND[kind], li], float(calib["cbar"][KIND[kind], li])
        for form in ("plain", "chat"):
            pos, fit, held = (site_x(v_sets[form][k], kind, li) for k in ("pos", "fit", "held"))
            w = lda(pos, [fit], cov)
            tau = float(torch.quantile(ratio(held, w, c, cbar), NEG_Q))
            med = float(ratio(pos, w, c, cbar).median())
            veto[(kind, li, form)] = (w, tau, SAT * (med - tau)) if med > tau else None
    rows = []
    for bench_path, rows_path in zip(argv[3::2], argv[4::2]):
        saved = read_rows(rows_path)
        for bf in load_bench(bench_path):
            if bf["relation"] != "ceo":
                continue
            prompts = test_prompts(bf, tok)
            x_all = store.ends(prompts["recall"], "\x00")
            vz = []
            for kind, li in [(k, l) for k in ("last", "mean") for l in range(len(LAYERS))]:
                x = site_x(x_all, kind, li)
                c, cbar = calib["c"][KIND[kind], li], float(calib["cbar"][KIND[kind], li])
                per = [(ratio(x, v[0], c, cbar) - v[1]) / v[2] if v is not None else torch.full((x.shape[0],), -1e9, device=x.device)
                       for v in (veto[(kind, li, form)] for form in ("plain", "chat"))]
                vz.append([round(float(z), 4) for z in torch.stack(per).max(0).values])
            templates = bf["test"]["templates_plain"] + bf["test"]["templates_chat"]
            for i, t in enumerate(templates):
                rows.append(dict(bench=bench_path, id=bf["id"], template=t, whole=bool(mentions(saved[bf["id"]]["answers"]["recall"][i], bf["target"])) if bf["id"] in saved else None,
                                 veto=[v[i] for v in vz]))
        log(f"{bench_path}: {sum(r['bench'] == bench_path for r in rows)} chief-executive test prompts read")
    dump(out_path, dict(mode="veto_cost", table=table_dir, sites=[list(s) for s in SITES]), rows)


def near(argv):
    model_dir, bench_path, table_dir, rows_path, base_path, out_path = argv[:6]
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    bench = {f["subject"] + "|" + f["relation"]: f for f in load_bench(bench_path)}
    saved_rows, base_rows = read_rows(rows_path), read_rows(base_path)
    tok, model, graft, calib = load(model_dir, table_dir)
    store = Store(model, tok)
    entries = sorted(graft._facts, key=lambda e: e["fact"]["relation"])
    entries = [e for e in entries if not only or bench[e["fact"]["subject"] + "|" + e["fact"]["relation"]]["id"] in only]
    rows, relation = [], None
    for n, e in enumerate(entries, 1):
        f = e["fact"]
        bf = bench[f["subject"] + "|" + f["relation"]]
        if f["relation"] != relation:
            store.flush_if_full()
            relation = f["relation"]
        t0 = time.time()
        sets = fact_sets(store, f, context_of(graft, f), tok)
        prompts = test_prompts(bf, tok)
        groups = [g for g in ("recall", "same", "near") if prompts[g]]
        tests = {g: store.ends(prompts[g], f["subject"]) for g in groups}
        z, gate = fit_fact(sets, calib, tests, saved=e["parts"]["forms"])
        ans, was, how = saved_rows[bf["id"]]["answers"], base_rows[bf["id"]]["answers"], leak_args(bf)
        hit = dict(recall=[mentions(t, bf["target"]) for t in ans["recall"]])
        for g in groups[1:]:
            hit[g] = [bool(leaks(t, bf["target"], p.text, b, **how)) for t, p, b in zip(ans[g], prompts[g], was[g])]
        rows.append(dict(id=bf["id"], relation=f["relation"], kind=bf.get("kind"), answer_type=bf.get("answer_type"), gate=gate, hit=hit,
                         z={g: z["asbuilt"][g] for g in groups}, passed=saved_rows[bf["id"]]["passed"], seconds=round(time.time() - t0, 1)))
        dump(out_path, dict(mode="near", bench=bench_path, table=table_dir), rows)
        r20 = READINGS.index(("last", "R1", 20))
        log(f"[{n}/{len(entries)}] {bf['id']} {f['relation']:<18} gate cos {min(g['cos_saved'] for g in gate.values()):.4f} | at the build's site: "
            f"open on {sum(v > 0 for v in z['asbuilt']['recall'][r20])}/{len(hit['recall'])} recall prompts ({sum(hit['recall'])} answered), "
            f"on {sum(v > 0 for v in z['asbuilt'].get('near', [[]] * len(READINGS))[r20])}/{len(hit.get('near', []))} near questions "
            f"({sum(hit.get('near', []))} leaking) | {rows[-1]['seconds']:.0f}s | kept {store.bytes / 2 ** 30:.1f} GB")
        store.end_fact()
        del sets, tests
        torch.cuda.empty_cache()


if __name__ == "__main__":
    {"wordings": wordings, "jobs": jobs, "near": near, "veto_cost": veto_cost}[sys.argv[1]](sys.argv[2:])

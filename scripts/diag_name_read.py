"""Step 13 of the lab notebook: the name read alone, where a one-head subject key would read it.

For each company of bench40e, bench40f and bench40g, a reading of the token mixer's input (the output of the layer's input norm) is fitted
at layers 16, 17, 18 and 20 to fire at the company's name and to stay shut everywhere else:
- A: the state at one token;
- B: the states at the token and the 3 before it, the reach of a DeltaNet head's 4-token convolution;
- C: A and B both over their thresholds.
Fitted on the training wordings and names (LDA with shrinkage, threshold at the largest negative reading), scored on the test wordings, the
test look-alikes, other stored companies and held-out ordinary text. Forward passes and linear solves only.

    python scripts/diag_name_read.py --out research/results/slots/21_name_read
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_MAX_TOKENS, RIDGE, blocks
from cramschool.graft.wordings import calibration_paragraphs
from cramschool.model_access import batch_ids

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("CRAMSCHOOL_MODEL", "models/Qwen3.5-0.8B")  # the model's local folder; CRAMSCHOOL_MODEL names another
LAYERS = (16, 17, 18, 20)
BENCHES = ("bench40e", "bench40f", "bench40g")
WIN = 4
HELD = range(240, 355)  # calibration paragraphs never used to fit: CALIBRATION_HELD and the compile's spare paragraphs
MSMARCO = os.environ.get("CRAMSCHOOL_MSMARCO", "datasets/msmarco/corpus/corpus-00000-of-00001.parquet")
MS_SKIP, MS_TOKENS = 500_000, 100_000  # passages far past the slice step 6 fitted on
CTX, NAME_LAST, NAME_OTHER, LA, UN = 0, 1, 2, 3, 4
FILLS = ("subj", "la", "un", "lower", "upper")
RULE = dict(readable=81, recall=0.95, lookalike=1 / 12, text=0.001)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- prompts

def companies_of(benches: list[str], limit: int | None) -> list[dict]:
    """One record per company: its facts' training and test wordings (plain and chat), its questions about other relations, its look-alikes
    and unrelated companies (shared by its facts)."""
    out = []
    for b in benches:
        with open(os.path.join(ROOT, "benchmarks", f"{b}.json"), encoding="utf-8") as fh:
            facts = json.load(fh)["facts"]
        by = {}
        for f in facts:
            by.setdefault(f["subject"], []).append(f)
        for subject, fs in list(by.items())[:limit]:
            rec = dict(bench=b, subject=subject, facts=[f["id"] for f in fs])
            for split in ("train", "test"):
                t = [(x, "plain") for f in fs for x in f[split]["templates_plain"]] + [(x, "chat") for f in fs for x in f[split]["templates_chat"]]
                s = [(x, "plain") for f in fs for x in f[split]["same_plain"]] + [(x, "chat") for f in fs for x in f[split]["same_chat"]]
                rec[split] = dict(templates=list(dict.fromkeys(t)), same=list(dict.fromkeys(s)),
                                  lookalikes=list(dict.fromkeys(x for f in fs for x in f[split]["lookalikes"])),
                                  others=list(dict.fromkeys(x for f in fs for x in f[split]["others"])))
            out.append(rec)
    return out


def render(tokenizer, text: str, form: str) -> tuple[str, bool]:
    if form == "plain":
        return text, False
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False), True


def tokenized(tokenizer, text: str, rendered: bool, name: str, role: int):
    """Token ids, and each token's role: the tokens of `name`'s last occurrence (for a subject, the last one marked), else context."""
    enc = tokenizer(text, add_special_tokens=not rendered, return_offsets_mapping=True)
    ids, offs = enc.input_ids, enc.offset_mapping
    roles = [CTX] * len(ids)
    at = text.rfind(name)
    if at < 0:
        return ids, roles, False
    end = at + len(name)
    inside = [i for i, (s, e) in enumerate(offs) if s < end and e > at and e > s]
    if not inside:
        return ids, roles, False
    for i in inside:
        roles[i] = role if role in (LA, UN) else NAME_OTHER
    if role not in (LA, UN):
        roles[inside[-1]] = NAME_LAST
    return ids, roles, True


def build_prompts(tokenizer, companies: list[dict]) -> tuple[list[dict], dict]:
    """Every prompt: ids, per-token roles, its company, its fill (subj, la, un, lower, upper), which look-alike, its split."""
    prompts, dropped = [], {}

    def add(k, split, fill, text, form, name, role, which=-1):
        t, rendered = render(tokenizer, text, form)
        ids, roles, ok = tokenized(tokenizer, t, rendered, name, role)
        if not ok:
            dropped[fill] = dropped.get(fill, 0) + 1
            if fill in ("subj", "lower", "upper"):
                return  # a positive whose name can't be found in the text is left out
        prompts.append(dict(ids=ids, roles=roles, comp=k, fill=fill, split=split, which=which, form=form))

    for k, c in enumerate(companies):
        s = c["subject"]
        for split in ("train", "test"):
            d = c[split]
            for tpl, form in d["templates"]:
                add(k, split, "subj", tpl.replace("{s}", s), form, s, NAME_LAST)
                for j, la in enumerate(d["lookalikes"]):
                    add(k, split, "la", tpl.replace("{s}", la), form, la, LA, j)
                for j, un in enumerate(d["others"]):
                    add(k, split, "un", tpl.replace("{s}", un), form, un, UN, j)
                if split == "test":
                    add(k, split, "lower", tpl.replace("{s}", s.lower()), form, s.lower(), NAME_LAST)
                    add(k, split, "upper", tpl.replace("{s}", s.upper()), form, s.upper(), NAME_LAST)
            for text, form in d["same"]:
                add(k, split, "subj", text, form, s, NAME_LAST)
    return prompts, dropped


def collisions(companies: list[dict]) -> list[str]:
    """Names used for one company's look-alikes or unrelated companies that are another company's subject (these would mislabel tokens)."""
    subjects = {c["subject"].lower(): c["subject"] for c in companies}
    out = []
    for c in companies:
        for split in ("train", "test"):
            for x in c[split]["lookalikes"] + c[split]["others"]:
                if x.lower() in subjects:
                    out.append(f"{c['subject']}: {x}")
    return out


def text_sequences(tokenizer) -> tuple[list[list[int]], list[list[int]]]:
    """Ordinary text: the calibration's fitting paragraphs (to fit), and held-out paragraphs plus MS MARCO passages (to score)."""
    import pyarrow.parquet as pq

    paras = calibration_paragraphs()

    def enc(t):
        return tokenizer(t, truncation=True, max_length=CALIBRATION_MAX_TOKENS).input_ids

    fit = [s for s in (enc(paras[i]) for i in CALIBRATION_FIT) if len(s) > 1]
    held = [s for s in (enc(paras[i]) for i in HELD if i < len(paras)) if len(s) > 1]
    pf, seen, total = pq.ParquetFile(MSMARCO), 0, 0
    for batch in pf.iter_batches(batch_size=20_000, columns=["text"]):
        n = batch.num_rows
        if seen + n <= MS_SKIP:
            seen += n
            continue
        for text in batch.column("text").to_pylist()[max(0, MS_SKIP - seen):]:
            s = enc(text)
            if len(s) > 1:
                held.append(s)
                total += len(s)
            if total >= MS_TOKENS:
                break
        seen += n
        if total >= MS_TOKENS:
            break
    return fit, held


# ---------------------------------------------------------------- forward passes

class _Stop(Exception):
    pass


@torch.inference_mode()
def read_states(model, tokenizer, seqs: list[list[int]], layers, batch: int = 64, label: str = "") -> dict[int, list[torch.Tensor]]:
    """The input norm's output at each layer (the token mixer's input), at every token of each sequence: fp16 on the CPU, in order."""
    device = next(model.parameters()).device
    seen, top = {}, max(layers)

    def grab(L):
        def hook(_m, _a, o):
            seen[L] = o
            if L == top:
                raise _Stop
        return hook

    hooks = [blocks(model)[L].input_layernorm.register_forward_hook(grab(L)) for L in layers]
    out = {L: [None] * len(seqs) for L in layers}
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    t0, steps = time.time(), (len(order) + batch - 1) // batch
    try:
        for n, start in enumerate(range(0, len(order), batch)):
            chunk = order[start:start + batch]
            ids, mask = batch_ids(tokenizer, [seqs[i] for i in chunk], device)
            try:
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            except _Stop:
                pass
            for L in layers:
                h = seen[L].to("cpu", torch.float16)
                for row, i in enumerate(chunk):
                    out[L][i] = h[row, :len(seqs[i])].clone()
            if n % 50 == 0 or n == steps - 1:
                log(f"  reading {label}: batch {n + 1}/{steps} | {time.time() - t0:.0f}s")
    finally:
        for h in hooks:
            h.remove()
    return out


# ---------------------------------------------------------------- one layer

class Site:
    """The tokens of a group of sequences at one layer: single-token features, and 4-token window features made on demand."""

    def __init__(self, parts: list[torch.Tensor], device):
        self.device = device
        self.X = torch.cat(parts).to(device)
        self.n, d = self.X.shape
        self.Xz = torch.cat([self.X, torch.zeros(1, d, dtype=self.X.dtype, device=device)])
        lengths = [p.shape[0] for p in parts]
        self.pos = torch.cat([torch.arange(L) for L in lengths]).to(device)
        g = torch.arange(self.n, device=device)
        self.idx = torch.stack([torch.where(self.pos >= j, g - j, torch.full_like(g, self.n)) for j in range(WIN)], 1)

    def feats(self, rows: torch.Tensor, reader: str) -> torch.Tensor:
        if reader == "A":
            return self.X[rows].float()
        return self.Xz[self.idx[rows]].reshape(rows.numel(), -1).float()

    def row_chunks(self, rows: torch.Tensor | None = None, size: int = 32768):
        rows = torch.arange(self.n, device=self.device) if rows is None else rows
        for s in range(0, rows.numel(), size):
            yield rows[s:s + size]


def moments(site: Site, reader: str, rows: torch.Tensor | None = None):
    """(sum, sum of outer products, count) over the rows, float64; None when there are no rows."""
    S1, S2, n = None, None, 0
    for r in site.row_chunks(rows):
        if r.numel() == 0:
            continue
        F = site.feats(r, reader)
        a, b = F.sum(0).double(), (F.T @ F).double()
        S1, S2, n = (a, b, r.numel()) if S1 is None else (S1 + a, S2 + b, n + r.numel())
    return (S1, S2, n) if n else None


def centred(S1, S2, n):
    return S2 - torch.outer(S1, S1) / n


def any_by_prompt(fires: torch.Tensor, prompt: torch.Tensor, n_prompts: int) -> torch.Tensor:
    acc = torch.zeros(n_prompts, fires.shape[1], dtype=torch.int32, device=fires.device)
    acc.index_add_(0, prompt, fires.int())
    return acc > 0


def analyse_layer(L: int, states: dict, prompts: list[dict], companies: list[dict], device) -> dict:
    t0 = time.time()
    K = len(companies)
    tr = [i for i, p in enumerate(prompts) if p["split"] == "train"]
    te = [i for i, p in enumerate(prompts) if p["split"] == "test"]
    train = Site([states["prompts"][i] for i in tr], device)
    test = Site([states["prompts"][i] for i in te], device)
    fit = Site(states["fit"], device)
    held = Site(states["held"], device)

    def per_token(ix, value):
        return torch.cat([torch.full((len(prompts[i]["ids"]),), value(j, i), dtype=torch.long) for j, i in enumerate(ix)]).to(device)

    tr_role = torch.cat([torch.tensor(prompts[i]["roles"]) for i in tr]).to(device)
    tr_comp = per_token(tr, lambda j, i: prompts[i]["comp"])
    te_role = torch.cat([torch.tensor(prompts[i]["roles"]) for i in te]).to(device)
    te_prompt = per_token(te, lambda j, i: j)
    groups_of = {}
    for j, i in enumerate(te):
        p = prompts[i]
        groups_of.setdefault((p["comp"], p["fill"]), []).append(j)
        if p["fill"] == "la":
            groups_of.setdefault((p["comp"], f"la{p['which']}"), []).append(j)
    te_subj_comp = torch.tensor([prompts[i]["comp"] if prompts[i]["fill"] == "subj" else -1 for i in te], device=device)

    # the site's statistics from the fitting paragraphs (first token left out, as the calibration does), and the constant direction
    S1, S2, n = moments(fit, "A", torch.nonzero(fit.pos >= 1)[:, 0])
    mu = S1 / n
    cov = S2 / n - torch.outer(mu, mu)
    eye_d = torch.eye(cov.shape[0], dtype=cov.dtype, device=device)
    c = torch.linalg.solve(cov + 1e-3 * cov.diagonal().mean() * eye_d, mu)
    cbar = float(mu @ c)
    W1, W2, wn = moments(fit, "B", torch.nonzero(fit.pos >= WIN - 1)[:, 0])
    covs = dict(A=cov, B=W2 / wn - torch.outer(W1 / wn, W1 / wn))
    cf = c.float()

    def reads(site, rows, reader, Wm):
        return (site.feats(rows, reader) @ Wm) / ((site.X[rows].float() @ cf) / cbar)[:, None]

    is_name = tr_role != CTX
    own_rows = [torch.nonzero(tr_comp == k)[:, 0] for k in range(K)]
    out = {}
    for reader in ("A", "B"):
        g_ctx = moments(train, reader, torch.nonzero(tr_role == CTX)[:, 0])
        g_names = moments(train, reader, torch.nonzero(is_name)[:, 0])
        g_text = moments(fit, reader)
        dim = covs[reader].shape[0]
        eye = torch.eye(dim, dtype=torch.float64, device=device)
        ws = []
        for k in range(K):
            r = own_rows[k]
            role = tr_role[r]
            pos = moments(train, reader, r[role == NAME_LAST])
            la = moments(train, reader, r[role == LA])
            un = moments(train, reader, r[role == UN])
            mine = moments(train, reader, r[role != CTX])
            names = (g_names[0] - mine[0], g_names[1] - mine[1], g_names[2] - mine[2])
            groups = [g for g in (la, un, names, g_ctx, g_text) if g is not None]
            m_neg = torch.stack([g[0] / g[2] for g in groups]).mean(0)
            within = centred(*pos) + sum(centred(*g) for g in groups)
            N = pos[2] + sum(g[2] for g in groups)
            Sm = covs[reader] + within / N
            ws.append(torch.linalg.solve(Sm + RIDGE * Sm.diagonal().mean() * eye, pos[0] / pos[2] - m_neg).float())
        Wm = torch.stack(ws, 1)

        # the threshold: the largest reading over every negative (training tokens away from the company's own name, and the fitting text)
        tau = torch.full((K,), -float("inf"), device=device)
        pos_vals, pos_comp = [], []
        for rows in train.row_chunks():
            R = reads(train, rows, reader, Wm)
            role, comp = tr_role[rows], tr_comp[rows]
            last = torch.nonzero(role == NAME_LAST)[:, 0]
            pos_vals.append(R[last, comp[last]])
            pos_comp.append(comp[last])
            own = torch.nonzero((role == NAME_LAST) | (role == NAME_OTHER))[:, 0]
            R[own, comp[own]] = -float("inf")
            tau = torch.maximum(tau, R.max(0).values)
        for rows in fit.row_chunks():
            tau = torch.maximum(tau, reads(fit, rows, reader, Wm).max(0).values)
        pv, pc = torch.cat(pos_vals), torch.cat(pos_comp)
        med = torch.stack([pv[pc == k].median() if (pc == k).any() else torch.tensor(-float("inf"), device=device) for k in range(K)])
        readable = med > tau

        # scoring: whether each test token and each held-out token reads over the threshold
        test_tok = torch.cat([reads(test, rows, reader, Wm) > tau[None, :] for rows in test.row_chunks()])
        held_tok = torch.cat([reads(held, rows, reader, Wm) > tau[None, :] for rows in held.row_chunks()])
        out[reader] = dict(tau=tau, med=med, readable=readable, test_tok=test_tok, held_tok=held_tok)
        log(f"  layer {L} reader {reader}: fitted and scored | {time.time() - t0:.0f}s")
    out["C"] = dict(readable=out["A"]["readable"] & out["B"]["readable"], test_tok=out["A"]["test_tok"] & out["B"]["test_tok"],
                    held_tok=out["A"]["held_tok"] & out["B"]["held_tok"], tau=None, med=None)

    # per company: the reading counted at the name's last token (recall, case) or anywhere in the prompt (leaks)
    last = te_role == NAME_LAST
    summary = {}
    for reader, R in out.items():
        fires_any = any_by_prompt(R["test_tok"], te_prompt, len(te))
        fires_last = any_by_prompt(R["test_tok"][last], te_prompt[last], len(te))
        held_rate = R["held_tok"].float().mean(0)
        rows = []
        for k, comp in enumerate(companies):
            ok = bool(R["readable"][k])

            def share(key, F):
                ix = groups_of.get((k, key))
                if not ix:
                    return None
                return float(F[ix, k].float().mean()) if ok else 0.0

            other = (te_subj_comp >= 0) & (te_subj_comp != k)
            rows.append(dict(company=comp["subject"], bench=comp["bench"], readable=ok,
                             tau=None if R["tau"] is None else float(R["tau"][k]), median_positive=None if R["med"] is None else float(R["med"][k]),
                             recall=share("subj", fires_last), lookalike=share("la", fires_any),
                             lookalike_by_kind={str(j): share(f"la{j}", fires_any) for j in range(3)},
                             lookalike_names=companies[k]["test"]["lookalikes"],
                             unrelated=share("un", fires_any), lower=share("lower", fires_last), upper=share("upper", fires_last),
                             other_companies=float(fires_any[other, k].float().mean()) if ok else 0.0,
                             text=float(held_rate[k]) if ok else 0.0))
        summary[reader] = rows
    log(f"layer {L} done | {time.time() - t0:.0f}s")
    del train, test, fit, held, out
    torch.cuda.empty_cache()
    return summary


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def judge(rows: list[dict]) -> dict:
    readable = sum(r["readable"] for r in rows)
    recall, look, text = med([r["recall"] for r in rows]), med([r["lookalike"] for r in rows]), med([r["text"] for r in rows])
    ok = [r for r in rows if r["readable"]]
    return dict(readable=readable, recall=recall, lookalike=look, text=text,
                lookalike_mean=statistics.mean(r["lookalike"] for r in rows if r["lookalike"] is not None),
                lookalike_readable=med([r["lookalike"] for r in ok]), no_lookalike_leak=sum(1 for r in ok if r["lookalike"] == 0),
                unrelated=med([r["unrelated"] for r in rows]), other_companies=med([r["other_companies"] for r in rows]),
                lower=med([r["lower"] for r in rows]), upper=med([r["upper"] for r in rows]), text_max=max(r["text"] for r in rows),
                lookalike_by_kind={j: med([r["lookalike_by_kind"][j] for r in rows]) for j in ("0", "1", "2")},
                by_bench={b: dict(readable=sum(r["readable"] for r in rows if r["bench"] == b),
                                  recall=med([r["recall"] for r in rows if r["bench"] == b]),
                                  lookalike=med([r["lookalike"] for r in rows if r["bench"] == b]))
                          for b in sorted({r["bench"] for r in rows})},
                passes=bool(readable >= RULE["readable"] and recall is not None and recall >= RULE["recall"]
                            and look is not None and look <= RULE["lookalike"] and text is not None and text <= RULE["text"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "results", "slots", "21_name_read"))
    ap.add_argument("--benches", nargs="*", default=list(BENCHES))
    ap.add_argument("--limit", type=int, default=None, help="companies per bench (smoke tests)")
    ap.add_argument("--layers", nargs="*", type=int, default=list(LAYERS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=device).eval()
    companies = companies_of(args.benches, args.limit)
    clash = collisions(companies)
    prompts, dropped = build_prompts(tok, companies)
    fit_seqs, held_seqs = text_sequences(tok)
    n_tr = sum(1 for p in prompts if p["split"] == "train")
    log(f"{len(companies)} companies, {len(prompts)} prompts ({n_tr} training), {sum(len(p['ids']) for p in prompts)} prompt tokens; text: "
        f"{sum(map(len, fit_seqs))} fitting, {sum(map(len, held_seqs))} held-out tokens; names not found: {dropped}; collisions: {clash} "
        f"| {time.time() - t0:.0f}s")
    reads = read_states(model, tok, [p["ids"] for p in prompts], args.layers, label="prompts")
    fits = read_states(model, tok, fit_seqs, args.layers, label="fitting text")
    helds = read_states(model, tok, held_seqs, args.layers, label="held-out text")
    del model
    torch.cuda.empty_cache()
    log(f"forward passes done | {time.time() - t0:.0f}s")
    results, verdicts = {}, {}
    for L in args.layers:
        results[L] = analyse_layer(L, dict(prompts=reads[L], fit=fits[L], held=helds[L]), prompts, companies, device)
        verdicts[L] = {r: judge(rows) for r, rows in results[L].items()}
        for r, v in verdicts[L].items():
            log(f"  layer {L} {r}: readable {v['readable']}/{len(companies)}, recall {v['recall']}, look-alike {v['lookalike']} "
                f"(readable only {v['lookalike_readable']}, mean {v['lookalike_mean']:.3f}, none on {v['no_lookalike_leak']}), text {v['text']} "
                f"(max {v['text_max']:.5f}), other companies {v['other_companies']}, lower/upper {v['lower']}/{v['upper']} "
                f"-> {'PASS' if v['passes'] else 'fail'}")
        reads[L] = fits[L] = helds[L] = None
    passing = [(v["lookalike"], v["text"], L, r) for L, vs in verdicts.items() for r, v in vs.items() if r in ("B", "C") and v["passes"]]
    chosen = min(passing) if passing else None
    decision = dict(rule=RULE, passing=[[L, r] for _, _, L, r in sorted(passing)],
                    chosen=dict(layer=chosen[2], reader=chosen[3]) if chosen else None,
                    verdict="a weights-only name read is viable" if chosen else "kill rule met: the subject key stays the name switch")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(plan="docs/lab-notebook.md, step 13", layers=args.layers, benches=args.benches, companies=len(companies),
                       prompts=len(prompts), dropped=dropped, collisions=clash, verdicts={str(L): v for L, v in verdicts.items()},
                       decision=decision, seconds=round(time.time() - t0)), fh, indent=1)
    with open(os.path.join(args.out, "per_company.json"), "w", encoding="utf-8") as fh:
        json.dump({str(L): v for L, v in results.items()}, fh, indent=1)
    log(f"decision: {decision['verdict']} {decision['chosen'] or ''} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())

"""Is the question detector limited by its fit or by what layer 20 holds? (Lab notebook, step 15.) For each fact of a saved table:

1. Reproduce. The fact is rebuilt with the table's own options while the LDA's inputs are captured (positives, negative groups, covariance).
   The rebuilt detectors must equal the saved ones, and their scores at the end of the bench's test prompts must equal the stored reading
   of the misses (recall_misses.json), before anything else counts.
2. In-sample. Each detector is refitted with the fact's test wordings added as positives, built from each wording exactly as the build
   builds a taught one (invented names, the subject behind the prefixes, capitalisation, Q/A form). Same negatives, same threshold rule.
   An upper bound: it uses test wordings on purpose, and is never shipped or used to draft wordings.
3. Leave one wording out. For each test wording, the refit with every other test wording added, scored on the held-out wording's prompt.
4. Two directions (if asked): the original detector OR a second one fitted on the test wordings alone against the same negatives.

Each fit is scored on the test recall prompts and the test questions about other relations of the same company (openings there are leaks).
Forward passes and closed-form solves only. GPU.

Usage: python scripts/diag_detector_ceiling.py <model folder> <bench json> <table folder> <recall_misses json> <out json> [--only f01,f02]
"""
import json
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.bench.scoring import BATCH, GROUPS, load_bench, test_prompts
from cramschool.gates import generate_batch
from cramschool.graft.core import mlp_of
from cramschool.facts import PREFIXES
from cramschool.graft import build as B
from cramschool.graft.core import lda, ratio

NEG_Q, SAT = 0.99, 0.5


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def detector(x_pos, negs, cov, x_neg, inside, c, cbar, scatter_only=()):
    """The build's fit and threshold rule: LDA, tau above the prompt negatives' 99th percentile and every inside-answer reading, delta half
    the way to the positives' median. None when the median doesn't clear tau."""
    w = lda(x_pos, negs, cov, scatter_only=scatter_only)
    tau = float(torch.quantile(ratio(x_neg, w, c, cbar), NEG_Q))
    if inside is not None:
        tau = max(tau, float(ratio(inside, w, c, cbar).max()))
    med = float(ratio(x_pos, w, c, cbar).median())
    return None if med <= tau else dict(w=w, tau=tau, delta=SAT * (med - tau))


def zs(x, d, c, cbar):
    if d is None:
        return torch.full((x.shape[0],), -float("inf"), device=x.device)
    return (ratio(x, d["w"], c, cbar) - d["tau"]) / d["delta"]


def end_reads(model, tok, prompts, layer):
    """Layer `layer`'s MLP input at the end of the bench's test prompts, read in the call the bench made: its first generation batch (all
    groups in order, 32 at a time, left padding), as the stored reading of the misses was. Returns (recall, same) readings."""
    flat = [p for g in GROUPS for p in prompts[g]]
    n_rec, n_same = len(prompts["recall"]), len(prompts["same"])
    out = []
    for start in range(0, n_rec + n_same, BATCH):
        seen = []
        h = mlp_of(model, layer).register_forward_pre_hook(lambda m, a: seen.append(a[0][:, -1].float()) if not seen else None)
        try:
            generate_batch(model, tok, flat[start:start + BATCH], 1, batch_size=BATCH)
        finally:
            h.remove()
        out.append(seen[0])
    x = torch.cat(out)
    return x[:n_rec], x[n_rec:n_rec + n_same]


def items_of(template, form, f, ctx, tok, rel):
    """The build's taught items of one wording of the relation (detector_data), by detector form."""
    inv = B.kind_parts(ctx, rel)[0]
    subj = f["subject"]
    cases = [subj.lower(), subj.upper()] + B.case_names(inv[:B.N_CASE])
    if form == "plain":
        return dict(plain=B.plain_items([template], inv) + B.plain_items([template], [subj], PREFIXES + ctx.get("context", ())) + B.plain_items([template], cases),
                    chat=[])
    return dict(plain=B.qa_items([template], inv + [subj]) + B.qa_items([template], cases), chat=B.chat_items(tok, [template], inv + [subj] + cases))


def main():
    model_dir, bench_path, table_dir, misses_path, out_path = sys.argv[1:6]
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    bench = {f["subject"] + "|" + f["relation"]: f for f in load_bench(bench_path)}
    with open(misses_path, encoding="utf-8") as fh:
        stored = json.load(fh)["recall"]
    stored_by = {}
    for r in stored:
        stored_by.setdefault(r["id"], []).append(r)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    graft = Graft.load(table_dir, model, tok)
    mctx = graft.calibration()
    c, cbar, layer, opts = mctx["c"], mctx["cbar"], graft.layer, graft.options
    out = []
    for e in graft._facts:
        f = e["fact"]
        bf = bench[f["subject"] + "|" + f["relation"]]
        if only and bf["id"] not in only:
            continue
        t0 = time.time()
        ctx = B.build_context([f], graft.wordings, name="table", guard=False, real_world=opts.real_world, context=opts.context,
                              cache=graft._read_cache(), inside=opts.inside, twins=opts.twins, teach_cap=opts.teach_cap, avoid_as=opts.avoid_as)
        captured = []
        real_lda = B.lda
        B.lda = lambda pos, negs, cov, ridge=0.1, scatter_only=(): captured.append((pos, negs, cov, scatter_only)) or real_lda(pos, negs, cov, ridge, scatter_only)
        try:
            rebuilt = B.build_parts(model, tok, f, ctx, mctx, layer, vectors_from=e["parts"])
        finally:
            B.lda = real_lda
        saved = e["parts"]["forms"]
        gate = {form: dict(cos=float(torch.nn.functional.cosine_similarity(rebuilt["forms"][form]["w"].float(), saved[form]["w"].float(), dim=0)),
                           tau=(rebuilt["forms"][form]["tau_neg"], saved[form]["tau_neg"])) for form in saved}
        fits = dict(zip(("plain", "chat"), captured[:2]))
        prompts = test_prompts(bf, tok)
        x_rec, x_same = end_reads(model, tok, prompts, layer)
        templates = [(t, "plain") for t in bf["test"]["templates_plain"]] + [(t, "chat") for t in bf["test"]["templates_chat"]]
        extra = []  # per test wording: its taught items' readings, by detector form
        for t, form in templates:
            it = items_of(t, form, f, ctx, tok, f["relation"])
            it = {k: B.guarded(v, ctx["guard"]) for k, v in it.items()}  # (text, prompt) pairs to prompts, as the build reads them
            extra.append({k: (B.last_reads(model, tok, v, layer) if v else None) for k, v in it.items()})
        results = dict(original={}, insample={}, lowo={}, second={})
        for form, (x_pos, negs, cov, scat) in fits.items():
            # build_parts' groups: [prompt negatives, inside-answer readings (unless scattered), reply readings, ...]
            n_in = rebuilt["counts"].get("inside", 0)
            x_neg = negs[0]
            inside = scat[0] if scat else (negs[1] if n_in else None)
            assert inside is None or inside.shape[0] == n_in, (inside.shape, n_in)
            orig = detector(x_pos, negs, cov, x_neg, inside, c, cbar, scat)
            more = [x[form] for x in extra if x[form] is not None]
            ins = detector(torch.cat([x_pos] + more), negs, cov, x_neg, inside, c, cbar, scat) if more else orig
            second = detector(torch.cat(more), negs, cov, x_neg, inside, c, cbar, scat) if more else None
            results["original"][form] = (zs(x_rec, orig, c, cbar), zs(x_same, orig, c, cbar))
            results["insample"][form] = (zs(x_rec, ins, c, cbar), zs(x_same, ins, c, cbar))
            results["second"][form] = (zs(x_rec, second, c, cbar), zs(x_same, second, c, cbar))
            lo_rec = torch.full((x_rec.shape[0],), -float("inf"), device=x_rec.device)
            lo_same = []
            for j in range(len(templates)):
                held = [x[form] for k, x in enumerate(extra) if k != j and x[form] is not None]
                d = detector(torch.cat([x_pos] + held), negs, cov, x_neg, inside, c, cbar, scat) if held else orig
                lo_rec[j] = zs(x_rec[j:j + 1], d, c, cbar)[0]
                lo_same.append(zs(x_same, d, c, cbar))
            results["lowo"][form] = (lo_rec, torch.stack(lo_same).max(0).values)  # a same-company prompt opens if any held-out fit opens it
        best = {k: (torch.stack([v[f_][0] for f_ in v]).max(0).values.tolist(), torch.stack([v[f_][1] for f_ in v]).max(0).values.tolist())
                for k, v in results.items()}
        best["either"] = ([max(a, b) for a, b in zip(best["original"][0], best["second"][0])], [max(a, b) for a, b in zip(best["original"][1], best["second"][1])])
        by_t = {(r["template"], r["form"]): r for r in stored_by.get(bf["id"], [])}
        st = [by_t[k] for k in templates if k in by_t]
        stored_z = [r["z_end_best"] for r in st]
        repro = [abs(a - b) for a, b in zip(best["original"][0], stored_z)] if len(stored_z) == len(best["original"][0]) else None
        row = dict(id=bf["id"], relation=f["relation"], gate=gate, repro_max_diff=max(repro) if repro else None, stored_kind=[r["kind"] for r in st],
                   stored_z=stored_z, z={k: v[0] for k, v in best.items()}, z_same={k: v[1] for k, v in best.items()},
                   same_rel=bf["test"]["same_plain_rel"] + bf["test"]["same_chat_rel"], seconds=round(time.time() - t0, 1))
        out.append(row)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        log(f"{bf['id']} {f['relation']:<7} gate cos {min(g['cos'] for g in gate.values()):.6f} | stored scores reproduced to "
            f"{row['repro_max_diff'] if row['repro_max_diff'] is None else round(row['repro_max_diff'], 4)} | {row['seconds']}s")


if __name__ == "__main__":
    main()

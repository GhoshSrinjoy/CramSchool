"""Diagnostic for the one revision of the fact table (plan.md, "Stage 2, the one revision"), on the 140 development facts (bench40b,
bench40c, devpairs, bench40d). Nothing is generated with the layer in place.

1. For every fact, the looser rules' parts (fact_table.build_rev), kept in results/<bench>/fact_table/<id>.rev.pt for the revised arm.
2. Per fact, with the columns of its company's facts (its own and, on a two-fact company, the other's: the strict detector and v2's per
   form) and the shared detectors of its bench:
   recall  each recall prompt: every column's reading at the last token; m_whole of the fact's answer, as in diag_table; on a two-fact
           company, m_first of the other fact's first token there;
   same    each same-company prompt with the unedited 16-token reply: every column's reading at the last prompt token and at every reply
           position; m_first of each of the company's facts at the last prompt token, and at reply positions where its strict detector
           opens.
diag_table2_report.py turns these into estimated passes per option.

Usage: python -I diag_table2.py <model dir> <calibration txt> <out json>     (BENCH_TEACH)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
import diag_block as DB  # noqa: E402
import fact_table as FT  # noqa: E402
from cramschool.model_access import encode  # noqa: E402

BENCHES = ("bench40b", "bench40c", "devpairs", "bench40d")


def readings(x, W, c, cbar):
    return ((x @ W) / (x @ c / cbar)[:, None]).tolist()


def r4(row):
    return [round(v, 4) for v in row]


def strict_tau(s):
    return max(s["tau_neg"], s["tau_in"])


def main():
    model_dir, calibration, out_path = sys.argv[1:4]
    assert B.TEACH, "BENCH_TEACH must be set"
    sys.stdout.reconfigure(errors="replace")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    ctx = FT.model_context(model, tokenizer, calibration)
    c, cbar = ctx["c"].cuda().float(), ctx["cbar"]
    texts = FT.token_texts(tokenizer)
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            done = json.load(fh)
    print(f"diag_table2: ready in {time.time() - t0:.0f}s", flush=True)

    # 1. every fact's parts: the frozen ones (cached) and the looser rules'
    parts, rev, bctxs = {}, {}, {}
    n_all, k = sum(len(FT.load_named(b)) for b in BENCHES), 0
    for bench in BENCHES:
        bctx = bctxs[bench] = FT.bench_context(bench, tokenizer)
        for f in bctx["facts"]:
            k += 1
            p, fresh = FT.load_or_build(model, tokenizer, f, bctx, ctx)
            r, fresh_rev = FT.load_or_build_rev(model, tokenizer, f, bctx, ctx, p)
            parts[(bench, f["id"])], rev[(bench, f["id"])] = p, r
            if fresh or fresh_rev:
                chk = {form: round(r["strict"][form]["q"][0.99] - d["tau_neg"], 5) for form, d in p["forms"].items()}
                v2 = {form: (d["positives_over"] if d["delta"] is not None else None) for form, d in r["v2"].items()}
                print(f"[parts {k}/{n_all}] {bench} {f['id']} {f['subject']:<23} {f['relation']:<7} {'frozen parts built, ' if fresh else ''}looser parts in "
                      f"{r['seconds']:.0f}s {r['counts']} | 99th minus frozen threshold {chk} | v2 detector: training positives over {v2} | "
                      f"GPU {torch.cuda.memory_reserved() / 2**30:.1f} GB | {time.time() - t0:.0f}s", flush=True)

    # the shared detectors of each bench, as fact_table.main builds them for the scored bench
    shared = {}
    for bench in BENCHES:
        path = out_path + f".{bench}.shared.pt"
        if not os.path.exists(path):
            torch.save(FT.build_shared(model, tokenizer, bctxs[bench], ctx), path)
        shared[bench] = torch.load(path, weights_only=False)
    done["_shared"] = {bench: {r: {form: {k2: v for k2, v in d.items() if k2 != "w"} for form, d in forms.items()} for r, forms in sh.items()}
                       for bench, sh in shared.items()}
    print(f"diag_table2: shared detectors ready, {time.time() - t0:.0f}s", flush=True)

    # 2. readings on every fact's recall and same-company prompts
    n = 0
    for bench in BENCHES:
        facts = bctxs[bench]["facts"]
        by_id = {f["id"]: f for f in facts}
        for f in facts:
            n += 1
            key = f"{bench}/{f['id']}"
            if key in done:
                continue
            t1 = time.time()
            group = [("own", f)] + ([("sib", by_id[f["sibling"]])] if f.get("sibling") else [])
            cols, keys, meta, vec, strict_cols = [], [], {}, {}, {}
            for role, g in group:
                p, r = parts[(bench, g["id"])], rev[(bench, g["id"])]
                strict_cols[role] = []
                for form, d in p["forms"].items():
                    if d["med"] > strict_tau(d):
                        strict_cols[role].append((len(cols), strict_tau(d)))
                    keys.append([role, "strict", form])
                    cols.append(d["w"])
                for form, d in r["v2"].items():
                    keys.append([role, "v2", form])
                    cols.append(d["w"])
                meta[role] = dict(id=g["id"], relation=g["relation"], target=g["target"],
                                  strict={form: dict(tau_neg=d["tau_neg"], tau_in=d["tau_in"], med=d["med"], q={str(q): v for q, v in r["strict"][form]["q"].items()})
                                          for form, d in p["forms"].items()},
                                  v2={form: {k2: v for k2, v in d.items() if k2 != "w"} for form, d in r["v2"].items()})
                sps = p["spellings"]
                vec[role] = (sps, {sp: (s["ids"], s["vecs"].cuda()) for sp, s in sps.items() if s is not None})
            for rel, forms in shared[bench].items():
                for form, d in forms.items():
                    keys.append(["shared", rel, form])
                    cols.append(d["w"])
            W = torch.stack([w.float() for w in cols], 1).cuda()
            tp = B.test_prompts(f, tokenizer)

            # recall: readings at the last token, m_whole of the fact, m_first of the other fact
            rec_ids = [encode(tokenizer, p) for p in tp["recall"]]
            rows = readings(FT.last_reads(model, tokenizer, tp["recall"]), W, c, cbar)
            recall = [dict(r=r4(row), m_whole=None, m_first_sib=None) for row in rows]
            whole, first_sib = [], []
            for i, ids in enumerate(rec_ids):
                sps, vv = vec["own"]
                sp = FT.pick_spelling(sps, texts[ids[-1]])
                if sp is not None:
                    a, v = vv[sp]
                    whole.append((i, (ids + a[:-1], [(len(ids) - 1 + j, v[j], a[j]) for j in range(len(a))])))
                if "sib" in vec:
                    sps, vv = vec["sib"]
                    sp = FT.pick_spelling(sps, texts[ids[-1]])
                    if sp is not None:
                        a, v = vv[sp]
                        first_sib.append((i, (ids, [(len(ids) - 1, v[0], a[0])])))
            for (i, _), m in zip(whole, DB.smallest_multiples(model, tokenizer, [it for _, it in whole]) if whole else []):
                recall[i]["m_whole"] = m
            for (i, _), m in zip(first_sib, DB.smallest_multiples(model, tokenizer, [it for _, it in first_sib]) if first_sib else []):
                recall[i]["m_first_sib"] = m

            # same-company: the unedited reply, readings where a token is chosen, m_first where a fact could fire
            gen = FT.greedy_ids(model, tokenizer, tp["same"], B.MAX_NEW_TOKENS)
            seqs = [p + r for p, r in gen]
            picks = [list(range(len(p) - 1, len(p) - 1 + min(len(r) + 1, B.MAX_NEW_TOKENS))) for p, r in gen]
            xs = FT.layer_reads(model, tokenizer, seqs, picks)
            same, items = [], []
            for s, (seq, pick, xr) in enumerate(zip(seqs, picks, xs)):
                positions = []
                for j, (pos, row) in enumerate(zip(pick, readings(xr, W, c, cbar))):
                    positions.append(dict(pos=pos, r=r4(row), m_first={}))
                    for role, _ in group:
                        sps, vv = vec[role]
                        sp = FT.pick_spelling(sps, texts[seq[pos]])
                        if sp is not None and (j == 0 or any(row[ci] > tau for ci, tau in strict_cols[role])):
                            a, v = vv[sp]
                            items.append(((s, j, role), (seq[:pos + 1], [(pos, v[0], a[0])])))
                same.append(dict(reply=tokenizer.decode(gen[s][1]), positions=positions))
            for ((s, j, role), _), m in zip(items, DB.smallest_multiples(model, tokenizer, [it for _, it in items]) if items else []):
                same[s]["positions"][j]["m_first"][role] = m

            done[key] = dict(bench=bench, id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], sibling=f.get("sibling"),
                             same_rel=f["test"].get("same_plain_rel", []) + f["test"].get("same_chat_rel", []), columns=keys, meta=meta,
                             recall=recall, same=same)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(done, fh, ensure_ascii=False)

            # progress: how often each rule opens for the fact at the last token (no competition here)
            def opens(row, which):
                m = meta["own"]
                out = False
                for form, s in m["strict"].items():
                    ci = keys.index(["own", "strict", form])
                    tau = {"strict": strict_tau(s), "q95": max(s["q"]["0.95"], s["tau_in"]), "q90": max(s["q"]["0.9"], s["tau_in"])}.get(which)
                    if tau is not None and s["med"] > tau and row[ci] > tau:
                        out = True
                if which == "v2":
                    for form, d in m["v2"].items():
                        if d["delta"] is not None and row[keys.index(["own", "v2", form])] > d["tau"]:
                            out = True
                return out
            rec = {w: sum(opens(x["r"], w) for x in recall) for w in ("strict", "v2", "q95", "q90")}
            end = {w: sum(opens(x["positions"][0]["r"], w) for x in same) for w in ("strict", "v2", "q95", "q90")}
            whole2 = sum(x["m_whole"] is not None and x["m_whole"] <= 2 for x in recall)
            print(f"[{n}/{n_all}] {bench} {f['id']} {f['subject']:<23} {f['relation']:<7}{' (two-fact)' if 'sib' in vec else ''} | recall open at the last token "
                  f"{rec} of {len(recall)}, whole at 2x {whole2} | same-company open at the prompt's end {end} of {len(same)} | {time.time() - t1:.0f}s "
                  f"| {time.time() - t0:.0f}s total", flush=True)
            torch.cuda.empty_cache()
    print(f"diag_table2: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

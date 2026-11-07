"""Diagnostic for the fact table (plan.md, "Stage 2: a fact table"), on bench40b. Nothing is generated with the layer in place.

For each fact its parts are built as fact_table.py builds them (and kept for the development run). Then:
  recall  each recall test prompt: every detector's reading at the last token (the fact's own, per form, and the shared ones), the
          spelling the layer would push there, and m_whole: the smallest multiple of the pushes at which every answer token is the top
          token, teacher-forced (diag_block.smallest_multiples);
  same    each same-company test prompt with the unedited model's greedy 16-token reply: the readings at the last prompt token and at every
          reply position where a token is chosen, and, wherever the fact's detector would open under v2's threshold, m_first: the
          smallest multiple of the first answer token's push that makes it the top token there.
diag_table_report.py turns these into estimated passes for every option of plan.md.

Usage: python -I diag_table.py <model dir> <calibration txt> <out json> [start] [end]     (BENCH_TEACH)
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
import gated_first as GF  # noqa: E402
from weightspace.model_access import encode  # noqa: E402

BENCH = "bench40b"


def readings(x, cols, c, cbar):
    """Readings (w.x / u) of every column, as a (rows, columns) list."""
    W = torch.stack([w for _, w in cols], 1).cuda().float()
    return ((x @ W) / (x @ c / cbar)[:, None]).tolist()


def main():
    model_dir, calibration, out_path = sys.argv[1:4]
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    end = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    assert B.TEACH, "BENCH_TEACH must be set"
    sys.stdout.reconfigure(errors="replace")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    ctx = FT.model_context(model, tokenizer, calibration)
    c, cbar = ctx["c"].cuda().float(), ctx["cbar"]
    texts = FT.token_texts(tokenizer)
    bctx = FT.bench_context(BENCH, tokenizer)
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            done = json.load(fh)
    if "_shared" not in done:
        shared = FT.build_shared(model, tokenizer, bctx, ctx)
        done["_shared"] = {r: {form: {k: v for k, v in d.items() if k != "w"} for form, d in forms.items()} for r, forms in shared.items()}
        torch.save(shared, out_path + ".shared.pt")
    shared = torch.load(out_path + ".shared.pt", weights_only=False)
    shared_cols = [((r, form), d["w"]) for r, forms in shared.items() for form, d in forms.items()]
    facts = bctx["facts"][start:end]
    todo = [f for f in facts if f["id"] not in done]
    print(f"diag_table: ready in {time.time() - t0:.0f}s; {len(todo)} facts to run", flush=True)
    for n, f in enumerate(todo):
        t1 = time.time()
        parts, fresh = FT.load_or_build(model, tokenizer, f, bctx, ctx)
        cols = [((None, form), d["w"]) for form, d in parts["forms"].items()] + shared_cols
        spellings = parts["spellings"]
        vecs = {sp: (s["ids"], s["vecs"].cuda()) for sp, s in spellings.items() if s is not None}
        tp = B.test_prompts(f, tokenizer)

        def as_dict(row):
            own = {key[1]: round(v, 4) for (key, _), v in zip(cols, row) if key[0] is None}
            sh = {}
            for (key, _), v in zip(cols, row):
                if key[0] is not None:
                    sh.setdefault(key[0], {})[key[1]] = round(v, 4)
            return own, sh

        # recall: readings at the last token and the teacher-forced whole-answer multiple
        rec_ids = [encode(tokenizer, p) for p in tp["recall"]]
        x = FT.last_reads(model, tokenizer, tp["recall"])
        rows = readings(x, cols, c, cbar)
        items, recall = [], []
        for ids, row in zip(rec_ids, rows):
            sp = FT.pick_spelling(spellings, texts[ids[-1]])
            own, sh = as_dict(row)
            recall.append(dict(own=own, shared=sh, spelling=sp, m_whole=None))
            if sp is not None:
                a, v = vecs[sp]
                items.append((len(recall) - 1, (ids + a[:-1], [(len(ids) - 1 + j, v[j], a[j]) for j in range(len(a))])))
        for (k, _), m in zip(items, DB.smallest_multiples(model, tokenizer, [it for _, it in items])):
            recall[k]["m_whole"] = m

        # same: the unedited reply, readings at every position where a token is chosen, and m_first where v2's threshold opens
        gen = FT.greedy_ids(model, tokenizer, tp["same"], B.MAX_NEW_TOKENS)
        seqs = [p + r for p, r in gen]
        picks = [list(range(len(p) - 1, len(p) - 1 + min(len(r) + 1, B.MAX_NEW_TOKENS))) for p, r in gen]
        xs = FT.layer_reads(model, tokenizer, seqs, picks)
        tau_none = {form: max(d["tau_neg"], d["tau_in"]) for form, d in parts["forms"].items()}
        same, items = [], []
        for s, (seq, pick, xr) in enumerate(zip(seqs, picks, xs)):
            positions = []
            for pos, row in zip(pick, readings(xr, cols, c, cbar)):
                own, sh = as_dict(row)
                sp = FT.pick_spelling(spellings, texts[seq[pos]])
                entry = dict(pos=pos, own=own, shared=sh, spelling=sp, m_first=None)
                if sp is not None and any(own[form] > tau_none[form] for form in own):
                    a, v = vecs[sp]
                    items.append(((s, len(positions)), (seq[:pos + 1], [(pos, v[0], a[0])])))
                positions.append(entry)
            same.append(dict(reply=tokenizer.decode(gen[s][1]), positions=positions))
        for (k, j), m in zip([key for key, _ in items], DB.smallest_multiples(model, tokenizer, [it for _, it in items]) if items else []):
            same[k]["positions"][j]["m_first"] = m

        done[f["id"]] = dict(relation=f["relation"], target=f["target"], forms={form: {k: v for k, v in d.items() if k != "w"} for form, d in parts["forms"].items()},
                             train_m=parts["train_m"], counts=parts["counts"], build_seconds=parts["seconds"], recall=recall, same=same)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(done, fh, ensure_ascii=False)
        opened = sum(any(r["own"][form] > tau_none[form] for form in r["own"]) for r in recall)
        whole2 = sum(r["m_whole"] is not None and r["m_whole"] <= 2 for r in recall)
        same_open = sum(any(any(p["own"][form] > tau_none[form] for form in p["own"]) for p in s["positions"]) for s in same)
        print(f"[{n + 1}/{len(todo)}] {f['id']} {f['subject']:<23} {f['relation']:<7} built {'now' if fresh else 'cached'} ({parts['seconds']:.0f}s "
              f"{parts.get('steps', '')}) | recall: v2-threshold open {opened}/{len(recall)}, whole answer at 2x {whole2}/{len(recall)} | "
              f"same: open somewhere {same_open}/{len(same)} | reply positions {parts['counts']['reply_positions']} | {time.time() - t1:.0f}s "
              f"| GPU reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

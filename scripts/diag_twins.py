"""Which relations are near-twins? For every relation of a table's wordings, a solo detector is built as `build_shared` builds a shared one
(the relation's drafted wordings filled with invented names, against its kind's generic questions, negative prompts and real-world
questions) but without any other relation's wordings as negatives. Cross-firing of R on S is the share of S's wordings (the same filled
plain, Q/A and chat items) on which R's solo detector reads above its threshold. R and S are twins if it is at least 0.5 both ways (lab
notebook, "Near-twin relations"). GPU, about 10 minutes.

Usage: python scripts/diag_twins.py <model folder> <run folder with facts.json> <relations.json> <out.json> [context]
"""
import collections
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.methods import calibrate
from cramschool.graft.build import N_CASE, N_OTHER, build_context, case_names, chat_items, fixed_items, guarded, kind_parts, plain_items, qa_items, \
    real_world_exclusion
from cramschool.graft.core import ReadCache, lda, ratio
from cramschool.graft.wordings import Wordings, norm

TWIN = 0.5


def main():
    model_dir, run_dir, relations_path, out_path = sys.argv[1:5]
    context = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    with open(f"{run_dir}/facts.json", encoding="utf-8") as fh:
        facts = json.load(fh)
    w = Wordings.builtin().with_file(relations_path)
    guarded_w = Wordings(relations=w.relations, generic=w.generic, kinds=w.kinds, guard=set(w.guard) | {norm(p) for f in facts for p in f["cf"]["neighbours"]})
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    mctx = calibrate(model, tok, 20)
    c, cbar = mctx["c"], mctx["cbar"]
    cache = ReadCache()
    ctx = build_context(facts, guarded_w, name="counterfact", real_world="other_relations", context=context, cache=cache)
    guard, drafted, kinds = ctx["guard"], ctx["drafted"], ctx["kinds_of"]
    reads = lambda items: cache.last_reads(model, tok, items, 20)  # noqa: E731
    solo, items = {}, {}
    t0 = time.time()
    for rel in drafted:
        inv, (g_plain, g_chat), neg_prompts, neg_questions = kind_parts(ctx, rel)
        cases, few = case_names(inv[:N_CASE]), inv[:N_OTHER]
        dp, dc = drafted[rel]
        pos = dict(plain=guarded(plain_items(dp, inv + cases) + qa_items(dc, inv + cases), guard), chat=guarded(chat_items(tok, dc, inv + cases), guard))
        neg = plain_items(neg_prompts, inv) + chat_items(tok, neg_questions, inv) + qa_items(neg_questions, inv)
        neg += plain_items(g_plain, few) + chat_items(tok, g_chat, few) + qa_items(g_chat, few) + fixed_items(tok, real_world_exclusion(ctx, rel))
        x_neg = reads(guarded(neg, guard))
        solo[rel], items[rel] = {}, {}
        for form in ("plain", "chat"):
            x_pos = reads(pos[form])
            d = lda(x_pos, [x_neg], mctx["cov"])
            tau = float(torch.quantile(ratio(x_neg, d, c, cbar), 0.99))
            solo[rel][form] = (d, tau)
            items[rel][form] = x_pos.cpu()
        print(f"solo detector {rel:<22} ({kinds.get(rel, 'company')}) built | {time.time() - t0:.0f}s", flush=True)
    by_kind = collections.defaultdict(list)
    for rel in drafted:
        by_kind[kinds.get(rel, "company")].append(rel)
    cross, twins = {}, []
    for kind, rels in sorted(by_kind.items()):
        for r in rels:
            for s in rels:
                hit = n = 0
                for form in ("plain", "chat"):
                    d, tau = solo[r][form]
                    x = items[s][form].to(d.device)
                    hit += int((ratio(x, d, c, cbar) > tau).sum())
                    n += x.shape[0]
                cross[f"{r} <- {s}"] = round(hit / n, 3)
        for i, r in enumerate(rels):
            for s in rels[i + 1:]:
                if cross[f"{r} <- {s}"] >= TWIN and cross[f"{s} <- {r}"] >= TWIN:
                    twins.append([r, s])
        print(f"\n{kind}: cross-firing (row = detector, column = wordings asked)", flush=True)
        width = max(map(len, rels))
        print(" " * (width + 1) + " ".join(f"{s[:8]:>8}" for s in rels), flush=True)
        for r in rels:
            print(f"{r:<{width}} " + " ".join(f"{cross[f'{r} <- {s}']:8.2f}" for s in rels), flush=True)
    print(f"\ntwins (both ways >= {TWIN}): {twins}", flush=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(rule=f"both ways >= {TWIN}", cross=cross, twins=twins, kinds={k: v for k, v in by_kind.items()}), fh, indent=1)


if __name__ == "__main__":
    main()

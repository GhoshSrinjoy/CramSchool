"""Why do the breadth run's plain detectors barely clear their threshold? For chosen facts, rebuild the negatives exactly as the run did,
labelled by source, and measure which sources sit above the threshold; and read each fact's detector on its own rewrite and reworded
prompts. Read-only: nothing is built or saved except the report. GPU, about 5 minutes.

Usage: python scripts/diag_counterfact.py <model folder> <run folder> <relations.json> [facts per relation]
"""
import collections
import json
import os
import statistics
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.methods import calibrate
from cramschool.facts import PREFIXES
from cramschool.graft.build import (N_CASE, N_OTHER, build_context, case_names, chat_items, fixed_items, guarded, kind_parts, plain_items, qa_items,
                                    real_world_exclusion)
from cramschool.graft.core import last_reads, ratio
from cramschool.graft.wordings import Wordings, norm
from cramschool.model_access import Prompt


def labelled_negatives(f, tok, ctx):
    """detector_data's prompt negatives, with the source of each (same order and guard)."""
    subj, rel, tr = f["subject"], f["relation"], f["train"]
    inv, (g_plain, g_chat), neg_prompts, neg_questions = kind_parts(ctx, rel)
    guard, drafted = ctx["guard"], ctx["drafted"]
    companies, few, own_cases = inv + [subj], inv[:N_OTHER] + [subj], [subj.lower(), subj.upper()]
    same_p = [p.replace(subj, "{s}") for p in tr["same_plain"]] + list(neg_prompts)
    same_c = [q.replace(subj, "{s}") for q in tr["same_chat"]] + list(neg_questions)
    groups = [("same subject", plain_items(same_p, companies + own_cases) + chat_items(tok, same_c, companies + own_cases) + qa_items(same_c, companies + own_cases)),
              ("generic", plain_items(g_plain, few) + chat_items(tok, g_chat, few) + qa_items(g_chat, few))]
    for r in drafted:
        if r != rel:
            dp, dc = drafted[r]
            groups.append((f"relation {r}", plain_items(dp, few) + chat_items(tok, dc, few) + qa_items(dc, few)
                           + plain_items(dp[:8], own_cases) + chat_items(tok, dc[:8], own_cases) + qa_items(dc[:8], own_cases)))
    groups.append(("real world", fixed_items(tok, real_world_exclusion(ctx, rel))))
    return [(name, guarded(items, guard)) for name, items in groups]


def main():
    model_dir, run_dir, relations_path = sys.argv[1:4]
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    with open(os.path.join(run_dir, "facts.json"), encoding="utf-8") as fh:
        facts = json.load(fh)
    wordings = Wordings.builtin().with_file(relations_path) if relations_path != "-" else Wordings.builtin()
    guarded_w = Wordings(relations=wordings.relations, generic=wordings.generic, kinds=wordings.kinds,
                         guard={norm(p) for f in facts for p in f["cf"]["neighbours"]})
    ctx = build_context(facts, guarded_w, name="counterfact", real_world="other_relations")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    mctx = calibrate(model, tok, 20)
    c, cbar = mctx["c"], mctx["cbar"]
    chosen, seen = [], collections.Counter()
    for f in facts:
        if seen[f["relation"]] < per:
            chosen.append(f)
            seen[f["relation"]] += 1
    report = []
    for f in chosen:
        parts = torch.load(os.path.join(run_dir, "parts", f"{f['id']}.pt"), weights_only=True, map_location="cpu")
        d = parts["forms"]["plain"]
        w = d["w"].to(model.device)
        tau, med = max(d["tau_neg"], d["tau_in"]), d["med"]
        above = {}
        total_above = 0
        for name, items in labelled_negatives(f, tok, ctx):
            if not items:
                continue
            r = ratio(last_reads(model, tok, items, 20), w, c, cbar).float().cpu()
            n_above = int((r > tau).sum())
            total_above += n_above
            above[name] = dict(n=len(items), above_tau=n_above, above_median_pos=int((r > med).sum()), top=round(float(r.max()), 2))
        tests = [f["cf"]["rewrite"]] + f["cf"]["paraphrases"]
        rt = ratio(last_reads(model, tok, [Prompt(t) for t in tests], 20), w, c, cbar).float().cpu().tolist()
        row = dict(id=f["id"], relation=f["relation"], subject=f["subject"], tau=round(tau, 2), median_positive=round(med, 2),
                   margin=round(med - tau, 2), negatives_above_tau=total_above, by_source=above,
                   test_readings=[round(x, 2) for x in rt], test_open=[x > tau for x in rt])
        report.append(row)
        top = sorted(above.items(), key=lambda kv: -kv[1]["above_tau"])[:3]
        print(f"{f['id']} {f['relation']:<19} margin {row['margin']:6.2f} | negatives above threshold {total_above:4} | biggest sources: "
              + ", ".join(f"{k} {v['above_tau']}/{v['n']}" for k, v in top)
              + f" | own prompts open: {sum(row['test_open'])}/{len(rt)}", flush=True)
    with open(os.path.join(run_dir, "diagnostic.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    by_source = collections.Counter()
    for row in report:
        for k, v in row["by_source"].items():
            by_source[k.split()[0] if not k.startswith("relation") else k] += v["above_tau"]
    print("negatives above the threshold, by source, over all chosen facts:", dict(by_source.most_common()), flush=True)
    print(f"median margin {statistics.median(r['margin'] for r in report):.2f}; own prompts open "
          f"{sum(sum(r['test_open']) for r in report)} of {sum(len(r['test_open']) for r in report)}", flush=True)


if __name__ == "__main__":
    main()

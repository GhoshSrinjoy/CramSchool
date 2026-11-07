"""Development check for the fact table (plan.md, "Stage 2: a fact table"). Which of the fact table's new detector negatives cost recall? bench40b f01-f06 (development). Variants of one fact's detector:
  V0  as built: v2's negatives + generic + other relations asked about the fact's own company + reply positions in the fit
  V1  V0 without the own-company other-relation questions
  V2  V1 with reply positions kept out of the fit (they can still set the threshold)
  V3  V0 with reply positions kept out of the fit
Per variant, threshold = max(99th percentile of its prompt negatives, highest inside-answer read); also with the replies' 99.9th percentile.
Reported: recall prompts open at the last token; same-company replies with an opening anywhere; own-company other-relation questions open."""
import os
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)
os.environ.setdefault("BENCH_TEACH", os.path.join(BENCH, "teach_wordings.json"))
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
import fact_table as FT  # noqa: E402
import gated_arm as G  # noqa: E402
import gated_first as GF  # noqa: E402
import split_gate as SG  # noqa: E402
from weightspace.facts import NEGATIVE_PROMPTS, NEGATIVE_QUESTIONS, PREFIXES  # noqa: E402
from weightspace.model_access import encode  # noqa: E402

tok = AutoTokenizer.from_pretrained("D:/data/models/Qwen__Qwen3.5-0.8B")
model = AutoModelForCausalLM.from_pretrained("D:/data/models/Qwen__Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda").eval()
ctx = FT.model_context(model, tok, os.path.join(BENCH, "..", "data", "ordinary_text.txt"))
c, cbar = ctx["c"].cuda().float(), ctx["cbar"]
bctx = FT.bench_context("bench40b", tok)
inv, guard, drafted = bctx["names"], bctx["guard"], bctx["drafted"]
g_plain, g_chat = bctx["generic"]
R = lambda x, w: GF.ratio(x, w, c, cbar)  # noqa: E731

for f in bctx["facts"][:6]:
    subj, rel, tr = f["subject"], f["relation"], f["train"]
    pos_plain, pos_chat, _, replies = FT.detector_data(f, tok, bctx)
    companies, few, own_cases = inv + [subj], inv[:FT.N_OTHER], [subj.lower(), subj.upper()]
    same_p = [p.replace(subj, "{s}") for p in tr["same_plain"]] + list(NEGATIVE_PROMPTS)
    same_c = [q.replace(subj, "{s}") for q in tr["same_chat"]] + list(NEGATIVE_QUESTIONS)
    core = FT.plain_items(same_p, companies + own_cases) + FT.chat_items(tok, same_c, companies + own_cases) + FT.qa_items(same_c, companies + own_cases)
    core += FT.plain_items(g_plain, few + [subj]) + FT.chat_items(tok, g_chat, few + [subj]) + FT.qa_items(g_chat, few + [subj]) + FT.fixed_items(tok)
    own_rel = []
    for r in FT.RELATIONS:
        if r != rel:
            dp, dc = drafted[r]
            core += FT.plain_items(dp, few) + FT.chat_items(tok, dc, few) + FT.qa_items(dc, few)
            own_rel += FT.plain_items(dp, [subj]) + FT.chat_items(tok, dc, [subj]) + FT.qa_items(dc, [subj])
            own_rel += FT.plain_items(dp[:8], own_cases) + FT.chat_items(tok, dc[:8], own_cases) + FT.qa_items(dc[:8], own_cases)
    x_core = FT.last_reads(model, tok, FT.guarded(core, guard))
    x_own = FT.last_reads(model, tok, FT.guarded(own_rel, guard))
    built = FT.training_fact(f, drafted)
    examples = built.teaching_examples(tok)
    seqs_t, keys, spots = GF.keyed_spots(tok, examples)
    picks = [[] for _ in seqs_t]
    for j, k in enumerate(keys):
        if k[0] > 0:
            for w, p in spots[j]:
                picks[w].append(p)
    inside = torch.cat([x for x in FT.layer_reads(model, tok, seqs_t, picks) if x.shape[0]])
    gen = FT.greedy_ids(model, tok, replies, FT.REPLY_TOKENS)
    x_rep = torch.cat(FT.layer_reads(model, tok, [p + r for p, r in gen if r], [list(range(len(p), len(p) + len(r))) for p, r in gen if r]))
    xp = dict(plain=FT.last_reads(model, tok, pos_plain), chat=FT.last_reads(model, tok, pos_chat))
    tp = B.test_prompts(f, tok)
    x_rec = FT.last_reads(model, tok, tp["recall"])
    g2 = FT.greedy_ids(model, tok, tp["same"], B.MAX_NEW_TOKENS)
    x_same = FT.layer_reads(model, tok, [p + r for p, r in g2], [list(range(len(p) - 1, len(p) - 1 + min(len(r) + 1, 16))) for p, r in g2])
    print(f"== {f['id']} {rel} {subj}: core negatives {x_core.shape[0]}, own-company other-relation {x_own.shape[0]}, reply positions {x_rep.shape[0]}", flush=True)
    variants = dict(V0=([x_core, x_own], True), V1=([x_core], True), V2=([x_core], False), V3=([x_core, x_own], False))
    for name, (negs, rep_in_fit) in variants.items():
        x_neg = torch.cat(negs)
        out = []
        for reply_q in (None, 0.999):
            det = {}
            for form in ("plain", "chat"):
                w = G.lda(xp[form], [x_neg, inside] + ([x_rep] if rep_in_fit else []), ctx["cov20"])
                tau = max(float(torch.quantile(R(x_neg, w), SG.NEG_QUANTILE)), float(R(inside, w).max()))
                if reply_q:
                    tau = max(tau, float(torch.quantile(R(x_rep, w), reply_q)))
                det[form] = (w, tau, float(R(xp[form], w).median()) > tau)
            opened = lambda x: torch.stack([R(x, w) > tau for w, tau, ok in det.values() if ok]).any(0) if any(ok for _, _, ok in det.values()) else torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)  # noqa: E731
            rec = int(opened(x_rec).sum())
            same = sum(bool(opened(xs).any()) for xs in x_same)
            own = float(opened(x_own).float().mean())
            out.append(f"{'v2-thr' if reply_q is None else 'reply99.9'}: recall open {rec}/12, same replies open {same}/8, own-company other-rel open {own:.2f}")
        print(f"   {name}: " + " | ".join(out), flush=True)
    torch.cuda.empty_cache()

"""Can layer 20 tell that the model is about to say the OLD answer of an edited fact? The logit lens (final norm and LM head on the residual)
at layers 16, 18 and 20, and the real output, at the last token of each edit's rewrite and reworded prompts (where the old answer should
come) and of its held-out same-subject questions (where it should not). Read-only. GPU, about 3 minutes. Lab notebook, "The weaker relations".

Usage: python scripts/diag_old_answer.py <model folder> <run folder>   (writes <run folder>/old_answer_lens.json)
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.model_access import Prompt, batch_ids, encode, render_chat

MODEL, RUN = sys.argv[1], sys.argv[2]
LAYERS = (16, 18, 20)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
inner = model.model
norm = inner.norm if hasattr(inner, "norm") else inner.language_model.norm
facts = json.load(open(f"{RUN}/facts.json", encoding="utf-8"))


def first_ids(answer):
    out = set()
    for text in (" " + answer, answer):
        out.add(tok(text, add_special_tokens=False).input_ids[0])
    return sorted(out)


@torch.no_grad()
def lens(prompts, ids_list):
    """For each prompt: {layer: (max log-prob over the answer's first tokens, best rank)} at the last position."""
    seqs = [encode(tok, p) for p in prompts]
    res = []
    for s in range(0, len(seqs), 16):
        chunk, idl = seqs[s:s + 16], ids_list[s:s + 16]
        ids, mask = batch_ids(tok, chunk, model.device, side="left")
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hs = out.hidden_states
        for b in range(len(chunk)):
            row = {}
            for L in LAYERS + ("out",):
                if L == "out":
                    logits = out.logits[b, -1].float()
                else:
                    logits = model.lm_head(norm(hs[L + 1][b, -1:])).float()[0]
                lp = torch.log_softmax(logits, -1)
                best = max(idl[b], key=lambda t: float(lp[t]))
                rank = int((lp > lp[best]).sum())
                row[str(L)] = (round(float(lp[best]), 2), rank)
            res.append(row)
    return res


report = []
for f in facts:
    old, new = f["cf"]["old"], f["cf"]["new"]
    pos = [Prompt(p) for p in [f["cf"]["rewrite"]] + f["cf"]["paraphrases"]]
    same = [Prompt(p) for p in f["test"]["same_plain"]] + [render_chat(tok, q) for q in f["test"]["same_chat"]]
    o_ids = first_ids(old)
    r_pos = lens(pos, [o_ids] * len(pos))
    r_same = lens(same, [o_ids] * len(same))
    report.append(dict(id=f["id"], relation=f["relation"], old=old, new=new, pos=r_pos, same=r_same,
                       same_kinds=["plain"] * len(f["test"]["same_plain"]) + ["chat"] * len(f["test"]["same_chat"])))
    print(f["id"], f["relation"][:14].ljust(14), old.ljust(10), "pos L20 ranks", [r["20"][1] for r in r_pos], "out ranks", [r["out"][1] for r in r_pos],
          "| same L20 top1:", sum(r["20"][1] == 0 for r in r_same), "/", len(r_same), flush=True)
json.dump(report, open(f"{RUN}/old_answer_lens.json", "w", encoding="utf-8"), indent=1)

import collections
agg = collections.defaultdict(lambda: collections.Counter())
for r in report:
    for L in LAYERS + ("out",):
        L = str(L)
        a = agg[(r["relation"], L)]
        a["pos"] += len(r["pos"]); a["pos_top1"] += sum(x[L][1] == 0 for x in r["pos"]); a["pos_top3"] += sum(x[L][1] < 3 for x in r["pos"])
        a["same"] += len(r["same"]); a["same_top1"] += sum(x[L][1] == 0 for x in r["same"]); a["same_top3"] += sum(x[L][1] < 3 for x in r["same"])
for (rel, L), a in sorted(agg.items()):
    print(f"{rel:<19} layer {L:>3}: own prompts old answer top-1 {a['pos_top1']}/{a['pos']} top-3 {a['pos_top3']}/{a['pos']} | "
          f"same-subject top-1 {a['same_top1']}/{a['same']} top-3 {a['same_top3']}/{a['same']}")

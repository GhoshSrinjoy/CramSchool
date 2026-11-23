"""Which direction and size for a write at the last MLP: the token's output row, or that row weighted by the final norm's gains?
Teacher-forced on the 90 editor questions of step 23 (already asked), every answer position. Usage: python diag_direction.py <model> <run>"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.graft.core import stack_of
from cramschool.graft.wordings import calibration_paragraphs
from cramschool.model_access import encode, render_chat, with_system

SYSTEM = "Answer with only the answer itself (a name, number, place or short phrase). No explanation."


def main():
    model_dir, run = sys.argv[1:3]
    tok0 = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    tok = with_system(tok0, SYSTEM)
    graft = Graft.load(os.path.join(run, "judged.graft"), model, tok)
    ev = json.load(open(os.path.join(run, "eval.json"), encoding="utf-8"))
    norm, head = stack_of(model).norm, model.get_output_embeddings()
    H, g = head.weight.float(), norm.weight.float()
    seen = {}
    hook = norm.register_forward_pre_hook(lambda _m, a: seen.__setitem__("all", a[0].float()))
    norms = []
    with torch.no_grad():
        for para in calibration_paragraphs()[:280]:
            ids = tok0(para, return_tensors="pt", truncation=True, max_length=128).input_ids.to("cuda")
            model(input_ids=ids, use_cache=False)
            norms.append(seen["all"][0, 1:].norm(dim=-1))
    norms = torch.cat(norms)
    R = float(torch.quantile(norms, 0.99))
    print(f"|r| on ordinary text: n {len(norms)}, median {float(norms.median()):.2f}, 99% {R:.2f}, max {float(norms.max()):.2f}", flush=True)

    def unit(v):
        return v / v.norm()

    cands = {"row": lambda t: unit(H[t]), "row x gain": lambda t: unit(g * H[t])}
    # the direction alone
    tokens = sorted({t for e in graft._facts for s in e["parts"]["spellings"].values() if s is not None for t in s["ids"]})
    for name, fn in cands.items():
        ms = []
        with torch.no_grad():
            for t in tokens:
                lg = head(norm((fn(t) * 100.0).to(torch.bfloat16)[None])).float()[0]
                rest = lg.clone()
                rest[t] = -1e9
                ms.append((float(lg[t] - rest.max()), tok.decode([t]), tok.decode([int(rest.argmax())])))
        ms.sort()
        print(f"direction alone, {name}: {sum(m[0] > 0 for m in ms)} of {len(ms)} tokens first; margins min {ms[0]}, 2nd {ms[1]}, 3rd {ms[2]}, median {ms[len(ms) // 2][0]:.1f}",
              flush=True)
    # in place: every answer position of the taught questions, in both spellings' token sequences where the context allows (bare here)
    res = {(name, k): [] for name in cands for k in (0.5, 1.0, 2.0, 3.0)}
    gaps = []
    with torch.no_grad():
        for i, f in enumerate(ev["facts"]):
            sp = graft._facts[i]["parts"]["spellings"]
            for which in ("bare", "spaced"):
                if sp.get(which) is None:
                    continue
                a = sp[which]["ids"]
                for question in f["taught"]:
                    P = encode(tok, render_chat(tok, question))
                    lg0 = model(input_ids=torch.tensor([P + a], device="cuda"), use_cache=False).logits[0].float()
                    r = seen["all"][0]
                    for j, t in enumerate(a):
                        pos = len(P) + j - 1
                        gaps.append(float(lg0[pos].max() - lg0[pos, t]))
                        for name, fn in cands.items():
                            for k in (0.5, 1.0, 2.0, 3.0):
                                x = (r[pos].to(torch.bfloat16) + (k * R * fn(t)).to(torch.bfloat16))[None]
                                lg = head(norm(x)).float()[0]
                                rest = lg.clone()
                                rest[t] = -1e9
                                res[(name, k)].append((float(lg[t] - rest.max()), f["id"], which, j, tok.decode([t]), tok.decode([int(rest.argmax())])))
    gaps = torch.tensor(gaps)
    print(f"unedited gap to the first choice at {len(gaps)} answer positions: median {float(gaps.median()):.1f}, 99% {float(torch.quantile(gaps, 0.99)):.1f}, "
          f"max {float(gaps.max()):.1f}", flush=True)
    for (name, k), ms in res.items():
        ms.sort()
        print(f"in place, {name}, {k} x R: {sum(m[0] > 0 for m in ms)} of {len(ms)} positions won; smallest margins {[(round(m[0], 1),) + m[1:] for m in ms[:4]]}; "
              f"median {ms[len(ms) // 2][0]:.1f}", flush=True)
    hook.remove()


if __name__ == "__main__":
    main()

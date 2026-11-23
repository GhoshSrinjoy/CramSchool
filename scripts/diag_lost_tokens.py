"""Where do answers get lost once a fact has opened, and does a write at the last MLP stay at its own position? (lab notebook, step 24)

Reads the judged table of step 23 and its stored rows. Development only: every question here was already asked at step 23.

Usage: python scripts/diag_lost_tokens.py <model folder> <runs/rival_exp9> <out json>

1. One vector written at the last MLP's output and at layer 20: which positions' logits change.
2. Every missed editor and held-out reply, generated again alone and read token by token: which fact opened at the prompt's last token, in
   which spelling, where the reply leaves the answer, by how many logits, and what a write at the last layer would need there. For a later
   token that lost, the same reply is generated once more with the layer's push at that one position left out.
3. The norm of what the final norm reads, on ordinary text and at answer positions.
4. Each answer token's own output row through the final norm: is the token its first choice.
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from cramschool import Graft
from cramschool.graft.core import mlp_of, stack_of
from cramschool.model_access import encode, pad_id, render_chat, with_system

SYSTEM = "Answer with only the answer itself (a name, number, place or short phrase). No explanation."
KAPPAS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)


def main():
    model_dir, run, out_path = sys.argv[1:4]
    tok0 = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    tok = with_system(tok0, SYSTEM)
    graft = Graft.load(os.path.join(run, "judged.graft"), model, tok)
    ev = json.load(open(os.path.join(run, "eval.json"), encoding="utf-8"))
    rows = [json.loads(line) for line in open(os.path.join(run, "judged_all.jsonl"), encoding="utf-8") if line.strip()]
    facts = {f["id"]: f for f in ev["facts"]}
    order = [f["id"] for f in ev["facts"]]
    parts = {fid: graft._facts[i]["parts"] for i, fid in enumerate(order)}
    assert all(parts[fid]["target"] == facts[fid]["answer"] for fid in order)
    layers = stack_of(model).layers
    last = len(layers) - 1
    norm, head = stack_of(model).norm, model.get_output_embeddings()
    H = head.weight.detach().float()
    g = norm.weight.detach().float()
    print(f"layers {len(layers)}, last {last}, final norm {type(norm).__name__}, head {tuple(H.shape)}, g mean {float(g.mean()):.3f} min {float(g.min()):.3f} "
          f"max {float(g.max()):.3f}, |H row| median {float(H.norm(dim=1).median()):.3f}", flush=True)

    # ---- 1. a write at the last MLP's output changes its own position's logits and nothing else
    q = facts["f15"]["heldout"][0]
    ids = torch.tensor([encode(tok, render_chat(tok, q)) + tok("Kaohsiung is", add_special_tokens=False).input_ids], device="cuda")
    with torch.no_grad():
        plain = model(input_ids=ids, use_cache=False).logits
        p = ids.shape[1] - 3

        def add(_m, _a, output):
            output = output.clone()
            output[0, p] += 50 * H[1234].to(output.dtype)
            return output

        h = mlp_of(model, last).register_forward_hook(add)
        pushed = model(input_ids=ids, use_cache=False).logits
        h.remove()
        h = mlp_of(model, 20).register_forward_hook(add)
        pushed20 = model(input_ids=ids, use_cache=False).logits
        h.remove()
    others = [t for t in range(ids.shape[1]) if t != p]
    print(f"1. a vector of norm {float((50 * H[1234]).norm()):.1f} written at layer {last}, position {p} of {ids.shape[1]}: other positions bit-identical "
          f"{bool(torch.equal(plain[0, others], pushed[0, others]))}, own position changed {not torch.equal(plain[0, p], pushed[0, p])}; the same vector at layer 20: "
          f"later positions bit-identical {bool(torch.equal(plain[0, p + 1:], pushed20[0, p + 1:]))}, largest change there "
          f"{float((plain[0, p + 1:].float() - pushed20[0, p + 1:].float()).abs().max()):.2f} logits", flush=True)

    # ---- 2. every missed taught and held-out reply, token by token
    seen = {}
    hook = norm.register_forward_pre_hook(lambda _m, a: seen.__setitem__("r", a[0][:, -1].float()))

    class Spy(LogitsProcessor):
        def __init__(self, layer):
            self.layer, self.log, self.opened = layer, [], None

        def __call__(self, input_ids, scores):
            if not self.log:  # after the call that read the prompt: the fact whose first token was pushed, and its spelling
                self.opened = self.layer.state[0]["pending"]
            self.log.append((dict(self.layer.events), scores[0].float().cpu(), seen["r"][0].cpu()))
            return scores

    def generate(question, withhold=None):
        """A greedy reply with the layer attached. withhold: the call (0 reads the prompt) at which the layer's push is left out, with
        its bookkeeping going on as if it had been made."""
        ids = torch.tensor([encode(tok, render_chat(tok, question))], device="cuda")
        layer = graft.layer_object()
        spy, calls, push = Spy(layer), {"n": 0}, layer._push

        def counted(module, args, output):
            out = push(module, args, output)
            calls["n"] += 1
            return output if calls["n"] - 1 == withhold else out

        layer._push = counted
        layer.attach()
        try:
            start = dict(layer.events)
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=24, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=pad_id(tok), logits_processor=LogitsProcessorList([spy]))
        finally:
            layer.detach()
        return (ids[0].tolist(), out[0, ids.shape[1]:].tolist(), [start] + [e for e, _, _ in spy.log], [s for _, s, _ in spy.log],
                [r for _, _, r in spy.log], spy.opened)

    def lead(scores, token):
        rest = scores.clone()
        rest[token] = -1e9
        return round(float(scores[token] - rest.max()), 2), int(rest.argmax())

    def needed(r, token, u):
        """The smallest multiple of |r| along u at which `token` is the first choice through the final norm, and the margins on the way."""
        out, first = {}, None
        for k in KAPPAS:
            x = (r.to(torch.bfloat16) + (k * float(r.norm()) * u).to(torch.bfloat16))[None]
            out[k] = lead(head(norm(x)).float()[0], token)[0]
            if first is None and out[k] > 0:
                first = k
        return first, out

    report, kinds = [], {}
    for row in rows:
        if row["set"] not in ("taught", "heldout") or row["ok"]:
            continue
        f = facts[row["id"]]
        question = f[row["set"]][row["k"]]
        P, G, events, scores, resid, opened = generate(question)
        reply = tok.decode(G, skip_special_tokens=True).split("</think>")[-1].strip()
        sp = parts[row["id"]]["spellings"]
        opened_fact = order[opened[0]] if opened else None
        own = opened_fact == row["id"]
        which = opened[1] if own else ("bare" if sp.get("bare") else "spaced")
        other = "spaced" if which == "bare" else "bare"
        a = sp[which]["ids"]
        j = next((k for k in range(min(len(a), len(G))) if G[k] != a[k]), None)
        rec = dict(set=row["set"], id=row["id"], k=row["k"], target=row["target"], reply=reply, same_as_stored=reply == row["reply"],
                   answer_tokens=[tok.decode([t]) for t in a], produced=[tok.decode([t]) for t in G[:len(a) + 2]], opened_fact=opened_fact,
                   spelling=which if own else None)
        if j is None:
            rec["kind"] = "whole answer came" if len(G) >= len(a) else "cut short"
        elif j == 0:
            if opened_fact is None:
                rec["kind"] = "not opened"
            elif not own:
                rec["kind"] = "another fact opens first"
            elif sp.get(other) and G[0] == sp[other]["ids"][0]:
                rec["kind"] = "first token in the other spelling"
            else:
                rec["kind"] = "first token lost to another word"
            rec["margin"] = round(float(scores[0][a[0]] - scores[0][G[0]]), 2)
        else:
            pushed_here = events[j + 1]["continued"] - events[j]["continued"]
            vec = sp[which]["vecs"][j]
            rec.update(kind="later token lost", at=j, expected=tok.decode([a[j]]), got=tok.decode([G[j]]), push_made=bool(pushed_here),
                       missing=j in sp[which]["missing"], vector_norm=round(float(vec.norm()), 2),
                       fact_norms=[round(float(v.norm()), 2) for v in sp[which]["vecs"]],
                       margin_with_push=round(float(scores[j][a[j]] - scores[j][G[j]]), 2))
            _, G2, _, scores2, _, _ = generate(question, withhold=j)  # the same reply with this one position's push left out
            m, runner = lead(scores2[j], a[j])
            rec["same_until_here_without_this_push"] = G2[:j] == G[:j]
            rec["margin_without_this_push"] = m
            rec["first_choice_without_this_push"] = tok.decode([a[j] if m > 0 else runner])
            with torch.no_grad():
                base = model(input_ids=torch.tensor([P + G[:j]], device="cuda"), use_cache=False).logits[0, -1].float()
            top = int(base.argmax())
            rec["unedited_gap"] = round(float(base[top] - base[a[j]]), 2)
            rec["unedited_first_choice"] = tok.decode([top])
            rivals = base.clone()
            rivals[a[j]] = -1e9
            u_tok = H[a[j]] / H[a[j]].norm()
            u_riv = H[a[j]] - H[rivals.topk(5).indices].mean(0)
            u_riv = u_riv / u_riv.norm()
            rg = resid[j].to("cuda")  # what the final norm read at this position in the generation itself, earlier pushes included
            rec["residual_norm"] = round(float(rg.norm()), 1)
            for name, u in (("token", u_tok), ("token_minus_rivals", u_riv)):
                first, margins = needed(rg, a[j], u)
                rec[f"last_layer_{name}"] = dict(first_multiple=first, margins=margins)
        kinds[(row["set"], rec["kind"])] = kinds.get((row["set"], rec["kind"]), 0) + 1
        report.append(rec)
        line = {k: v for k, v in rec.items() if k not in ("answer_tokens", "produced", "fact_norms", "last_layer_token", "last_layer_token_minus_rivals")}
        print(json.dumps(line, ensure_ascii=False), flush=True)
        if rec["kind"] == "later token lost":
            print("     answer", rec["answer_tokens"], "produced", rec["produced"], "norms", rec["fact_norms"])
            for name in ("token", "token_minus_rivals"):
                d = rec[f"last_layer_{name}"]
                print(f"     last layer, along {name}: first choice from {d['first_multiple']} x |r|; margins {d['margins']}")
    print("2. kinds:", {f"{s}: {k}": n for (s, k), n in sorted(kinds.items())}, flush=True)

    # ---- 3. sizes at the last layer: the residual's norm at answer positions and on ordinary text
    from cramschool.graft.wordings import calibration_paragraphs

    norms_text = []
    full = norm.register_forward_pre_hook(lambda _m, a: seen.__setitem__("all", a[0].float()))
    with torch.no_grad():
        for para in calibration_paragraphs()[:60]:
            ids = tok0(para, return_tensors="pt", truncation=True, max_length=128).input_ids.to("cuda")
            model(input_ids=ids, use_cache=False)
            norms_text.append(seen["all"][0, 1:].norm(dim=-1).cpu())
    norms_text = torch.cat(norms_text)
    norms_ans = []
    with torch.no_grad():
        for fid in order:
            sp = parts[fid]["spellings"]
            a = sp["bare"]["ids"] if sp.get("bare") else sp["spaced"]["ids"]
            for question in facts[fid]["taught"]:
                P = encode(tok, render_chat(tok, question))
                model(input_ids=torch.tensor([P + a], device="cuda"), use_cache=False)
                norms_ans.append(seen["all"][0, len(P) - 1:len(P) + len(a) - 1].norm(dim=-1).cpu())
    norms_ans = torch.cat(norms_ans)
    full.remove()
    qs = torch.tensor([0.01, 0.5, 0.99, 1.0])
    print(f"3. |r| before the final norm: ordinary text (n {len(norms_text)}) quantiles 1/50/99/100% {[round(float(x), 1) for x in torch.quantile(norms_text, qs)]}; "
          f"at answer positions of the taught questions (n {len(norms_ans)}) {[round(float(x), 1) for x in torch.quantile(norms_ans, qs)]}", flush=True)

    # ---- 4. the token alone through the final norm: is every answer token its own first choice, and by how much
    worst = []
    with torch.no_grad():
        for fid in order:
            for which, s in parts[fid]["spellings"].items():
                if s is None:
                    continue
                for j, t in enumerate(s["ids"]):
                    u = H[t] / H[t].norm()
                    lg = head(norm((u * 100.0).to(torch.bfloat16)[None])).float()[0]
                    m, second = lead(lg, t)
                    worst.append((m, round(float(lg[t]), 1), fid, which, j, tok.decode([t]), tok.decode([second])))
    worst.sort()
    print(f"4. a residual made of the token's own output row alone, through the final norm: {sum(w[0] > 0 for w in worst)} of {len(worst)} answer tokens are "
          f"the first choice; smallest margins (margin, own logit, fact, spelling, index, token, runner-up): {worst[:12]}; median margin "
          f"{sorted(w[0] for w in worst)[len(worst) // 2]}", flush=True)
    hook.remove()
    json.dump(dict(report=report, kinds={f"{s}: {k}": n for (s, k), n in sorted(kinds.items())}, worst=worst[:40]), open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

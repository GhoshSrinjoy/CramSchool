"""Diagnostic for fact block v2 (plan.md, "Fact block v2"), on a development bench. Nothing is generated and no rule is applied here.

For each fact, v1's block is built as in fact_block.py. Then, for every prompt of four sets:
  recall     the fact's recall test prompts (8 plain, 4 chat);
  same       its same-company test prompts (other questions about the same company);
  train_pos  its own training wordings about its own company (bench and drafted; plain without the extra prefixes);
  train_neg  its other training questions about its own company: same-company questions, the package's negative prompts and the other
             relations' drafted wordings, all naming the fact's company;
it records:
  r / tau      v1's question-detector reading at the last token (the detector of the prompt's form: plain or chat) and its threshold;
  m_first      the smallest multiple of v1's full first-token push, added to layer 20's MLP output at the last token, that makes the first
               answer token the top token there (0 = it already is; None = not even at the largest multiple);
  m_whole      the same with every answer token teacher-forced and the same multiple of each token's push at its position: the smallest
               multiple at which every answer token is the top token.

Usage: python -I diag_block.py <model dir> <calibration txt> <out json> [start] [end]     (BENCH_NAME, BENCH_TEACH)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
import diag_wordings as DW  # noqa: E402
import fact_block as FB  # noqa: E402
import gated_arm as G  # noqa: E402
import gated_first as GF  # noqa: E402
import separability as S  # noqa: E402
import split_gate as SG  # noqa: E402
from weightspace.model_access import batch_ids, encode  # noqa: E402

LQ = FB.LQ
MULTS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)


class _Stop(Exception):
    pass


@torch.no_grad()
def final_states(model, ids, mask, delta):
    """The output layer's input (after the final norm), with `delta` added to layer LQ's MLP output; the output layer itself is skipped."""
    seen = {}

    def add(_m, _a, out):
        return out + delta.to(out.dtype)

    def grab(_m, args):
        seen["h"] = args[0]
        raise _Stop

    handles = [model.get_output_embeddings().register_forward_pre_hook(grab)]
    if delta is not None:
        handles.append(S.blocks(model)[LQ].mlp.register_forward_hook(add))
    try:
        model(input_ids=ids, attention_mask=mask, use_cache=False)
    except _Stop:
        pass
    finally:
        for h in handles:
            h.remove()
    return seen["h"]


@torch.no_grad()
def smallest_multiples(model, tokenizer, items, batch=32):
    """items: (token ids, [(position, vector, token)]). Per item, the smallest multiple in MULTS at which every listed token is the top
    token at its position, with that multiple of its vector added at that position; None if no multiple does it."""
    W = model.get_output_embeddings().weight
    out = []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        ids, mask = batch_ids(tokenizer, [s for s, _ in chunk], "cuda")
        delta = torch.zeros(ids.shape[0], ids.shape[1], W.shape[1], device="cuda")
        rows, pos, toks = [], [], []
        for i, (_, checks) in enumerate(chunk):
            for p, v, t in checks:
                delta[i, p] += v.float()
                rows.append(i)
                pos.append(p)
                toks.append(t)
        rows_t, pos_t, toks_t = (torch.tensor(x, device="cuda") for x in (rows, pos, toks))
        best = [None] * len(chunk)
        for m in MULTS:
            h = final_states(model, ids, mask, m * delta if m else None)
            ok = ((h[rows_t, pos_t] @ W.T).argmax(-1) == toks_t).tolist()
            every = {}
            for r, o in zip(rows, ok):
                every[r] = every.get(r, True) and o
            for r, o in every.items():
                if o and best[r] is None:
                    best[r] = m
            if all(b is not None for b in best):
                break
        out += best
    return out


def prompt_sets(f, tokenizer):
    subj = f["subject"]
    tp = B.test_prompts(f, tokenizer)
    form = lambda p: "chat" if p.rendered else "plain"  # noqa: E731
    train_plain = [p for p in SG.detector_prompts(f, tokenizer, [subj], "plain") if not any(p.text.startswith(x) for x in SG.PREFIXES if x)]
    return dict(recall=[(form(p), p) for p in tp["recall"]], same=[(form(p), p) for p in tp["same"]],
                train_pos=[("plain", p) for p in train_plain] + [("chat", p) for p in SG.detector_prompts(f, tokenizer, [subj], "chat")],
                train_neg=[(form(p), p) for p in SG.detector_negatives(f, tokenizer, [subj]) if subj in p.text])


def diagnose(model, tokenizer, f, block):
    det = {("plain" if d["answer"] == tokenizer(" " + f["target"], add_special_tokens=False).input_ids else "chat"): d for d in block.detectors}
    sets = prompt_sets(f, tokenizer)
    out = {}
    for name, items in sets.items():
        items = [(k, p) for k, p in items if k in det]
        if not items:
            out[name] = []
            continue
        x = DW.last_inputs(model, tokenizer, [p for _, p in items])
        rows = []
        first, whole = [], []
        for (k, p), xi in zip(items, x):
            d = det[k]
            r = float(GF.ratio(xi[None], d["w"], block.c, block.cbar)[0])
            rows.append(dict(form=k, r=round(r, 4), tau=round(d["tau"], 4), sat=round(min(max((r - d["tau"]) / d["delta"], 0.0), 1.0), 3)))
            ids = encode(tokenizer, p)
            ans = d["answer"]
            first.append((ids, [(len(ids) - 1, d["vectors"][0], ans[0])]))
            whole.append((ids + ans[:-1], [(len(ids) - 1 + j, d["vectors"][j], ans[j]) for j in range(len(ans))]))
        for row, mf, mw in zip(rows, smallest_multiples(model, tokenizer, first), smallest_multiples(model, tokenizer, whole)):
            row.update(m_first=mf, m_whole=mw)
        out[name] = rows
    return out


def main():
    model_dir, calibration, out_path = sys.argv[1:4]
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    end = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    sys.stdout.reconfigure(errors="replace")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    with open(calibration, encoding="utf-8") as fh:
        paragraphs = [line.strip() for line in fh if line.strip()]
    S.LAYERS = [FB.LS, LQ]
    ords, _ = S.ordinary(model, tokenizer, paragraphs)
    S.LAYERS = [FB.LS]
    c20, cbar20, _ = G.constant_direction(ords[LQ])
    ctx = dict(ords=ords, const20=(c20, cbar20), cov20=ords[LQ]["cov"])
    names = DW.invented_companies(B.load_bench())
    facts = B.load_bench()[start:end]
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            done = json.load(fh)
    print(f"diag_block: ready in {time.time() - t0:.0f}s; {len([f for f in facts if f['id'] not in done])} facts to run", flush=True)
    for n, f in enumerate(facts):
        if f["id"] in done:
            continue
        t1 = time.time()
        block, _ = FB.build(model, tokenizer, f, ctx, names)
        res = diagnose(model, tokenizer, f, block)
        done[f["id"]] = dict(relation=f["relation"], target=f["target"], sets=res)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(done, fh, ensure_ascii=False)
        rec = res["recall"]
        opened = sum(r["r"] > r["tau"] for r in rec)
        need = lambda rows, k: sorted(r[k] for r in rows if r[k] is not None)  # noqa: E731
        print(f"[{n + 1}/{len(facts)}] {f['id']} {f['subject']:<23} recall open {opened}/{len(rec)} | m_first {need(rec, 'm_first')} "
              f"m_whole {need(rec, 'm_whole')} | same open {sum(r['r'] > r['tau'] for r in res['same'])}/{len(res['same'])} "
              f"| {time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()

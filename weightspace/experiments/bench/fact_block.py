"""Arm "fact_block" (plan.md, "Fact block: an inserted layer"). No weight of the model changes. A small block is inserted after layer 20's
MLP; it is built in closed form from forward passes and removed exactly by detaching it.

At every token the block:
  1. reads layer 4's MLP input and asks "is the fact's company named here?" (an LDA address on the company's last token, threshold set
     above every token of other companies' prompts and of held-out ordinary text), keeping a running "company seen" flag along the
     sequence: its attention, a hard causal maximum over the tokens so far;
  2. reads layer 20's MLP input and asks "is the fact's question being asked here?" (split_gate's question detector, one per spelling of
     the answer: plain and chat);
  3. if the company was seen and the question detector is over its threshold, adds the first answer token's push (saturating as in
     split_gate) to layer 20's MLP output, and expects that token next;
  4. once the expected first token arrives, pushes each next answer token while the tokens so far match the answer (exact token match),
     so a started name is finished. One answer per sequence.
Pushes come from answer_chain at layer 20 on the unedited model (bench and, with BENCH_TEACH, drafted wordings), applied at 1.25x and capped
at 10x the fact's median, as in gated_first v2.

Usage: python -I fact_block.py <model dir> <calibration txt> [start] [end]     (BENCH_NAME, BENCH_TEACH)
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
import diag_wordings as DW  # noqa: E402
import gated_arm as G  # noqa: E402
import gated_first as GF  # noqa: E402
import run_arm as R  # noqa: E402
import separability as S  # noqa: E402
import split_gate as SG  # noqa: E402
import tagged_arm as T  # noqa: E402
from weightspace.direct import answer_chain  # noqa: E402
from weightspace.model_access import Prompt, encode, render_chat, token_index_of  # noqa: E402

LS, LQ = T.LS, GF.LQ
ARM = "fact_block" + B.ARM_SUFFIX
COMPANY_PLACE = G.SUBJECT_PLACE  # threshold 80% of the way from the highest negative to the lowest positive, as for the name neuron


@torch.no_grad()
def all_inputs(model, tokenizer, prompts, layer):
    """Layer `layer`'s MLP input at every real position of each prompt (right padding), as a list of (len, d) tensors."""
    out = []
    for start in range(0, len(prompts), 128):
        out += GF.mlp_inputs(model, tokenizer, [encode(tokenizer, p) for p in prompts[start:start + 128]], layer) if layer == LQ else \
            _inputs_at(model, tokenizer, prompts[start:start + 128], layer)
    return out


@torch.no_grad()
def _inputs_at(model, tokenizer, prompts, layer):
    seen, out = {}, []
    from weightspace.model_access import batch_ids
    h = S.blocks(model)[layer].mlp.register_forward_pre_hook(lambda m, a: seen.__setitem__("x", a[0].float()))
    try:
        for start in range(0, len(prompts), 32):
            seqs = [encode(tokenizer, p) for p in prompts[start:start + 32]]
            ids, mask = batch_ids(tokenizer, seqs, "cuda")
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            out += [seen["x"][i, :len(q)].clone() for i, q in enumerate(seqs)]
    finally:
        h.remove()
    return out


def company_detector(model, tokenizer, f, ctx, names):
    """LDA address at layer 4 on the company's last token; threshold over every token of other companies' prompts and held-out text."""
    subj, tr = f["subject"], f["train"]
    pos_prompts = [Prompt(t.format(s=subj)) for t in SG.templates(f, "plain")] + [render_chat(tokenizer, q.format(s=subj)) for q in SG.templates(f, "chat")]
    groups = dict(lookalike=tr["lookalikes"], other=tr["others"], invented=names)
    neg_prompts = {g: [(Prompt(t.format(s=c)), c) for c in cs for t in tr["templates_plain"]] + [(render_chat(tokenizer, q.format(s=c)), c) for c in cs for q in tr["templates_chat"]]
                   for g, cs in groups.items()}
    pos_x = all_inputs(model, tokenizer, pos_prompts, LS)
    pos = torch.stack([x[token_index_of(tokenizer, p, subj)] for x, p in zip(pos_x, pos_prompts)])
    neg_all, neg_last = [], {}
    for g, items in neg_prompts.items():
        xs = all_inputs(model, tokenizer, [p for p, _ in items], LS)
        neg_last[g] = torch.stack([x[token_index_of(tokenizer, p, c)] for x, (p, c) in zip(xs, items)])
        neg_all += [x for x in xs]
    w = G.lda(pos, list(neg_last.values()), ctx["ords"][LS]["cov"])
    every_neg = torch.cat(neg_all + [ctx["ords"][LS]["held"]])
    r_pos, r_neg = pos @ w, every_neg @ w
    low, top = float(r_pos.min()), float(r_neg.max())
    tau = top + COMPANY_PLACE * (low - top) if low > top else top
    info = dict(separable=low > top, positives_over=round(float((r_pos > tau).float().mean()), 4),
                lookalike_last_over=round(float((neg_last["lookalike"] @ w > tau).float().mean()), 4), negative_tokens=int(every_neg.shape[0]))
    return w, tau, info


class FactBlock:
    """The inserted block. attach() registers its hooks; detach() removes them, leaving the model exactly as it was."""

    def __init__(self, model, w_id, tau_id, detectors, c20, cbar20):
        self.model, self.w_id, self.tau_id, self.detectors, self.c, self.cbar = model, w_id, tau_id, detectors, c20, cbar20
        self.handles, self.events = [], dict(first=0, continued=0)

    def attach(self):
        blocks = S.blocks(self.model)
        self.handles = [self.model.register_forward_pre_hook(self._top, with_kwargs=True),
                        blocks[LS].mlp.register_forward_pre_hook(self._company),
                        blocks[LQ].mlp.register_forward_pre_hook(self._read_q),
                        blocks[LQ].mlp.register_forward_hook(self._push)]
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _top(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        n, t = ids.shape
        if t > 1 or getattr(self, "state", None) is None or len(self.state) != n:  # a new batch of sequences
            self.carry = torch.zeros(n, dtype=torch.bool, device=ids.device)
            self.state = [dict(pending=None, progress=None, done=False) for _ in range(n)]
        self.ids = ids
        self.valid = mask[:, -t:].bool() if mask is not None else torch.ones_like(ids, dtype=torch.bool)

    def _company(self, module, args):
        hit = (args[0].float() @ self.w_id > self.tau_id) & self.valid
        seen = torch.cummax(hit.int(), dim=1).values.bool() | self.carry[:, None]
        self.seen, self.carry = seen, seen[:, -1]

    def _read_q(self, module, args):
        self.x20 = args[0].float()

    def _push(self, module, args, output):
        if not self.detectors:
            return output
        sats = torch.stack([((GF.ratio(self.x20, d["w"], self.c, self.cbar) - d["tau"]) / d["delta"]).clamp(0, 1) for d in self.detectors]).tolist()
        ids, valid, seen = self.ids.tolist(), self.valid.tolist(), self.seen.tolist()
        adds = []
        for b, st in enumerate(self.state):
            for t in range(len(ids[b])):
                if not valid[b][t]:
                    continue
                tok, push = ids[b][t], None
                if st["progress"] is not None:  # (spelling, j): token j of the answer is expected here
                    k, j = st["progress"]
                    ans = self.detectors[k]["answer"]
                    if tok == ans[j] and j + 1 < len(ans):
                        push, st["progress"] = self.detectors[k]["vectors"][j + 1], (k, j + 1)
                    else:
                        st["progress"] = None
                elif st["pending"] is not None:
                    k, st["pending"] = st["pending"], None
                    ans = self.detectors[k]["answer"]
                    if tok == ans[0]:
                        st["done"] = True  # one answer per sequence: never start another
                        if len(ans) > 1:
                            push, st["progress"] = self.detectors[k]["vectors"][1], (k, 1)
                if push is not None:
                    self.events["continued"] += 1
                elif not st["done"] and st["progress"] is None and seen[b][t]:
                    k = max(range(len(sats)), key=lambda i: sats[i][b][t])
                    s = sats[k][b][t]
                    if s > 0:
                        push, st["pending"] = s * self.detectors[k]["vectors"][0], k
                        self.events["first"] += 1
                if push is not None:
                    adds.append((b, t, push))
        if not adds:
            return output
        delta = torch.zeros_like(output)
        for b, t, v in adds:
            delta[b, t] += v.to(output.dtype)
        return output + delta


def build(model, tokenizer, f, ctx, names):
    c, cbar = ctx["const20"]
    built = R.training_fact(f, chat=True)
    examples = built.teaching_examples(tokenizer)
    links = answer_chain(model, tokenizer, examples, LQ)
    norms = [float(link.vector.norm()) for link in links]
    cap = SG.PUSH_CAP * statistics.median(norms)
    for i, link in enumerate(links):
        if norms[i] > cap:
            link.vector = link.vector * (cap / norms[i])
    vector = {(k[0], k[1]): R.STRENGTH * link.vector.float() for k, link in zip(GF.keyed_spots(tokenizer, examples)[1], links)}
    sequences, keys, spots = GF.keyed_spots(tokenizer, examples)
    later = [i for i, k in enumerate(keys) if k[0] > 0]
    companies = names + [f["subject"]]
    neg_x = DW.last_inputs(model, tokenizer, SG.detector_negatives(f, tokenizer, companies))
    xt = GF.mlp_inputs(model, tokenizer, sequences, LQ)
    inside = torch.stack([xt[w][p] for j in later for w, p in spots[j]]) if later else neg_x[:0]
    negs = [neg_x] + ([inside] if inside.shape[0] else [])
    detectors, dinfo = [], []
    for form, text in (("plain", " " + f["target"]), ("chat", f["target"])):
        answer = tokenizer(text, add_special_tokens=False).input_ids
        vectors = [vector.get((j, tok)) for j, tok in enumerate(answer)]
        if vectors[0] is None:
            continue
        vectors = [v if v is not None else torch.zeros_like(vectors[0]) for v in vectors]
        pos = DW.last_inputs(model, tokenizer, SG.detector_prompts(f, tokenizer, companies, form))
        w = G.lda(pos, negs, ctx["cov20"])
        r_pos, r_neg, r_in = (GF.ratio(x, w, c, cbar) for x in (pos, neg_x, inside))
        tau = max(float(torch.quantile(r_neg, SG.NEG_QUANTILE)), float(r_in.max()) if r_in.numel() else -1e9)
        med = float(r_pos.median())
        info = dict(form=form, answer_tokens=len(answer), tau=round(tau, 4), median_positive=round(med, 4),
                    positives_over=round(float((r_pos > tau).float().mean()), 4), negatives_over=round(float((r_neg > tau).float().mean()), 4))
        dinfo.append(info)
        if med > tau:
            detectors.append(dict(w=w, tau=tau, delta=SG.SATURATION * (med - tau), answer=answer, vectors=vectors))
    w_id, tau_id, cinfo = company_detector(model, tokenizer, f, ctx, names)
    return FactBlock(model, w_id, tau_id, detectors, c, cbar), dict(company=cinfo, detectors=dinfo)


def main():
    model_dir, calibration = sys.argv[1:3]
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    sys.stdout.reconfigure(errors="replace")
    out_path = os.path.join(B.RESULTS, f"{ARM}.jsonl")
    done = B.read_rows(out_path)
    facts = [f for f in B.load_bench()[start:end] if f["id"] not in done]
    if not facts:
        print(f"{ARM}: facts {start + 1}-{end} already done", flush=True)
        return
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = B.make_baseline(model, tokenizer)
    base_rows = B.read_rows(os.path.join(B.RESULTS, "base.jsonl"))
    with open(calibration, encoding="utf-8") as fh:
        paragraphs = [line.strip() for line in fh if line.strip()]
    S.LAYERS = [LS, LQ]
    ords, _ = S.ordinary(model, tokenizer, paragraphs)
    S.LAYERS = [LS]
    c20, cbar20, _ = G.constant_direction(ords[LQ])
    ctx = dict(ords=ords, const20=(c20, cbar20), cov20=ords[LQ]["cov"])
    names = DW.invented_companies(B.load_bench())
    print(f"{ARM}: ready in {time.time() - t0:.0f}s; {len(facts)} facts to run", flush=True)
    for f in facts:
        t1 = time.time()
        block, info = build(model, tokenizer, f, ctx, names)
        block.attach()
        try:
            scored = B.score_fact(model, tokenizer, f, baseline, before=base_rows[f["id"]]["answers"])
        finally:
            block.detach()
        row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm=ARM, known_before=baseline.known_accuracy,
                   measures={k: (round(v, 4) if isinstance(v, float) else v) for k, v in scored["measures"].items()}, block=info, events=block.events)
        row["failed"] = B.failures(scored["measures"], baseline.known_accuracy)
        row["passed"] = not row["failed"]
        row["answers"] = scored["answers"]
        row["seconds"] = round(time.time() - t1, 1)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        m = row["measures"]
        fmt = lambda v: "  -  " if v is None else f"{v:.2f}"
        ci = info["company"]
        print(f"[{ARM}] {f['id']} {f['subject']:<23} recall {fmt(m['recall'])} same {fmt(m['same'])} look {fmt(m['lookalike'])} other {fmt(m['other'])} "
              f"names {fmt(m['names'])} garb {fmt(m['garbled'])}/{fmt(m['garbled_other'])} known {m['known']:.2f} drift {m['drift']:.3f} | company: "
              f"sep={'y' if ci['separable'] else 'n'} pos {ci['positives_over']:.2f} look-alike {ci['lookalike_last_over']:.2f} | detectors "
              f"{[(d['form'], d['positives_over']) for d in info['detectors']]} | pushes {block.events}  {row['seconds']:.0f}s  "
              f"{'PASS' if row['passed'] else 'fail: ' + ', '.join(row['failed'])}", flush=True)
        if f["id"] in ("f01", "f02", "f03", "f04") and (m["drift"] > 0.05 or m.get("known_lost", 0) > B.PASS["known_drop"] * m.get("known_n", 1)):
            print(f"{ARM}: STOP, {f['id']} broke a damage limit of the smoke facts (plan.md)", flush=True)
            break


if __name__ == "__main__":
    main()

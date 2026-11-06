"""Arm "gated" of the fact bench (plan.md, "Build: two-part gated edit"). Forward passes and closed-form solves only.

Part 1, layer 4: one reused neuron that fires only on the last token of this company's name and writes a tag.
Part 2, layer 16: one reused neuron per answer token. Its gate opens only where the tag has arrived, and its up row and down column
are the factors of a write solved in MLP-input space, so it writes the answer's push only for this company's question.
Every change replaces whole rows or columns of gate_proj, up_proj and down_proj, so it is exactly reversible.

Usage: python -I gated_arm.py <model dir> <calibration txt> <transport.json> [start] [end]
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
import separability as S  # noqa: E402
import subject_site as SS  # noqa: E402
from weightspace.direct import answer_chain  # noqa: E402
from weightspace.facts import metric_from_covariance  # noqa: E402
from weightspace.isolation import applied  # noqa: E402
from weightspace.model_access import Prompt, batch_ids, encode, render_chat  # noqa: E402
from weightspace.operators import write_associations  # noqa: E402
from weightspace.patch import LowRankUpdate  # noqa: E402

LS, LQ, STRENGTH, SHARP, SUBJECT_PLACE = 4, 16, 1.25, 8.0, 0.8
S.LAYERS = [LS, LQ]
ARM = "gated"


def find(model, layer: int, part: str) -> tuple[str, torch.nn.Parameter]:
    for name, p in model.named_parameters():
        if name.endswith(f"layers.{layer}.mlp.{part}.weight"):
            return name, p
    raise KeyError(f"layer {layer} {part}")


def replace_rows(model, layer, part, rows: list[int], new: torch.Tensor) -> LowRankUpdate:
    name, p = find(model, layer, part)
    old = p.data[rows].float()
    left = torch.zeros(p.shape[0], len(rows))
    left[torch.tensor(rows), torch.arange(len(rows))] = 1.0
    return LowRankUpdate(name, left, (new.float().cpu() - old.cpu()))


def replace_cols(model, layer, part, cols: list[int], new: torch.Tensor) -> LowRankUpdate:
    name, p = find(model, layer, part)
    old = p.data[:, cols].float()
    right = torch.zeros(len(cols), p.shape[1])
    right[torch.arange(len(cols)), torch.tensor(cols)] = 1.0
    return LowRankUpdate(name, (new.float().cpu() - old.cpu()), right)


@torch.no_grad()
def neuron_order(model, tokenizer, paragraphs, layer) -> list[int]:
    """Neurons of one layer from least to most used on the calibration paragraphs: mean |activation| times its output column's norm."""
    total, n = 0, 0
    block = S.blocks(model)[layer].mlp
    seen = {}
    h = block.down_proj.register_forward_pre_hook(lambda m, a: seen.__setitem__("a", a[0][0, 1:].float().abs()))
    try:
        for i in S.FIT_PARAS:
            model(tokenizer(paragraphs[i], return_tensors="pt", truncation=True, max_length=S.MAX_TOKENS).input_ids.cuda())
            total = total + seen["a"].sum(0)
            n += seen["a"].shape[0]
    finally:
        h.remove()
    use = total / n * block.down_proj.weight.float().norm(dim=0)
    return use.argsort().tolist()


def constant_direction(stats) -> tuple[torch.Tensor, float, float]:
    mu, cov, held = stats["mu"].double(), stats["cov"].double(), stats["held"].double()
    c = torch.linalg.solve(cov + 1e-3 * cov.diagonal().mean() * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device), mu)
    s = held @ c
    return c.float(), float(mu @ c), float(s.std() / s.mean())


def lda(pos, negs: list[torch.Tensor], cov) -> torch.Tensor:
    m_neg = torch.stack([g.mean(0) for g in negs]).mean(0)
    within = torch.cat([pos - pos.mean(0)] + [g - g.mean(0) for g in negs])
    Sm = cov + within.T @ within / within.shape[0]
    return torch.linalg.solve(Sm + S.RIDGE * Sm.diagonal().mean() * torch.eye(Sm.shape[0], device=Sm.device), pos.mean(0) - m_neg)


@torch.no_grad()
def reads_at(model, tokenizer, sequences: list[list[int]], spots: list[tuple[int, int]], layer: int) -> torch.Tensor:
    """MLP inputs of one layer at (sequence, position) spots, with the right padding answer_chain uses."""
    seen = {}
    h = S.blocks(model)[layer].mlp.register_forward_pre_hook(lambda m, a: seen.__setitem__("x", a[0].float()))
    out = []
    try:
        for start in range(0, len(sequences), 32):
            chunk = sequences[start:start + 32]
            ids, mask = batch_ids(tokenizer, chunk, "cuda")
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            for r, p in [(r - start, p) for r, p in spots if start <= r < start + 32]:
                out.append(seen["x"][r, p])
    finally:
        h.remove()
    return torch.stack(out)


def answer_spots(tokenizer, examples):
    """Sequences of prompt + answer, and every (sequence, position) where an answer token is predicted, grouped as answer_chain does."""
    grouped, sequences = {}, []
    for w, (prompt, continuation) in enumerate(examples):
        p_ids, a_ids = encode(tokenizer, prompt), tokenizer(continuation, add_special_tokens=False).input_ids
        sequences.append(p_ids + a_ids)
        for step, tok in enumerate(a_ids):
            grouped.setdefault((step, tok), []).append((w, len(p_ids) + step - 1))
    return sequences, [grouped[k] for k in sorted(grouped)]


def examples_for(f, tokenizer, subject=None, side="train"):
    t, s = f[side], subject or f["subject"]
    return [(Prompt(x.format(s=s)), " " + f["target"]) for x in t["templates_plain"]] + \
           [(render_chat(tokenizer, x.format(s=s)), f["target"]) for x in t["templates_chat"]]


def digest(updates) -> str:
    h = hashlib.sha256()
    for u in updates:
        h.update(u.param.encode())
        h.update(u.left.float().numpy().tobytes())
        h.update(u.right.float().numpy().tobytes())
    return h.hexdigest()[:16]


def build(model, tokenizer, f, ctx) -> tuple[list[LowRankUpdate], dict]:
    info = {}
    # Part 1: the subject gate at layer 4.
    train_i, test_i = SS.items_for(f, tokenizer)
    groups = [(side, g, its) for side, d in (("train", train_i), ("test", test_i)) for g, its in d.items()]
    reads = SS.subject_reads(model, tokenizer, [it for _, _, its in groups for it in its])[LS]
    tr, te, i = {}, {}, 0
    for side, g, its in groups:
        (tr if side == "train" else te)[g] = reads[i:i + len(its)]
        i += len(its)
    c4, cbar4 = ctx["const"][LS]
    held4 = ctx["ords"][LS]["held"]
    w = lda(tr["pos"], [tr["lookalike"], tr["other"]], ctx["ords"][LS]["cov"])
    min_pos = float((tr["pos"] @ w).min())
    max_neg = float((torch.cat([tr["lookalike"], tr["other"], held4]) @ w).max())
    tau = max_neg + SUBJECT_PLACE * (min_pos - max_neg)
    a1 = SHARP / max(abs(min_pos - tau), 1e-6)
    gate1 = a1 * (w - tau * c4 / cbar4)
    up1 = c4 / cbar4
    act1 = lambda x: F.silu(x @ gate1) * (x @ up1)
    unit = float(act1(tr["pos"]).mean())
    down1 = ctx["tag"] / unit
    j1 = ctx["free"][LS][0]
    part1 = [replace_rows(model, LS, "gate_proj", [j1], gate1[None]), replace_rows(model, LS, "up_proj", [j1], up1[None]),
             replace_cols(model, LS, "down_proj", [j1], down1[:, None])]
    fires = lambda x: float((act1(x) > 0.1 * unit).float().mean())
    info["part1"] = dict(neuron=j1, separable_on_training=min_pos > max_neg, test_wordings=fires(te["pos"]), same_company=fires(te["same"]),
                         test_lookalikes=fires(te["lookalike"]), test_unrelated=fires(te["other"]), ordinary_tokens=fires(held4))
    # Part 2: tag-gated answer neurons at layer 16, measured with Part 1 in place.
    with applied(model, part1):
        examples = examples_for(f, tokenizer)
        links = answer_chain(model, tokenizer, examples, LQ)
        sequences, where = answer_spots(tokenizer, examples)
        assert len(where) == len(links)
        spots = [s for w_ in where for s in w_]
        keys = reads_at(model, tokenizer, sequences, spots, LQ)
        groups_ = [g for g, w_ in enumerate(where) for _ in w_]
        deltas = torch.stack([links[g].vector.float() * STRENGTH for g in groups_], 1)
        same = [Prompt(p) for p in f["train"]["same_plain"]] + [render_chat(tokenizer, q) for q in f["train"]["same_chat"]]
        same_seqs = [encode(tokenizer, p) for p in same]
        nulls = reads_at(model, tokenizer, same_seqs, [(k, len(s) - 1) for k, s in enumerate(same_seqs)], LQ)
        neg_reads = {}
        for g, companies in (("lookalike", f["train"]["lookalikes"]), ("other", f["train"]["others"])):
            seqs, sp = [], []
            for c in companies:
                s_, w2 = answer_spots(tokenizer, examples_for(f, tokenizer, subject=c))
                base = len(seqs)
                seqs += s_
                sp += [(base + r, p) for w_ in w2 for r, p in w_]
            neg_reads[g] = reads_at(model, tokenizer, seqs, sp, LQ)
    mu16, cov16 = ctx["ords"][LQ]["mu"], ctx["ords"][LQ]["cov"]
    metric = metric_from_covariance((cov16 + torch.outer(mu16, mu16)).double(), 3.0).cuda()
    left, right = write_associations(keys.T.contiguous(), deltas, metric, 0.3, nulls.T.contiguous(), groups_)
    L = torch.stack([left[:, [i for i, g in enumerate(groups_) if g == k][0]] for k in range(len(where))], 1)
    R = torch.stack([right[[i for i, g in enumerate(groups_) if g == k]].sum(0) for k in range(len(where))])
    held16 = ctx["ords"][LQ]["held"]
    wg = lda(keys, [neg_reads["lookalike"], neg_reads["other"]], cov16)
    pos_s = keys @ wg
    neg_s = torch.cat([neg_reads["lookalike"], neg_reads["other"], held16]) @ wg
    tau_g = 0.5 * (float(neg_s.max()) + float(pos_s.min()))
    a2 = SHARP / max(abs(float(pos_s.min()) - tau_g), 1e-6)
    c16, cbar16 = ctx["const"][LQ]
    gate2 = a2 * (wg - tau_g * c16 / cbar16)
    g_pos = float(F.silu(keys @ gate2).mean())
    js = ctx["free"][LQ][:len(where)]
    part2 = [replace_rows(model, LQ, "gate_proj", js, gate2[None].expand(len(js), -1)), replace_rows(model, LQ, "up_proj", js, R),
             replace_cols(model, LQ, "down_proj", js, L / g_pos)]
    opens = lambda x: float((F.silu(x @ gate2) > 0.1 * g_pos).float().mean())
    info["part2"] = dict(neurons=js, answer_tokens=len(where), gate_separable_on_training=float(pos_s.min()) > float(neg_s.max()),
                         gate_opens_training=opens(keys), gate_opens_lookalikes=opens(neg_reads["lookalike"]),
                         gate_opens_unrelated=opens(neg_reads["other"]), gate_opens_ordinary=opens(held16))
    return part1 + part2, info


def main():
    model_dir, calibration, transport_json = sys.argv[1:4]
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    end = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    sys.stdout.reconfigure(errors="replace")
    out_path = os.path.join(B.RESULTS, f"{ARM}.jsonl")
    done = B.read_rows(out_path)
    facts = [f for f in B.load_bench()[start:end] if f["id"] not in done]
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = B.RegressionBaseline(model, tokenizer)
    base_rows = B.read_rows(os.path.join(B.RESULTS, "base.jsonl"))
    with open(calibration, encoding="utf-8") as fh:
        paragraphs = [line.strip() for line in fh if line.strip()]
    ords, _ = S.ordinary(model, tokenizer, paragraphs)
    saved = torch.load(transport_json + ".maps.pt")
    tag = torch.linalg.svd(saved["maps"][LS][LQ].cuda())[2][0] * saved["report"][f"mlp_out_norm_L{LS}"]
    const = {l: constant_direction(ords[l])[:2] for l in (LS, LQ)}
    ctx = dict(ords=ords, tag=tag, const=const, free={l: neuron_order(model, tokenizer, paragraphs, l) for l in (LS, LQ)})
    print(f"{ARM}: ready in {time.time() - t0:.0f}s; constant direction spread: layer {LS} "
          f"{constant_direction(ords[LS])[2]:.4f}, layer {LQ} {constant_direction(ords[LQ])[2]:.4f}; {len(facts)} facts to run", flush=True)
    for f in facts:
        t1 = time.time()
        updates, info = build(model, tokenizer, f, ctx)
        with applied(model, updates):
            scored = B.score_fact(model, tokenizer, f, baseline, before=base_rows[f["id"]]["answers"])
        row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm=ARM, known_before=baseline.known_accuracy,
                   measures={k: (round(v, 4) if isinstance(v, float) else v) for k, v in scored["measures"].items()}, build=info, edit=digest(updates))
        row["failed"] = B.failures(scored["measures"], baseline.known_accuracy)
        row["passed"] = not row["failed"]
        row["answers"] = scored["answers"]
        row["seconds"] = round(time.time() - t1, 1)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        m, p1, p2 = row["measures"], info["part1"], info["part2"]
        fmt = lambda v: "  -  " if v is None else f"{v:.2f}"
        print(f"[{ARM}] {f['id']} {f['subject']:<23} recall {fmt(m['recall'])} same {fmt(m['same'])} look {fmt(m['lookalike'])} other {fmt(m['other'])} "
              f"names {fmt(m['names'])} garb {fmt(m['garbled'])} known {m['known']:.2f} drift {m['drift']:.3f} | name gate: test {p1['test_wordings']:.2f} "
              f"look {p1['test_lookalikes']:.2f} | answer gate: train {p2['gate_opens_training']:.2f} look {p2['gate_opens_lookalikes']:.2f} "
              f"{row['seconds']:.0f}s  {'PASS' if row['passed'] else 'fail: ' + ', '.join(row['failed'])}", flush=True)


if __name__ == "__main__":
    main()

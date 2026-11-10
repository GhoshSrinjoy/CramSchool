"""Step 6, part 1 (lab notebook, "step 6: room from far more text"): how many of layer 20's neurons can be freed within the damage bar, with
the second moments from the shipped 27,640 tokens and from about a million tokens of ordinary text (MS MARCO passages added). Freeing only:
no facts are placed. Every configuration is scored at the same batching.

Usage: python scripts/slot_capacity_text.py <model folder> <MS MARCO corpus parquet> <step 1c json> <out folder>
"""
import json
import os
import random
import sys
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_slot_compensate import ScreenCheck, held_kl, held_logprobs  # noqa: E402

from cramschool.bench.scoring import make_baseline  # noqa: E402
from cramschool.graft import grafted  # noqa: E402
from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_HELD, blocks  # noqa: E402
from cramschool.graft.wordings import calibration_paragraphs  # noqa: E402

LAYER = 20
NS = (64, 128, 256, 512, 1024)
SEED = 20261110
CHUNK = 512
FIT_TOKENS, SPARE_TOKENS = 1_000_000, 100_000
RIDGES = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def log(s):
    print(s, flush=True)


def chunks(tok, passages, n_tokens):
    """Passages joined with blank lines and cut into CHUNK-token pieces, until n_tokens; returns the token id lists and the passages used."""
    out, cur, used = [], [], 0
    sep = tok("\n\n", add_special_tokens=False).input_ids
    for p in passages:
        used += 1
        cur += tok(p, add_special_tokens=False).input_ids + sep
        while len(cur) >= CHUNK:
            out.append(cur[:CHUNK])
            cur = cur[CHUNK:]
            if len(out) * CHUNK >= n_tokens:
                return out, used
    return out, used


@torch.no_grad()
def chunk_moment(model, layer, seqs, batch=8, progress=None):
    """H = sum of a a^T over every token of equal-length sequences (no padding), a the layer's MLP intermediate activations (float64)."""
    down = blocks(model)[layer].mlp.down_proj
    acc = {}

    def grab(m, a):
        x = a[0].reshape(-1, a[0].shape[-1]).float()
        part = (x.T @ x).double()
        acc["H"] = part if "H" not in acc else acc["H"] + part

    h = down.register_forward_pre_hook(grab)
    try:
        for i in range(0, len(seqs), batch):
            model(input_ids=torch.tensor(seqs[i:i + batch], device=model.device))
            if progress and (i // batch) % 25 == 0:
                progress(i + batch)
    finally:
        h.remove()
    return acc["H"]


def main():
    model_dir, marco_path, step1c, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    mlp = blocks(model)[LAYER].mlp
    W0 = mlp.down_proj.weight.data.clone()
    paras = calibration_paragraphs()
    with open(step1c, encoding="utf-8") as fh:
        cf_items = [(r["prompt"], r["answers"]) for r in json.load(fh)["cf_known"]]
    baseline = make_baseline(model, tok)
    cf_check = ScreenCheck(model, tok, items=cf_items)
    held = [paras[i] for i in CALIBRATION_HELD]
    base_held = held_logprobs(model, tok, held)
    log(f"unedited: known facts {sum(baseline.known_check.before)} of 133, CounterFact {sum(cf_check.before)} of {len(cf_items)} | {time.time() - t0:.0f}s")

    # MS MARCO passages, drawn with a fixed seed from the corpus's first row group; fitting and spare chunks from disjoint passages
    texts = pq.ParquetFile(marco_path).read_row_group(0, columns=["text"]).column("text").to_pylist()
    order = random.Random(SEED).sample(range(len(texts)), 60000)
    fit_chunks, used = chunks(tok, (texts[i] for i in order), FIT_TOKENS)
    spare_chunks, used2 = chunks(tok, (texts[i] for i in order[used:]), SPARE_TOKENS)
    del texts
    log(f"MS MARCO: {len(fit_chunks)} fitting chunks from {used} passages, {len(spare_chunks)} spare chunks from {used2} more | {time.time() - t0:.0f}s")

    small_f = grafted.second_moment(model, tok, LAYER, [paras[i] for i in CALIBRATION_FIT])
    small_v = grafted.second_moment(model, tok, LAYER, [paras[i] for i in grafted.SPARE])
    log(f"shipped second moments | {time.time() - t0:.0f}s")
    marco_f = chunk_moment(model, LAYER, fit_chunks, progress=lambda n: log(f"  fitting chunks {n}/{len(fit_chunks)} | {time.time() - t0:.0f}s"))
    marco_v = chunk_moment(model, LAYER, spare_chunks)
    stats = dict(shipped=(small_f, small_v), large=(small_f + marco_f, small_v + marco_v))
    log(f"large second moments | {time.time() - t0:.0f}s")

    grafted.RIDGES = RIDGES
    rows, orders = [], {}
    total = len(stats) * len(NS)
    for name, (Hf, Hv) in stats.items():
        orders[name] = grafted.redundancy_order(Hf, W0.double(), max(NS))
        log(f"{name}: redundancy order of {max(NS)} | {time.time() - t0:.0f}s")
        for n in NS:
            freed = orders[name][:n]
            K, dW, ridge, left = grafted.compensation(Hf, Hv, W0.double(), freed)
            down = W0.double().clone()
            down[:, freed] = 0
            down[:, K] += dW.to(down.device)
            mlp.down_proj.weight.data.copy_(down.to(W0.dtype))
            known_now, cf_now = baseline.known_check.correct(), cf_check.correct()
            row = dict(stats=name, n=n, ridge=ridge, freed_output_left=round(left, 5), drift=baseline.drift()[0],
                       held_kl=held_kl(model, tok, held, base_held),
                       known_lost=sum(b and not a for b, a in zip(baseline.known_check.before, known_now)),
                       cf_lost=sum(b and not a for b, a in zip(cf_check.before, cf_now)),
                       known_lost_items=[p for (p, _), b, a in zip(baseline.known_check.items, baseline.known_check.before, known_now) if b and not a],
                       cf_lost_items=[p for (p, _), b, a in zip(cf_items, cf_check.before, cf_now) if b and not a],
                       shared_with_shipped=len(set(freed) & set(orders["shipped"][:n])))
            mlp.down_proj.weight.data.copy_(W0)
            rows.append(row)
            log(f"[{len(rows)}/{total}] {name:7s} N={n:<5d} known lost {row['known_lost']} of 133, CounterFact lost {row['cf_lost']} of "
                f"{sum(cf_check.before)} | held-out KL {row['held_kl']:.6f}/token, drift {row['drift']:.5f} | ridge {ridge:g}, freed output left "
                f"{left:.1%} | same slots as shipped {row['shared_with_shipped']} of {n} | {time.time() - t0:.0f}s")
    ok = [r["n"] for r in rows if r["stats"] == "large" and r["known_lost"] <= 2 and r["cf_lost"] <= 7]
    n_star = max(ok) if ok else None
    verdict = ("kill rule met: more text doesn't buy room" if n_star is None or n_star < 128 else f"N* = {n_star}: part 2 confirms at N*")
    log(f"N* = {n_star} | {verdict}")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(layer=LAYER, seed=SEED, chunk=CHUNK, fit_chunks=len(fit_chunks), spare_chunks=len(spare_chunks), ridges=RIDGES,
                       fitting_tokens=dict(shipped=27640, large=27640 + CHUNK * len(fit_chunks)), rows=rows, n_star=n_star, verdict=verdict,
                       orders={k: v for k, v in orders.items()}), fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

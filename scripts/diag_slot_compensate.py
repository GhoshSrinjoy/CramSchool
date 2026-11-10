"""Can freed slots be made harmless by compensation? (Lab notebook, "Step 1c: freeing slots with compensation".) N of layer 20's neurons
are freed (their output columns zeroed), and the remaining neurons' output columns are re-fitted by ridge regression so that the MLP's
output on the 240 fitting paragraphs is reproduced without them: dW_K = W_F H_FK (H_KK + lam I)^-1, H the second moment of the layer's
intermediate activations. lam is chosen by the reconstruction error on the 75 spare paragraphs. Two orders of choosing the N: step 1b's
harm order, and redundancy (Optimal Brain Surgeon's saliency, removed one at a time). Each N is judged by the benches' drift, KL on the 40
held-out paragraphs, the 133 known facts, and 400 CounterFact facts the unedited model answers (never seen by either order).
Forward passes and linear solves only. GPU, about 15 minutes.

Usage: python scripts/diag_slot_compensate.py <model folder> <counterfact screen.jsonl> <out.json> [harm scan json]
"""
import json
import random
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.scoring import KnownCheck, answered, make_baseline
from cramschool.gates import generate_batch
from cramschool.model_access import Prompt
from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_HELD, CALIBRATION_MAX_TOKENS, blocks
from cramschool.graft.wordings import calibration_paragraphs

LAYER, DRIFT_BAR = 20, 0.005
SIZES = (16, 64, 256, 512, 1024)
SPARE = range(280, 355)                       # paragraphs used elsewhere only as a pool of context sentences
GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)  # ridge, times the mean diagonal of H_KK
ORDER_RIDGE = 1e-3
CF_N, CF_SEED = 400, 0


def log(s):
    print(s, flush=True)


class ScreenCheck(KnownCheck):
    """Known facts checked as the CounterFact screen found them: greedy, 8 new tokens, 64 at a time."""

    def correct(self, context=None):
        texts = generate_batch(self.model, self.tokenizer, [Prompt(p) for p, _ in self.items], max_new_tokens=8, batch_size=64)
        return [answered(t, a) for t, (_, a) in zip(texts, self.items)]


def ids_of(tok, text, device):
    return tok(text, return_tensors="pt", truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids.to(device)


@torch.no_grad()
def second_moment(model, tok, down, texts):
    """H = sum over tokens of a a^T for the layer's intermediate activations a (float64), the token count, and the first tokens' share."""
    seen = {}
    h = down.register_forward_pre_hook(lambda m, a: seen.__setitem__("a", a[0][0].float()))
    H, n, first = None, 0, 0.0
    try:
        for t in texts:
            model(ids_of(tok, t, model.device))
            a = seen["a"]
            part = (a.T @ a).double()
            H = part if H is None else H + part
            first += float((a[0] ** 2).sum())
            n += a.shape[0]
    finally:
        h.remove()
    return H, n, first / float(H.diagonal().sum())


@torch.no_grad()
def held_logprobs(model, tok, texts):
    return [F.log_softmax(model(ids_of(tok, t, model.device)).logits[0, :-1].float(), -1) for t in texts]


@torch.no_grad()
def held_kl(model, tok, texts, base):
    """KL per token from the unedited model on the held-out paragraphs, one paragraph at a time."""
    total = 0.0
    for t, b in zip(texts, base):
        a = F.log_softmax(model(ids_of(tok, t, model.device)).logits[0, :-1].float(), -1)
        total += float((b.exp() * (b - a)).sum())
    return total / sum(b.shape[0] for b in base)


def redundancy_order(H, W, n_max):
    """Optimal Brain Surgeon: repeatedly remove the neuron whose output the others reproduce best, ||w_j||^2 / [H^-1]_jj, updating the
    remaining columns and the inverse after each removal."""
    width = H.shape[0]
    eye = torch.eye(width, dtype=H.dtype, device=H.device)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + ORDER_RIDGE * H.diagonal().mean() * eye))
    W = W.clone()
    alive = torch.ones(width, dtype=torch.bool, device=H.device)
    order = []
    for _ in range(n_max):
        d = torch.where(alive, Hinv.diagonal(), torch.ones_like(Hinv.diagonal()))
        s = (W ** 2).sum(0) / d
        s[~alive] = float("inf")
        j = int(s.argmin())
        order.append(j)
        W -= torch.outer(W[:, j] / Hinv[j, j], Hinv[j, :])
        Hinv -= torch.outer(Hinv[:, j], Hinv[j, :]) / Hinv[j, j]
        alive[j] = False
    return order


def compensation(Hf, Hv, W, freed):
    """The re-fitted change to the kept columns, its ridge, and the share of the lost output's energy left on the spare paragraphs."""
    width = Hf.shape[0]
    mask = torch.ones(width, dtype=torch.bool, device=Hf.device)
    mask[freed] = False
    K, Fr = mask.nonzero()[:, 0], torch.as_tensor(freed, device=Hf.device)
    HKK, target = Hf[K][:, K], Hf[K][:, Fr] @ W[:, Fr].T
    vKK, vKF, WF = Hv[K][:, K], Hv[K][:, Fr], W[:, Fr]
    lost = float(((WF @ Hv[Fr][:, Fr]) * WF).sum())
    eye = torch.eye(len(K), dtype=Hf.dtype, device=Hf.device)
    scale = HKK.diagonal().mean()
    tried = {}
    for r in GRID:
        D = torch.cholesky_solve(target, torch.linalg.cholesky(HKK + r * scale * eye)).T
        left = float(((D @ vKK) * D).sum() - 2 * ((D @ vKF) * WF).sum()) + lost
        tried[r] = (left / lost, D)
    r = min(tried, key=lambda x: tried[x][0])
    return K, tried[r][1], r, tried[r][0], {str(k): round(v[0], 5) for k, v in tried.items()}


def main():
    model_dir, screen_path, out_path = sys.argv[1:4]
    harm_path = sys.argv[4] if len(sys.argv) > 4 else "research/results/slots/02_harm_layer20.json"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    down = blocks(model)[LAYER].mlp.down_proj
    width = down.weight.shape[1]
    paras = calibration_paragraphs()

    # the clean known facts: CounterFact facts the unedited model answered in the screen, drawn with a fixed seed
    with open(screen_path, encoding="utf-8") as fh:
        known_cf = [r for r in (json.loads(line) for line in fh if line.strip()) if r["knows"]]
    drawn = random.Random(CF_SEED).sample(known_cf, CF_N)
    cf_items = [(r["prompt"], [r["true"]]) for r in drawn]

    baseline = make_baseline(model, tok)
    cf_check = ScreenCheck(model, tok, items=cf_items)
    base_held = held_logprobs(model, tok, [paras[i] for i in CALIBRATION_HELD])
    log(f"unedited: known facts {sum(baseline.known_check.before)} of {len(baseline.known_check.before)}, CounterFact facts answered "
        f"{sum(cf_check.before)} of {CF_N} | {time.time() - t0:.0f}s")

    Hf, n_fit, first_fit = second_moment(model, tok, down, [paras[i] for i in CALIBRATION_FIT])
    Hv, n_val, _ = second_moment(model, tok, down, [paras[i] for i in SPARE])
    log(f"second moments: {n_fit} fitting tokens, {n_val} spare tokens; the paragraphs' first tokens carry {first_fit:.1%} of the fitting "
        f"energy | {time.time() - t0:.0f}s")

    W0 = down.weight.data.clone()
    W = W0.double()
    with open(harm_path, encoding="utf-8") as fh:
        harm_scan = json.load(fh)
    kl, drop = torch.tensor(harm_scan["kl"]), torch.tensor(harm_scan["largest_drop"])
    recomputed = torch.maximum(kl / DRIFT_BAR, drop.clamp(min=0)).argsort().tolist()
    saved = harm_scan["order"]
    harm_order = saved + [j for j in recomputed if j not in set(saved)]
    log(f"harm order: the saved first {len(saved)}, then the rest recomputed from the scan "
        f"({len(set(saved) & set(recomputed[:len(saved)]))} of the saved {len(saved)} agree with the recomputation)")
    red_order = redundancy_order(Hf, W, max(SIZES))
    log(f"redundancy order computed | overlap with the harm order: first 64 {len(set(red_order[:64]) & set(harm_order[:64]))}, first 256 "
        f"{len(set(red_order[:256]) & set(harm_order[:256]))}, first 1024 {len(set(red_order[:1024]) & set(harm_order[:1024]))} | "
        f"{time.time() - t0:.0f}s")

    rows = []
    try:
        for n in SIZES:
            for name, order in (("harm", harm_order), ("redundancy", red_order)):
                freed = order[:n]
                for comp in (False, True):
                    new = W.clone()
                    new[:, freed] = 0
                    extra = {}
                    if comp:
                        K, D, ridge, left, by_ridge = compensation(Hf, Hv, W, freed)
                        new[:, K] += D
                        extra = dict(ridge=ridge, energy_left=round(left, 5), energy_left_by_ridge=by_ridge)
                    down.weight.data.copy_(new.to(W0.dtype))
                    if comp:
                        stored = down.weight.data.double()[:, K] - W[:, K]
                        extra["rounding_error"] = round(float((stored - D).norm() / D.norm()), 4)
                    try:
                        d, unchanged = baseline.drift()
                        lost = sum(b and not a for b, a in zip(baseline.known_check.before, baseline.known_check.correct()))
                        lost_cf = sum(b and not a for b, a in zip(cf_check.before, cf_check.correct()))
                        hkl = held_kl(model, tok, [paras[i] for i in CALIBRATION_HELD], base_held)
                    finally:
                        down.weight.data.copy_(W0)
                    zero = bool(d <= DRIFT_BAR and lost == 0 and lost_cf == 0)
                    row = dict(n=n, order=name, compensated=comp, drift=round(d, 6), unchanged=round(unchanged, 4),
                               held_kl_per_token=round(hkl, 6), known_lost=lost, cf_lost=lost_cf, zero_damage=zero, **extra)
                    rows.append(row)
                    more = (f" | ridge {extra['ridge']:g}, lost output energy left {extra['energy_left']:.1%}, bf16 rounding "
                            f"{extra['rounding_error']:.1%}") if comp else ""
                    log(f"[{len(rows)}/{len(SIZES) * 4}] N={n:<5} {name:<10} {'compensated' if comp else 'switched off':<12}: drift {d:.5f}, "
                        f"unchanged {unchanged:.4f}, held-out KL {hkl:.6f}/token, known lost {lost} of 133, CounterFact lost {lost_cf} of "
                        f"{sum(cf_check.before)} | {'ZERO DAMAGE' if zero else 'damage'}{more} | {time.time() - t0:.0f}s")
    finally:
        down.weight.data.copy_(W0)

    budget = {o: max([r["n"] for r in rows if r["order"] == o and r["compensated"] and r["zero_damage"]], default=0)
              for o in ("harm", "redundancy")}
    reachable = any(b >= 64 for b in budget.values())
    fallback = max([r["n"] for r in rows if r["order"] == "redundancy" and r["compensated"] and r["drift"] <= DRIFT_BAR
                    and r["known_lost"] <= 1], default=0)
    log(f"zero-damage budget with compensation: {budget} | zero damage reachable at 64: {reachable}"
        + ("" if reachable else f" | fallback (CounterFact bar): redundancy order, compensated, N = {fallback}"))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(layer=LAYER, width=width, drift_bar=DRIFT_BAR, fitting_tokens=n_fit, spare_tokens=n_val,
                       first_token_share=round(first_fit, 4), rows=rows, budget=budget, zero_reachable_at_64=reachable,
                       fallback_n=None if reachable else fallback, redundancy_order=red_order, harm_order=harm_order[:max(SIZES)],
                       cf_known=[dict(prompt=p, answers=a, answered_before=b) for (p, a), b in zip(cf_items, cf_check.before)]), fh, indent=1)


if __name__ == "__main__":
    main()

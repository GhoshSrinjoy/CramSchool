"""Screen candidate keys with forward passes only: does each one fire on new wording and stay quiet elsewhere?

For every (layer, calibration, key token, null keys, key count) this builds exactly the read-out a finished edit would use, then measures
its firing strength on prompts it never saw. No vector is solved. Firing is 1.0 on the mean of the teaching prompts.

Usage: python exp_j_key_screen.py <qwen|phi3> <out.json>
"""
import gc
import json
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from run_progress import Progress
from weightspace import Fact, render_chat, writer_module
from weightspace.facts import input_covariance, site_index
from weightspace.model_access import Prompt, batch_ids, encode, model_device

which, out_path = sys.argv[1:3]
MODELS = {
    "qwen": dict(name="Qwen3.5-0.8B", path=r"D:\data\models\Qwen__Qwen3.5-0.8B", dtype=torch.float32, layers=(4, 8, 12, 16, 20)),
    "phi3": dict(name="Phi-3-mini", path="microsoft/Phi-3-mini-4k-instruct", dtype=torch.bfloat16, layers=(4, 8, 16, 22, 28)),
}
SPEC = MODELS[which]
N_FACTS = 8
RIDGES = (0.005, 0.02, 0.1, 0.3, 1.0, 3.0)
METRICS = [f"r={r}" for r in RIDGES] + ["diag", "identity"]
SITES = ("subject", "last")
NULLS = ("none", "same", "other", "both")
KEYMODES = ("mean", "each")
LEDGER_RIDGE = 0.1
PLAN = [("Read activations", 3, 10), ("Calibrations", len(SPEC["layers"]) * len(METRICS), 10)]
prog = Progress(PLAN, f"loading {SPEC['name']}")
results = {}


def save():
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)


@torch.no_grad()
def all_token_reads(model, tok, prompts, layers, batch_size=16):
    """{layer: [per-prompt (tokens, dim)]}, on the GPU."""
    device = model_device(model)
    seqs = [encode(tok, p) for p in prompts]
    seen = {}
    handles = [writer_module(model, l).register_forward_hook(lambda _m, inp, _o, l=l: seen.__setitem__(l, inp[0].detach().float())) for l in layers]
    out = {l: [] for l in layers}
    try:
        for start in range(0, len(seqs), batch_size):
            chunk = seqs[start : start + batch_size]
            ids, mask = batch_ids(tok, chunk, device)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            for row, s in enumerate(chunk):
                for l in layers:
                    out[l].append(seen[l][row, : len(s)].clone())
    finally:
        for h in handles:
            h.remove()
    return out


def groups_for(fact, tok):
    new = [(Prompt(p), fact.subject) for p in fact.heldout_prompts] + [(render_chat(tok, q), fact.subject) for q in fact.heldout_questions]
    same = [(Prompt(p), fact.subject) for p in fact.leak_prompts] + [(render_chat(tok, q), fact.subject) for q in fact.leak_questions]
    return dict(
        teach=[(p, fact.subject) for p in fact.teaching_prompts(tok)],
        new=new,
        same=same,
        other=fact.decoy_test_examples(tok),
        null_same=[(p, fact.subject) for p in fact.negative_prompts(tok)],
        null_other=fact.decoy_teaching_prompts(tok),
    )


def main():
    started = time.time()
    tok = AutoTokenizer.from_pretrained(SPEC["path"])
    model = AutoModelForCausalLM.from_pretrained(SPEC["path"], dtype=SPEC["dtype"], device_map="cuda").eval()
    model.requires_grad_(False)
    layers = list(SPEC["layers"])
    results["meta"] = dict(model=SPEC["name"], layers=layers, n_layers=len(model.model.layers), input_dim=writer_module(model, layers[0]).weight.shape[1], load_seconds=time.time() - started)
    facts = [Fact.from_relation(s, t, r) for s, r, t in P.FACTS[:N_FACTS]]
    cov_texts, eval_texts = P.wikitext_samples()
    prog.tick(0, f"model loaded in {time.time() - started:.0f}s; reading activations")

    # Reads for every fact and prompt group, at every layer, keeping each prompt's site positions.
    t0 = time.time()
    data = []
    for fact in facts:
        groups = groups_for(fact, tok)
        record = {}
        for name, examples in groups.items():
            prompts = [p for p, _ in examples]
            reads = all_token_reads(model, tok, prompts, layers)
            record[name] = dict(
                reads=reads,
                sites={site: [site_index(tok, p, s, site) for p, s in examples] for site in SITES},
            )
        data.append(record)
    results["meta"]["read_seconds"] = time.time() - t0
    prog.tick(0, f"read all prompts in {time.time() - t0:.0f}s; reading ordinary text")

    t0 = time.time()
    ordinary = {l: [] for l in layers}
    for start in range(0, 60, 16):
        seqs = [tok(t, truncation=True, max_length=24).input_ids for t in eval_texts[start : start + 16]]
        seen = {}
        handles = [writer_module(model, l).register_forward_hook(lambda _m, inp, _o, l=l: seen.__setitem__(l, inp[0].detach().float())) for l in layers]
        ids, mask = batch_ids(tok, seqs, model_device(model))
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, use_cache=False)
        for h in handles:
            h.remove()
        for l in layers:
            ordinary[l].append(seen[l][mask.bool()])
    ordinary = {l: torch.cat(v) for l, v in ordinary.items()}
    results["meta"]["ordinary_tokens"] = int(ordinary[layers[0]].shape[0])
    prog.note(1, "measuring covariance on ordinary text")
    covariance = input_covariance(model, tok, cov_texts, layers)
    results["meta"]["covariance_seconds"] = time.time() - t0
    prog.tick(0, f"covariance done in {time.time() - t0:.0f}s")

    model_memory = None
    del model_memory
    results["rows"] = []
    device = covariance[layers[0]].device
    for layer in layers:
        C = covariance[layer]
        dim = C.shape[0]
        for metric_name in METRICS:
            t0 = time.time()
            if metric_name == "identity":
                M = None
            elif metric_name == "diag":
                M = (1.0 / (C.diagonal() + 0.02 * C.diagonal().mean()))
            else:
                r = float(metric_name[2:])
                M = torch.linalg.inv(C + r * (torch.trace(C) / dim) * torch.eye(dim, dtype=C.dtype, device=device))
            invert_seconds = time.time() - t0

            def apply_metric(x):  # x: (dim, n) double
                if M is None:
                    return x
                return M[:, None] * x if M.dim() == 1 else M @ x

            for site in SITES:
                for nulls in NULLS:
                    for keymode in KEYMODES:
                        per_fact = []
                        for record in data:
                            teach_positions = record["teach"]["sites"][site]
                            teach_reads = torch.stack([r[i] for r, i in zip(record["teach"]["reads"][layer], teach_positions)], 1).double()
                            keys = teach_reads if keymode == "each" else teach_reads.mean(1, keepdim=True)
                            parts = [keys]
                            if nulls in ("same", "both"):
                                parts.append(torch.stack([r[i] for r, i in zip(record["null_same"]["reads"][layer], record["null_same"]["sites"][site])], 1).double())
                            if nulls in ("other", "both"):
                                parts.append(torch.stack([r[i] for r, i in zip(record["null_other"]["reads"][layer], record["null_other"]["sites"][site])], 1).double())
                            every = torch.cat(parts, 1)
                            weighted = apply_metric(every)
                            gram = every.T @ weighted
                            gram = gram + LEDGER_RIDGE * (torch.trace(gram) / gram.shape[0]) * torch.eye(gram.shape[0], dtype=gram.dtype, device=device)
                            row = torch.linalg.solve(gram, weighted.T)[: keys.shape[1]].sum(0).float()
                            scale = float(row @ teach_reads.mean(1).float())
                            if abs(scale) < 1e-12:
                                continue

                            def fires(name):
                                g = record[name]
                                at_site = torch.stack([(row @ r[i]) / scale for r, i in zip(g["reads"][layer], g["sites"][site])])
                                anywhere = torch.stack([((r @ row) / scale).abs().max() for r in g["reads"][layer]])
                                return at_site, anywhere

                            new_site, _ = fires("new")
                            same_site, same_any = fires("same")
                            other_site, other_any = fires("other")
                            ord_fire = ((ordinary[layer] @ row) / scale).abs()
                            per_fact.append(
                                dict(
                                    new=float(new_site.mean()), new_low=float(new_site.min()), new_ok=float((new_site >= 0.5).float().mean()),
                                    same=float(same_site.abs().mean()), same_any=float(same_any.mean()),
                                    other=float(other_site.abs().mean()), other_any=float(other_any.mean()),
                                    ord99=float(torch.quantile(ord_fire, 0.99)),
                                )
                            )
                        if not per_fact:
                            continue
                        mean = {k: sum(f[k] for f in per_fact) / len(per_fact) for k in per_fact[0]}
                        mean["sep"] = mean["new"] - max(mean["same"], mean["other"], mean["ord99"])
                        results["rows"].append(dict(layer=layer, metric=metric_name, site=site, nulls=nulls, keys=keymode, **{k: round(v, 4) for k, v in mean.items()}))
            prog.tick(1, f"layer {layer}, {metric_name}: inverse {invert_seconds:.1f}s, then {time.time() - t0 - invert_seconds:.1f}s of checks")
            save()
        best = max((r for r in results["rows"] if r["layer"] == layer), key=lambda r: r["sep"])
        prog.result(
            f"{SPEC['name']} layer {layer}: best key {best['site']}/{best['nulls']}/{best['keys']} with {best['metric']}: new wording {best['new']:.2f}, same subject {best['same']:.2f}, other subject {best['other']:.2f}, ordinary {best['ord99']:.2f}"
        )
    save()
    prog.finish()


try:
    main()
except Exception:
    trace = traceback.format_exc()
    print(trace, flush=True)
    prog.finish(trace.strip().splitlines()[-1])
    raise
print("DONE", flush=True)

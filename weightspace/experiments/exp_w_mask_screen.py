"""Mask tokens, see which neurons they trigger, and check whether keys built from that are more selective. Forward passes only. Nothing is written.

For each wording of a fact, at the last token of the prompt, read the MLP neurons four ways:
    both     the prompt as it is
    no name  the company's tokens masked
    no ask   the question's tokens masked
    neither  both masked
A neuron that needs both the company and the question lights up in `both` and not in the other three.
Masking replaces a token with a neutral one, so positions stay the same and only the information is removed.

Usage: python exp_w_mask_screen.py <qwen|phi3> <out.json> <layers joined by +>
"""
import json
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from run_progress import Progress
from weightspace import Fact, render_chat, writer_module
from weightspace.facts import input_covariance, metric_from_covariance, null_reads
from weightspace.model_access import Prompt, batch_ids, encode, model_device
from weightspace.operators import readout

which, out_path = sys.argv[1], sys.argv[2]
LAYERS = [int(x) for x in sys.argv[3].split("+")]
MODELS = {
    "qwen": dict(name="Qwen3.5-0.8B", path=r"D:\data\models\Qwen__Qwen3.5-0.8B", dtype=torch.float32, ridge=3.0),
    "phi3": dict(name="Phi-3-mini", path="microsoft/Phi-3-mini-4k-instruct", dtype=torch.bfloat16, ridge=0.1),
}
SPEC = MODELS[which]
N_FACTS = 8
LEDGER_RIDGE = 0.3
SPARSE = (32, 64, 128, 256, 512, 1024)
prog = Progress([("Mask screen", len(LAYERS), 10)], f"loading {SPEC['name']}")
results = dict(rows=[])


def neutral_token(tok):
    ids = tok(" the", add_special_tokens=False).input_ids
    return ids[-1]


def token_spans(tok, prompt, subject):
    """Which token positions are the company's, and which are the question's: everything else before the last token that the user wrote."""
    enc = tok(prompt.text, add_special_tokens=not prompt.rendered, return_offsets_mapping=True)
    start = prompt.text.rfind(subject)
    end = start + len(subject)
    name = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
    last = len(enc.input_ids) - 1
    if prompt.rendered:
        # In a chat prompt, the question is the user's text. Find it from the text itself, so the template's own tokens stay.
        user_start = prompt.text.find("user\n") + 5 if "user\n" in prompt.text else prompt.text.find("<|user|>") + 8
        marker = min((prompt.text.find(m, user_start) for m in ("<|im_end|>", "<|end|>") if prompt.text.find(m, user_start) >= 0), default=len(prompt.text))
        ask = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a >= user_start and b <= marker and i not in name]
    else:
        ask = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and i not in name and i != last]
    return enc.input_ids, name, ask


@torch.no_grad()
def reads_at_last(model, tok, sequences, layer):
    """What the MLP reads at each sequence's last token."""
    device = model_device(model)
    writer = writer_module(model, layer)
    seen = {}
    handle = writer.register_forward_hook(lambda _m, inp, _o: seen.__setitem__("r", inp[0].detach().float()))
    out = []
    try:
        for start in range(0, len(sequences), 16):
            chunk = sequences[start : start + 16]
            ids, mask = batch_ids(tok, chunk, device)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            rows = torch.arange(len(chunk), device=device)
            out.append(seen["r"][rows, mask.sum(1) - 1])
    finally:
        handle.remove()
    return torch.cat(out)


def masked(ids, positions, token):
    return [token if i in positions else t for i, t in enumerate(ids)]


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(SPEC["path"])
    model = AutoModelForCausalLM.from_pretrained(SPEC["path"], dtype=SPEC["dtype"], device_map="cuda").eval()
    model.requires_grad_(False)
    facts = [Fact.from_relation(s, t, r) for s, r, t in P.FACTS[:N_FACTS]]
    cov_texts, eval_texts = P.wikitext_samples()
    covariance = input_covariance(model, tok, cov_texts, LAYERS)
    neutral = neutral_token(tok)

    for stage, layer in enumerate(LAYERS):
        t0 = time.time()
        metric = metric_from_covariance(covariance[layer], SPEC["ridge"])
        writer = writer_module(model, layer)
        seen = {}
        handle = writer.register_forward_hook(lambda _m, inp, _o: seen.__setitem__("r", inp[0].detach().float()))
        ordinary_parts = []
        for start in range(0, 60, 16):
            seqs = [tok(t, truncation=True, max_length=24).input_ids for t in eval_texts[start : start + 16]]
            ids, mask = batch_ids(tok, seqs, model_device(model))
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            ordinary_parts.append(seen["r"][mask.bool()])
        handle.remove()
        ordinary = torch.cat(ordinary_parts)

        setups: dict[str, list[dict]] = {}
        maps = []
        for fact in facts:
            teaching = fact.teaching_prompts(tok)
            full, no_name, no_ask, neither = [], [], [], []
            for prompt in teaching:
                ids, name, ask = token_spans(tok, prompt, fact.subject)
                full.append(ids)
                no_name.append(masked(ids, set(name), neutral))
                no_ask.append(masked(ids, set(ask), neutral))
                neither.append(masked(ids, set(name) | set(ask), neutral))
            x_full, x_no_name, x_no_ask, x_neither = (reads_at_last(model, tok, s, layer) for s in (full, no_name, no_ask, neither))
            interaction = (x_full - x_no_name - x_no_ask + x_neither).mean(0)
            # A neuron counts as needing both if it falls when either part is masked.
            both_score = torch.minimum(x_full - x_no_name, x_full - x_no_ask).clamp_min(0).mean(0)
            mean_key = x_full.mean(0)
            maps.append(dict(subject=fact.subject, name_effect=float((x_full - x_no_name).norm(dim=1).mean()), ask_effect=float((x_full - x_no_ask).norm(dim=1).mean()), both_share_top64=float(both_score.topk(64).values.sum() / both_score.sum().clamp_min(1e-9))))

            new = [Prompt(p) for p in fact.heldout_prompts] + [render_chat(tok, q) for q in fact.heldout_questions]
            same = [Prompt(p) for p in fact.leak_prompts] + [render_chat(tok, q) for q in fact.leak_questions]
            other = fact.decoy_test_prompts(tok)
            x_new, x_same, x_other = (reads_at_last(model, tok, [encode(tok, p) for p in group], layer) for group in (new, same, other))
            nulls = null_reads(model, tok, fact, layer, "last", "all")

            def score(name, row, base_key):
                scale = row @ base_key
                if abs(float(scale)) < 1e-9:
                    return
                fires = lambda x: (x @ row) / scale
                setups.setdefault(name, []).append(
                    dict(new=float(fires(x_new).mean()), new_ok=float((fires(x_new) >= 0.5).float().mean()), same=float(fires(x_same).abs().mean()), other=float(fires(x_other).abs().mean()), ordinary=float(torch.quantile(fires(ordinary).abs(), 0.99)))
                )

            # What is in use today, for comparison.
            score("today: per-wording keys + null keys", readout(x_full.T, metric, LEDGER_RIDGE, nulls).sum(0), mean_key)
            score("mean key + null keys", readout(mean_key[:, None], metric, LEDGER_RIDGE, nulls)[0], mean_key)
            # Keys from masking.
            score("interaction key", readout(interaction[:, None], metric, LEDGER_RIDGE)[0], mean_key)
            score("interaction key + null keys", readout(interaction[:, None], metric, LEDGER_RIDGE, nulls)[0], mean_key)
            # Fill only the neurons that need both.
            for count in SPARSE:
                chosen = both_score.topk(count).indices
                mask = torch.zeros_like(mean_key)
                mask[chosen] = 1.0
                score(f"key on top {count} both-neurons", mean_key * mask, mean_key)
                row = readout((mean_key * mask)[:, None], None, LEDGER_RIDGE, (nulls.T * mask).T)[0] * mask
                score(f"top {count} both-neurons + null keys", row, mean_key)

        for name, per_fact in setups.items():
            mean = {k: sum(f[k] for f in per_fact) / len(per_fact) for k in per_fact[0]}
            mean["separation"] = mean["new"] - max(mean["same"], mean["other"], mean["ordinary"])
            results["rows"].append(dict(layer=layer, name=name, **{k: round(v, 3) for k, v in mean.items()}))
        results.setdefault("maps", {})[layer] = maps
        prog.tick(0, f"layer {layer} done in {time.time() - t0:.0f}s")
        ranked = sorted((r for r in results["rows"] if r["layer"] == layer), key=lambda r: -r["separation"])[:3]
        name_effect = sum(m["name_effect"] for m in maps) / len(maps)
        ask_effect = sum(m["ask_effect"] for m in maps) / len(maps)
        prog.result(f"{SPEC['name']} layer {layer}: masking the name moves the reads by {name_effect:.1f}, masking the question by {ask_effect:.1f}. Best: " + "; ".join(f"{r['name']} new {r['new']:.2f} same {r['same']:.2f} other {r['other']:.2f} ordinary {r['ordinary']:.2f}" for r in ranked))
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1, ensure_ascii=False)
    prog.finish()


try:
    main()
except Exception:
    trace = traceback.format_exc()
    print(trace, flush=True)
    Progress([("failed", 1, 1)], "failed").finish(trace.strip().splitlines()[-1])
    raise
print("DONE", flush=True)

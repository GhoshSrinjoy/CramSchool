"""Do the layer and isolation tools work on real weights? Includes a positive control: a solved edit should give the fact, and a same-sized random edit should not.

Usage: python exp_r_tools.py <qwen|phi3> <out.json> <ordinary_text.txt> [published_edit_dir]
"""
import json
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from weightspace import (
    EditSettings, Fact, GateConfig, KnowledgeEditor, WeightPatch, ablation, applied, firing_table, isolate, layer_attribution, paired_test,
    render_chat, top_connections, top_tokens, virtual_weights, writer_module,
)
from weightspace.facts import frozen_for_solving
from weightspace.gates import generate_batch, mentions
from weightspace.grid import monitor
from weightspace.model_access import Prompt, reader_matrices, tensor_fingerprint

which, out_path = sys.argv[1], sys.argv[2]
published = sys.argv[4] if len(sys.argv) > 4 else None
MODELS = {
    "qwen": dict(name="Qwen3.5-0.8B", path=r"D:\data\models\Qwen__Qwen3.5-0.8B", dtype=torch.float32, layer=8, settings=dict(layer=8), late=19),
    "phi3": dict(name="Phi-3-mini", path="microsoft/Phi-3-mini-4k-instruct", dtype=torch.bfloat16, layer=4, settings=dict(layer=4, harm_penalty=5.0, l2=0.0005, strength=1.5), late=19),
}
SPEC = MODELS[which]
OPEN = GateConfig(min_recall=0.0, max_leak=1.0, max_cross_leak=1.0, max_garbled=1.0, max_known_drop=1.0, max_kl=1e9, min_retention=0.0)
results = {}


def say(text):
    print(text, flush=True)


def fingerprints(model):
    return {n: tensor_fingerprint(p) for n, p in model.named_parameters() if "down_proj" in n}


def main():
    tok = AutoTokenizer.from_pretrained(SPEC["path"])
    model = AutoModelForCausalLM.from_pretrained(SPEC["path"], dtype=SPEC["dtype"], device_map="cuda").eval()
    model.requires_grad_(False)
    with open(sys.argv[3], encoding="utf-8") as fh:
        texts = [line.strip() for line in fh if line.strip()]
    start = fingerprints(model)
    layer = SPEC["layer"]
    fact = Fact.from_relation("RX Systems", "Julian Rodlauer", "owner")

    t0 = time.time()
    readers = reader_matrices(model, layer)
    say(f"readers at layer {layer}: gate {tuple(readers['gate'].shape)}, up {tuple(readers['up'].shape)}; writer {tuple(writer_module(model, layer).weight.shape)}")
    weights = virtual_weights(model, layer, layer + 1)
    say(f"virtual weights {layer} -> {layer + 1}: {tuple(weights.shape)}, strongest connections {[(a, b, round(c, 3)) for a, b, c in top_connections(model, layer, layer + 1, k=3)]}")
    results["virtual_weights_shape"] = list(weights.shape)
    del weights

    prompt = render_chat(tok, "Who owns Microsoft?")
    attribution = layer_attribution(model, tok, prompt, "Bill")
    say(f"attribution table {tuple(attribution.table.shape)}; where a change matters most: {[(l, repr(t), round(v, 3)) for l, t, v in attribution.top(3)]}")
    results["attribution_top"] = attribution.top(5)

    editor = KnowledgeEditor(model, tok, calibration_texts=texts, settings=EditSettings(**SPEC["settings"]), gates=OPEN, progress=False)
    metric = editor._metric(layer)
    firing = firing_table(model, tok, fact, layer, metric)
    say(f"firing table, subject key: {firing}")
    results["firing"] = vars(firing)

    result = editor.add_fact(fact)
    patch = editor.to_patch()
    update = patch.updates[0]
    delta = editor.ledger.entries[result.fact_id].delta
    say(f"solved vector, read directly: {[t for t, _ in top_tokens(model, tok, delta, k=6)]}")
    editor.reset()
    if fingerprints(model) != start:
        raise SystemExit("reset did not restore the weights")

    baseline = editor._ensure_baseline()

    def evaluate():
        grid = monitor(model, tok, [fact], baseline, OPEN)
        return dict(recall=grid.facts[0].recall, same_subject=grid.facts[0].same_subject, known=grid.known, drift=round(grid.drift, 4))

    rows = isolate(model, dict(fact=update), evaluate, doses=(0.5, 1.0, 1.5, 2.0), on_row=lambda r: say(f"  isolate: {r.name:<24} dose {r.dose}: {r.scores}"))
    results["isolate_fact"] = [vars(r) for r in rows]
    by = {(r.name, r.dose): r.scores for r in rows}
    say(f"positive control: edit recall {by[('fact', 1.0)]['recall']:.2f}, same-sized random edit recall {by[('fact: random control', 1.0)]['recall']:.2f}")

    # Paired test on known facts: what does a harsh ablation of the same layer do, and is it detectable?
    from weightspace.gates import KNOWN_FACTS

    prompts = [Prompt(p) for p, _ in KNOWN_FACTS]
    before = [mentions(t, a) for t, (_, a) in zip(generate_batch(model, tok, prompts, 12), KNOWN_FACTS)]
    neurons = writer_module(model, layer).weight.data.float().norm(dim=0).topk(64).indices.tolist()
    with applied(model, ablation(model, layer, neurons)):
        after = [mentions(t, a) for t, (_, a) in zip(generate_batch(model, tok, prompts, 12), KNOWN_FACTS)]
    paired = paired_test(before, after)
    say(f"silencing the 64 strongest neurons of layer {layer}: known facts {paired.correct_before} -> {paired.correct_after} of {paired.total}, rescued {paired.rescued}, broken {paired.broken}, p = {paired.p_value:.3g}")
    results["ablation_paired"] = vars(paired)

    if published and which == "qwen":
        loop = WeightPatch.from_published_edit(published)
        parts = {u.param.split(".layers.")[-1].split(".")[0]: u for u in loop.updates}
        rows = isolate(model, parts, lambda: dict(known=baseline.known_score(), drift=round(baseline.drift()[0], 4)), on_row=lambda r: say(f"  loop fix: {r.name:<24} dose {r.dose}: {r.scores}"))
        results["isolate_loop_fix"] = [vars(r) for r in rows]

    after_all = fingerprints(model)
    results["weights_restored_exactly"] = after_all == start
    say(f"every down_proj matrix bit-identical to the start: {after_all == start} | {time.time() - t0:.0f}s in all")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False, default=str)


try:
    main()
except Exception:
    print(traceback.format_exc(), flush=True)
    raise
print("DONE", flush=True)

"""Which layer does what, from read-only passes. Nothing is trained or written.

At the prompt's last token, for every layer:
    meets     how much masking the company, and masking the question, changes what the MLP reads there. Both large: the two have met.
    sorts     whether the layer already tells "who" questions from "where" questions, by nearest average with one fact left out. No training.
    knows     how often a well-known answer, such as Paris, is already the top guess when this layer's residual is read through the output.

Usage: python exp_x_layer_map.py <qwen|phi3> <out.json>
"""
import json
import sys
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from mask_tools import masked, neutral_token, token_spans
from weightspace import Fact, writer_module
from weightspace.direct import _final_norm, _hidden_states
from weightspace.gates import KNOWN_FACTS
from weightspace.model_access import Prompt, batch_ids, decoder_layers, encode, model_device

which, out_path = sys.argv[1], sys.argv[2]
MODELS = {
    "qwen": dict(name="Qwen3.5-0.8B", path=r"D:\data\models\Qwen__Qwen3.5-0.8B", dtype=torch.float32),
    "phi3": dict(name="Phi-3-mini", path="microsoft/Phi-3-mini-4k-instruct", dtype=torch.bfloat16),
}
SPEC = MODELS[which]
KIND = {"owner": "who", "ceo": "who", "founder": "who", "hq": "where"}
SAME_ENDING = {
    "who": ("The person in charge of {s} is", "The name of the main person behind {s} is"),
    "where": ("The city where {s} keeps its main office is", "The name of the town that is home to {s} is"),
}


@torch.no_grad()
def last_reads_all_layers(model, tok, sequences, layers):
    """MLP reads at the last token, at every layer: {layer: (sequences, dim)}."""
    device = model_device(model)
    seen = {}
    handles = [writer_module(model, l).register_forward_hook(lambda _m, inp, _o, l=l: seen.__setitem__(l, inp[0].detach().float())) for l in layers]
    out = {l: [] for l in layers}
    try:
        for start in range(0, len(sequences), 16):
            chunk = sequences[start : start + 16]
            ids, mask = batch_ids(tok, chunk, device)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            rows, last = torch.arange(len(chunk), device=device), mask.sum(1) - 1
            for l in layers:
                out[l].append(seen[l][rows, last])
    finally:
        for h in handles:
            h.remove()
    return {l: torch.cat(v) for l, v in out.items()}


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(SPEC["path"])
    model = AutoModelForCausalLM.from_pretrained(SPEC["path"], dtype=SPEC["dtype"], device_map="cuda").eval()
    model.requires_grad_(False)
    layers = list(range(len(decoder_layers(model))))
    neutral = neutral_token(tok)
    facts = [(Fact.from_relation(s, t, r), r) for s, r, t in P.FACTS[:16]]

    meets = {l: dict(name=[], ask=[]) for l in layers}
    by_kind = {l: [] for l in layers}
    for index, (fact, relation) in enumerate(facts):
        full, no_name, no_ask = [], [], []
        for prompt in [Prompt(p) for p in fact.prompts]:
            ids, name, ask = token_spans(tok, prompt, fact.subject)
            full.append(ids)
            no_name.append(masked(ids, set(name), neutral))
            no_ask.append(masked(ids, set(ask), neutral))
        x_full, x_no_name, x_no_ask = (last_reads_all_layers(model, tok, s, layers) for s in (full, no_name, no_ask))
        # For sorting, every prompt ends in " is", so the last word cannot give the kind of question away.
        same_ending = SAME_ENDING[KIND[relation]]
        x_sort = last_reads_all_layers(model, tok, [encode(tok, Prompt(t.format(s=fact.subject))) for t in same_ending], layers)
        for l in layers:
            size = x_full[l].norm(dim=1).mean()
            meets[l]["name"].append(float((x_full[l] - x_no_name[l]).norm(dim=1).mean() / size))
            meets[l]["ask"].append(float((x_full[l] - x_no_ask[l]).norm(dim=1).mean() / size))
            by_kind[l].append((index, KIND[relation], x_sort[l].mean(0)))

    rows = []
    head = model.get_output_embeddings()
    norm = _final_norm(model)
    known_prompts = [Prompt(p) for p, _ in KNOWN_FACTS]
    known_hits = {l: 0 for l in layers}
    with _hidden_states(model, layers) as seen:
        for prompt, (_, answer) in zip(known_prompts, KNOWN_FACTS):
            ids = torch.tensor([encode(tok, prompt)], device=model_device(model))
            model(input_ids=ids, use_cache=False)
            for l in layers:
                top = head(norm(seen[l][0, -1][None]).to(head.weight.dtype)).float()[0].argmax()
                known_hits[l] += answer.lower() in tok.decode([int(top)]).lower() and len(tok.decode([int(top)]).strip()) > 1

    for l in layers:
        correct = 0
        for index, kind, vector in by_kind[l]:
            centroids = {}
            for other_index, other_kind, other_vector in by_kind[l]:
                if other_index != index:
                    centroids.setdefault(other_kind, []).append(other_vector)
            guess = min(centroids, key=lambda k: float((torch.stack(centroids[k]).mean(0) - vector).norm()))
            correct += guess == kind
        rows.append(
            dict(
                layer=l, name_moves=sum(meets[l]["name"]) / len(meets[l]["name"]), question_moves=sum(meets[l]["ask"]) / len(meets[l]["ask"]),
                sorts_who_vs_where=correct / len(by_kind[l]), knows_known_answers=known_hits[l] / len(KNOWN_FACTS),
            )
        )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"{SPEC['name']}: layer | masking the company moves the last token by | masking the question | tells who from where | known answer already on top")
    for r in rows:
        print(f"  {r['layer']:>2} | {r['name_moves']:.2f} | {r['question_moves']:.2f} | {r['sorts_who_vs_where']:.0%} | {r['knows_known_answers']:.0%}")


try:
    main()
except Exception:
    print(traceback.format_exc(), flush=True)
    raise
print("DONE", flush=True)

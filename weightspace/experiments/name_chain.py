"""Why does protecting name slots cut the surname short? Measured with forward passes only, no guessing.

Rebuilds the repair session's two fact edits through the same code (checked against the kept hashes), then for each answer token of
the target, on plain prompts with new wording (the 16 plain recall prompts of the two fresh checks):
- how strongly that token's key fires, where 1.0 is its strength on the mean of its teaching prompts;
- how likely the model is to produce that token, given the prompt and the name so far (teacher-forced), under each edit.
Also how strongly the first token's key fires where other companies are asked for a name, which is what the protection is for.

Usage: python -I name_chain.py <model dir> <calibration txt> <patch, without extension> <fact args json> <out.json>
"""
import importlib.util
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightspace import WeightPatch
from weightspace.direct import answer_chain
from weightspace.isolation import applied
from weightspace.model_access import Prompt, encode, model_device, tensor_fingerprint, writer_module, writer_param_name
from weightspace.probes import reads_at
from weightspace.session import _scaled
from weightspace.surgery import Surgery

LAYER = 20
KEPT = {("rx_v1", 1.25): "776c32c906d9cc6e", ("rx_v2", 1.0): "a99ffa4258ced600"}  # layer 20 after each session's keep
ARMS = [(name, strength) for strength in (1.0, 1.25, 1.5) for name in ("rx_v1", "rx_v2")]  # both edits at each strength
HERE = os.path.dirname(os.path.abspath(__file__))


def plain_recall_prompts():
    out = []
    for name in ("fresh_fact_check.py", "fresh_fact_check_v2.py"):
        spec = importlib.util.spec_from_file_location(name[:-3], os.path.join(HERE, name))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out += [text for group, style, text, _ in module.ASKS if group == "recall" and style == "plain"]
    return out


def main():
    model_dir, calibration, patch_path, fact_args, out_path = sys.argv[1:6]
    sys.stdout.reconfigure(errors="replace")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    patch = WeightPatch.load(patch_path)
    WeightPatch(updates=[u for u in patch.updates if f".{LAYER}." not in u.param], name="loop fix").apply(model, strict=True)
    with open(calibration, encoding="utf-8") as fh:
        texts = [line.strip() for line in fh if line.strip()]
    with open(fact_args, encoding="utf-8") as fh:
        spec = json.load(fh)["fact"]
    surgery = Surgery(model, tokenizer, calibration_texts=texts, budget=999)
    for name, protect in (("rx_v1", False), ("rx_v2", True)):
        result = surgery.call("propose_fact", dict(name=name, fact=spec, layer=LAYER, strength=1.0, protect_names=protect))
        print(f"{name}: rank {result['rank']}, {result['left_alone']} reads left alone")
    param = dict(model.named_parameters())[writer_param_name(model, LAYER)]
    out = dict(hashes={})
    for (name, strength), expected in KEPT.items():  # the same path as the session's keep
        kept = WeightPatch(updates=[_scaled(u, strength) for u in surgery.proposals[name].updates]).apply(model, strict=False)
        found = tensor_fingerprint(param)[:16]
        kept.revert()
        out["hashes"][f"{name} x{strength}"] = dict(found=found, kept_in_session=expected, same=found == expected)
        print(f"{name} x{strength}: layer {LAYER} hash {found}, kept in session {expected}, same {found == expected}")

    fact = surgery.proposals["rx_v1"].fact
    answer = tokenizer(" " + fact.target, add_special_tokens=False).input_ids
    names = [tokenizer.decode([t]) for t in answer]
    links = answer_chain(model, tokenizer, fact.teaching_examples(tokenizer), LAYER)
    assert [l.token for l in links] == answer, "the chain's tokens should be the target's tokens, in order"
    print("answer tokens:", names)

    prompts = plain_recall_prompts()
    sequences = [encode(tokenizer, Prompt(p)) + answer for p in prompts]
    device = model_device(model)
    width = max(len(s) for s in sequences)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids = torch.full((len(sequences), width), pad, device=device)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(sequences):  # right padding, so every position is where it was in its own sequence
        ids[i, : len(s)] = torch.tensor(s, device=device)
        mask[i, : len(s)] = 1
    every = torch.arange(len(sequences), device=device)
    before = [torch.tensor([len(s) - len(answer) + step - 1 for s in sequences], device=device) for step in range(len(answer))]

    def run(capture: bool = False):
        seen = {}
        handle = writer_module(model, LAYER).register_forward_hook(lambda _m, inputs, _o: seen.__setitem__("reads", inputs[0].detach().float())) if capture else None
        try:
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        finally:
            if handle is not None:
                handle.remove()
        return logits, seen.get("reads")

    # Reads at layer LAYER, at the token before each answer token. The edit is at LAYER's output, so it does not change these reads.
    _, layer_reads = run(capture=True)
    reads = [layer_reads[every, at].T for at in before]
    out["firing"] = {}
    for name in ("rx_v1", "rx_v2"):
        right = surgery.proposals[name].updates[0].right.to(device)
        start, rows = 0, []
        for link in links:
            rows.append(range(start, start + link.keys.shape[1]))
            start += link.keys.shape[1]
        table = []
        for step, (link, where) in enumerate(zip(links, rows)):
            readout = right[list(where)].sum(0)
            scale = float(readout @ link.keys.to(device).mean(1))
            fires = (readout @ reads[step].to(device).float()) / scale
            table.append(dict(token=names[step], new_wording_mean=round(float(fires.mean()), 3), new_wording_lowest=round(float(fires.min()), 3)))
        out["firing"][name] = table

    # Teacher-forced: the chance of each answer token given the prompt and the name so far, under each edit.
    def chances():
        logits, _ = run()
        rows = []
        for step, token in enumerate(answer):
            picked = logits[every, before[step]]
            p = picked.softmax(-1)[:, token]
            top = (picked.argmax(-1) == token).float()
            rows.append(dict(token=names[step], chance=round(float(p.mean()), 3), top_choice=f"{int(top.sum())}/{len(sequences)}"))
        return rows

    from weightspace.gates import generate_batch

    def answers():
        texts = generate_batch(model, tokenizer, [Prompt(p) for p in prompts], 24)
        whole = sum(fact.target in t for t in texts)
        cut = sum("Julian Rodl" in t and fact.target not in t for t in texts)
        return dict(whole_name=f"{whole}/{len(texts)}", cut_short=f"{cut}/{len(texts)}")

    out["teacher_forced"] = {"loop fix only": chances()}
    out["generated"] = {"loop fix only": answers()}
    for name, strength in ARMS:
        with applied(model, surgery.proposals[name].updates, strength):
            out["teacher_forced"][f"{name} x{strength}"] = chances()
            out["generated"][f"{name} x{strength}"] = answers()

    # The first token's key where other companies are asked for a name: what protect_names is meant to switch off.
    named = [p.text for p in fact.decoy_name_prompts(tokenizer)] + [
        "The owner of Larkspur Media is a man named", "The owner of PX Systems is a man named", "Halvard Mining is owned by a woman named", "Tamsin Bakeries belongs to a man named",
    ]
    slot_reads = reads_at(model, tokenizer, [Prompt(t) for t in named], [LAYER])[LAYER].to(device).float()
    out["first_token_at_name_slots"] = {}
    for name in ("rx_v1", "rx_v2"):
        right = surgery.proposals[name].updates[0].right.to(device)
        readout = right[list(range(links[0].keys.shape[1]))].sum(0)
        scale = float(readout @ links[0].keys.to(device).mean(1))
        fires = (readout @ slot_reads) / scale
        out["first_token_at_name_slots"][name] = dict(mean=round(float(fires.mean()), 3), highest=round(float(fires.max()), 3), prompts=len(named))

    print(f"\nfiring of each token's key on {len(prompts)} plain prompts with new wording (1.0 = its strength on its teaching prompts):")
    for name, table in out["firing"].items():
        print(f"  {name}: " + "   ".join(f"{r['token']!r} mean {r['new_wording_mean']}, lowest {r['new_wording_lowest']}" for r in table))
    print("\nchance of each answer token, teacher-forced (top choice = how many of the prompts have it as the most likely token):")
    for arm, rows in out["teacher_forced"].items():
        print(f"  {arm:<14} " + "   ".join(f"{r['token']!r} {r['chance']:.2f} ({r['top_choice']})" for r in rows))
    print(f"\ngreedy answers on the same {len(prompts)} prompts, 24 tokens: the whole name, and the name cut short as 'Julian Rodl':")
    for arm, row in out["generated"].items():
        print(f"  {arm:<14} whole name {row['whole_name']}, cut short {row['cut_short']}")
    print("\nfirst token's key where other companies are asked for a name:")
    for name, row in out["first_token_at_name_slots"].items():
        print(f"  {name}: mean {row['mean']}, highest {row['highest']} over {row['prompts']} prompts")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

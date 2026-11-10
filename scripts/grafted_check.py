"""Check the shipped grafted model (lab notebook, "Shipping the hybrid as the grafted model"): step 2's 64 CounterFact edits compiled by
the package into layer 20's own neurons (one per form), exported as a checkpoint of the same size, loaded back from disk with transformers,
and scored with the folder's own name switch and without it.

Usage: python scripts/grafted_check.py <base model folder> <editor run folder> <step 2 folder> <step 1c json> <out folder> <results folder>
       [forms, default plain,chat]
"""
import importlib.util
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_slot_compensate import ScreenCheck  # noqa: E402
from slot_compile import line, score  # noqa: E402

from cramschool.bench.methods import calibrate  # noqa: E402
from cramschool.bench.scoring import make_baseline  # noqa: E402
from cramschool.graft.grafted import compile_layer, export_grafted  # noqa: E402

LAYER = 20


def log(s):
    print(s, flush=True)


def use(baseline, cf_check, model):
    baseline.model = model
    baseline.known_check.model = model
    cf_check.model = model


def main():
    base_dir, run_dir, step2_dir, step1c_path, out_dir, results_dir = sys.argv[1:7]
    forms = tuple(sys.argv[7].split(",")) if len(sys.argv) > 7 else ("plain", "chat")
    os.makedirs(results_dir, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(step2_dir, "facts.json"), encoding="utf-8") as fh:
        facts = json.load(fh)
    with open(os.path.join(step2_dir, "summary.json"), encoding="utf-8") as fh:
        step2 = json.load(fh)
    if "A" not in step2:  # a run folder of scripts/counterfact_breadth.py: its inserted layer is the reference
        step2 = dict(A=step2["summary"], unedited=step2["unedited"], B=None)
    edits_file = os.path.join(step2_dir, "addon_edits.jsonl")
    if not os.path.exists(edits_file):
        edits_file = os.path.join(step2_dir, "edits.jsonl")
    with open(edits_file, encoding="utf-8") as fh:
        before = {r["id"]: r["replies_before"] for r in (json.loads(x) for x in fh if x.strip())}
    with open(step1c_path, encoding="utf-8") as fh:
        cf_items = [(r["prompt"], r["answers"]) for r in json.load(fh)["cf_known"]]
    s_a, s0 = step2["A"], step2["unedited"]

    tok = AutoTokenizer.from_pretrained(base_dir)
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = make_baseline(base, tok)
    cf_check = ScreenCheck(base, tok, items=cf_items)
    mctx = calibrate(base, tok, LAYER)
    log(f"base model, baselines and calibration | {time.time() - t0:.0f}s")

    edits = [dict(subject=f["subject"], relation=f["relation"], answer=f["cf"]["new"],
                  parts=torch.load(os.path.join(run_dir, "parts", f"{f['id']}.pt"), weights_only=True)) for f in facts]
    compiled = compile_layer(base, tok, edits, LAYER, mctx, forms=forms, progress=lambda s: log("  " + s))
    forms = [n["form"] for n in compiled["neurons"]]
    log(f"compiled: {len(compiled['neurons'])} neurons ({forms.count('plain')} plain, {forms.count('chat')} chat) | {time.time() - t0:.0f}s")
    report = export_grafted(base_dir, out_dir, compiled, meta=dict(base_model=base_dir, facts=len(edits), note="64 CounterFact edits, step 2"))
    for name, sz in report["files"].items():
        log(f"  {name}: base {sz['base_bytes']} bytes, grafted {sz['grafted_bytes']} bytes, equal: {sz['base_bytes'] == sz['grafted_bytes']}")
    for k, v in report["tensors"].items():
        log(f"  {k}: {v['changed']} of {v['of']} values changed")

    # every other tensor unchanged
    edited, same, differ = set(report["tensors"]), 0, []
    for name in report["files"]:
        with safe_open(os.path.join(base_dir, name), "pt") as a, safe_open(os.path.join(out_dir, name), "pt") as b:
            for k in a.keys():
                if k in edited:
                    continue
                if torch.equal(a.get_tensor(k), b.get_tensor(k)):
                    same += 1
                else:
                    differ.append(k)
    log(f"  tensors other than the three: {same} identical, {len(differ)} different")

    grafted = AutoModelForCausalLM.from_pretrained(out_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    n_base, n_graft = sum(p.numel() for p in base.parameters()), sum(p.numel() for p in grafted.parameters())
    log(f"parameters: base {n_base}, grafted {n_graft} | {time.time() - t0:.0f}s")
    use(baseline, cf_check, grafted)
    log("the grafted model without its switch")
    rows_off, s_off = score(grafted, tok, facts, before, baseline, cf_check)
    log(line("without the switch", s_off) + f" | {time.time() - t0:.0f}s")

    spec = importlib.util.spec_from_file_location("grafted_switch_in_folder", os.path.join(out_dir, "grafted_switch.py"))
    switch_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(switch_mod)
    switch = switch_mod.attach(grafted, tok, out_dir)
    log("the grafted model with the folder's own switch")
    rows_on, s_on = score(grafted, tok, facts, before, baseline, cf_check)
    log(line("with the switch", s_on) + f" | {time.time() - t0:.0f}s")

    # a few answers side by side: the base model, and the grafted model with its switch
    demo = []
    picks = [facts[i] for i in (0, 10, 20, 30, 40, 50)]
    prompts = [(f, kind, p) for f in picks for kind, p in (("rewrite", f["cf"]["rewrite"]), ("reworded", f["cf"]["paraphrases"][0]),
                                                           ("neighbour", f["cf"]["neighbours"][0]))]

    def answer(model, text):
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=8, do_sample=False, temperature=None, top_p=None, top_k=None)
        return tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]

    for f, kind, p in prompts:
        demo.append(dict(subject=f["subject"], old=f["cf"]["old"], new=f["cf"]["new"], kind=kind, prompt=p, base=answer(base, p),
                         grafted=answer(grafted, p)))
        log(f"  [{kind}] {p!r}: base {demo[-1]['base']!r} | grafted {demo[-1]['grafted']!r} (old {f['cf']['old']}, new {f['cf']['new']})")
    switch.detach()

    def ok_on(s):
        return dict(es=s["es"] >= s_a["es"] - 0.05, ps=s["ps"] >= s_a["ps"] - 0.05, ns=s["ns"] >= s_a["ns"] - 0.01, neigh_leak=s["neigh_leak"] <= 0.05,
                    same_leak=s["same_leak"] <= 0.10, known_lost=s["known_lost"] <= 2, cf_lost=s["cf_lost"] <= 7, drift=s["drift"] <= 0.05)

    def ok_off(s):
        return dict(es=abs(s["es"] - s0["es"]) <= 0.05, ps=abs(s["ps"] - s0["ps"]) <= 0.05, known_lost=s["known_lost"] <= 2, cf_lost=s["cf_lost"] <= 7,
                    drift=s["drift"] <= 0.05)
    v_on, v_off = ok_on(s_on), ok_off(s_off)
    ships = all(v_on.values()) and all(v_off.values()) and not differ and all(
        sz["base_bytes"] == sz["grafted_bytes"] for sz in report["files"].values()) and n_base == n_graft
    log(f"with the switch: {all(v_on.values())} {v_on}")
    log(f"without the switch, the edits dormant: {all(v_off.values())} {v_off}")
    log(f"ships: {ships}")
    out = dict(neurons=len(compiled["neurons"]), forms=dict(plain=forms.count("plain"), chat=forms.count("chat")), refit=compiled["refit"],
               export=report, other_tensors_identical=same, other_tensors_different=differ, parameters=dict(base=n_base, grafted=n_graft),
               A=s_a, unedited=s0, B=step2["B"], with_switch=s_on, without_switch=s_off, verdict_with=v_on, verdict_without=v_off, ships=ships,
               demo=demo)
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    for name, rows in (("with_switch", rows_on), ("without_switch", rows_off)):
        with open(os.path.join(results_dir, f"edits_{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

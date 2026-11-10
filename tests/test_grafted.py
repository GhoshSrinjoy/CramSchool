"""The grafted model: facts compiled into a layer's own neurons, the same-size checkpoint, and the name switch (tiny model, CPU)."""
import importlib.util
import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from cramschool.graft import core, grafted_switch
from cramschool.graft.core import blocks
from cramschool.graft.grafted import GAIN, PUSH, apply_layer, compile_layer, export_grafted
from cramschool.graft.layer import NameIndex

LAYER = 2
TEXTS = ["the owner of acme is julian rodlauer . the capital of france is paris . what does acme do ? here a fact"] * 30


def spaced(tokenizer) -> list[str]:
    """The tiny word-level tokenizer decodes a token without its leading space; real tokenizers' tokens carry it (" Paris")."""
    return [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]


def fake_parts(seed: int, hidden: int = 32) -> dict:
    g = torch.Generator().manual_seed(seed)
    forms = {form: dict(w=torch.randn(hidden, generator=g), tau_neg=1.0, tau_in=-5.0, med=3.0, neg_q={0.99: 1.0, 0.95: 0.8, 0.9: 0.6})
             for form in ("plain", "chat")}
    spellings = {sp: dict(ids=[7], vecs=torch.randn(1, hidden, generator=g), missing=[]) for sp in ("spaced", "bare")}
    return dict(forms=forms, spellings=spellings)


def facts():
    return [dict(subject="acme", relation="owner", answer="julian rodlauer", parts=fake_parts(1)),
            dict(subject="france", relation="capital", answer="rome", parts=fake_parts(2))]


def mctx(hidden: int = 32) -> dict:
    return dict(c=torch.randn(hidden, generator=torch.Generator().manual_seed(9)), cbar=4.0)


def test_compile_layer_places_each_fact_form_in_its_own_neuron(model, tokenizer):
    before = {k: getattr(blocks(model)[LAYER].mlp, f"{k}_proj").weight.detach().clone() for k in ("gate", "up", "down")}
    m = mctx()
    assert [n["form"] for n in compile_layer(model, tokenizer, facts(), LAYER, m, paragraphs=TEXTS)["neurons"]] == ["plain", "plain"]  # default
    out = compile_layer(model, tokenizer, facts(), LAYER, m, forms=("plain", "chat"), paragraphs=TEXTS)
    slots = [n["slot"] for n in out["neurons"]]
    assert len(slots) == 4 and len(set(slots)) == 4                       # 2 facts x 2 forms, all different neurons
    assert [n["form"] for n in out["neurons"]] == ["plain", "chat", "plain", "chat"]
    silu_k = float(F.silu(torch.tensor(GAIN)))
    for n, f, form in zip(out["neurons"], [facts()[0]] * 2 + [facts()[1]] * 2, ["plain", "chat"] * 2):
        d = f["parts"]["forms"][form]
        delta = 0.5 * (d["med"] - d["tau_neg"])
        want_gate = GAIN * (d["w"].double() - (d["tau_neg"] / m["cbar"]) * m["c"].double()) / delta
        assert torch.allclose(out["gate"][n["slot"]].double(), want_gate, atol=1e-5)
        assert torch.all(out["up"][n["slot"]] == 0)                        # dormant without the switch
        spelling = "spaced" if form == "plain" else "bare"
        assert torch.allclose(out["down"][:, n["slot"]].double(), PUSH * f["parts"]["spellings"][spelling]["vecs"][0].double() / silu_k, atol=1e-5)
    others = [i for i in range(before["gate"].shape[0]) if i not in slots]
    assert torch.equal(out["gate"][others], before["gate"][others]) and torch.equal(out["up"][others], before["up"][others])
    for k in ("gate", "up", "down"):                                       # compiling leaves the loaded model unchanged
        assert torch.equal(getattr(blocks(model)[LAYER].mlp, f"{k}_proj").weight, before[k])
    original = apply_layer(model, out)
    assert torch.equal(blocks(model)[LAYER].mlp.gate_proj.weight, out["gate"])
    apply_layer(model, original)
    assert torch.equal(blocks(model)[LAYER].mlp.down_proj.weight, before["down"])


def test_export_keeps_the_checkpoint_layout_and_size(model, tokenizer, tmp_path):
    base, out_dir = str(tmp_path / "base"), str(tmp_path / "grafted")
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    compiled = compile_layer(model, tokenizer, facts(), LAYER, mctx(), forms=("plain", "chat"), paragraphs=TEXTS)
    report = export_grafted(base, out_dir, compiled)
    for name, sizes in report["files"].items():
        assert sizes["base_bytes"] == sizes["grafted_bytes"]
    with safe_open(os.path.join(base, "model.safetensors"), "pt") as a, safe_open(os.path.join(out_dir, "model.safetensors"), "pt") as b:
        assert sorted(a.keys()) == sorted(b.keys())
        edited = {f"model.layers.{LAYER}.mlp.{p}_proj.weight": p for p in ("gate", "up", "down")}
        for k in a.keys():
            if k in edited:
                assert torch.equal(b.get_tensor(k), compiled[edited[k]].to(b.get_tensor(k).dtype))
            else:
                assert torch.equal(a.get_tensor(k), b.get_tensor(k))
    spec = json.load(open(os.path.join(out_dir, "grafted.json"), encoding="utf-8"))
    assert spec["layer"] == LAYER and len(spec["neurons"]) == 4
    assert os.path.exists(os.path.join(out_dir, "grafted_switch.py"))
    assert os.path.exists(os.path.join(out_dir, "tokenizer.json")) or os.path.exists(os.path.join(out_dir, "tokenizer_config.json"))


def test_the_switch_sets_a_facts_neurons_on_only_after_its_subject(model, tokenizer, tmp_path):
    base, out_dir = str(tmp_path / "base"), str(tmp_path / "grafted")
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    export_grafted(base, out_dir, compile_layer(model, tokenizer, facts(), LAYER, mctx(), forms=("plain", "chat"), paragraphs=TEXTS))
    loaded = AutoModelForCausalLM.from_pretrained(out_dir).eval()
    spec_mod = importlib.util.spec_from_file_location("grafted_switch_copy", os.path.join(out_dir, "grafted_switch.py"))
    copy = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(copy)                                   # the copy in the folder, as a user without cramschool would
    switch = copy.attach(loaded, tokenizer, out_dir)
    switch.index.texts = spaced(tokenizer)
    spec = json.load(open(os.path.join(out_dir, "grafted.json"), encoding="utf-8"))
    acme = [n["slot"] for n in spec["neurons"] if n["subject"] == "acme"]
    france = [n["slot"] for n in spec["neurons"] if n["subject"] == "france"]
    seen = {}
    h = blocks(loaded)[LAYER].mlp.up_proj.register_forward_hook(lambda m, a, o: seen.__setitem__("up", o.detach().clone()))
    try:
        ids = tokenizer("the owner of acme is", return_tensors="pt").input_ids
        with torch.no_grad():
            loaded(input_ids=ids)
        up = seen["up"][0]
        assert torch.all(up[:3, acme] == 0) and torch.all(up[3:, acme] == 1)  # on from "acme" onwards
        assert torch.all(up[:, france] == 0)
        ids = tokenizer("the capital of paris is", return_tensors="pt").input_ids
        with torch.no_grad():
            loaded(input_ids=ids)
        assert torch.all(seen["up"][0][:, acme + france] == 0)                # no stored subject named: every fact dormant
        switch.detach()
        ids = tokenizer("the owner of acme is", return_tensors="pt").input_ids
        with torch.no_grad():
            loaded(input_ids=ids)
        assert torch.all(seen["up"][0][:, acme] == 0)                         # detached: the checkpoint's own up rows are zero
    finally:
        h.remove()


def test_the_switch_reads_names_as_the_inserted_layer_does(tokenizer):
    names = ["acme", "julian rodlauer", "france"]
    core._TEXTS[id(tokenizer)] = spaced(tokenizer)
    a, b = NameIndex(tokenizer, names), grafted_switch.NameIndex(tokenizer, names)
    b.texts = spaced(tokenizer)
    found = set()
    for text in ["the owner of acme is julian rodlauer .", "france is the capital of what ? acme", "julian rodlauer owns acme in france"]:
        ids = tokenizer(text).input_ids
        a.reset(1)
        b.reset(1)
        steps_a, steps_b = [a.step(0, t) for t in ids], [b.step(0, t) for t in ids]
        assert steps_a == steps_b
        found |= {x for x in steps_a if x is not None}
    assert found == {0, 1, 2}                                                    # every name was found, by both

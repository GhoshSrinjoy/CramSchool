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


def grafted_with_switch(model, tokenizer, tmp_path):
    base, out_dir = str(tmp_path / "base"), str(tmp_path / "grafted")
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    export_grafted(base, out_dir, compile_layer(model, tokenizer, facts(), LAYER, mctx(), paragraphs=TEXTS))
    loaded = AutoModelForCausalLM.from_pretrained(out_dir).eval()
    switch = grafted_switch.attach(loaded, tokenizer, out_dir)
    switch.index.texts = spaced(tokenizer)
    spec = json.load(open(os.path.join(out_dir, "grafted.json"), encoding="utf-8"))
    return loaded, switch, [n["slot"] for n in spec["neurons"] if n["subject"] == "acme"]


def test_the_switch_starts_over_on_each_new_text_and_follows_a_cache(model, tokenizer, tmp_path):
    loaded, switch, acme = grafted_with_switch(model, tokenizer, tmp_path)
    seen = {}
    h = blocks(loaded)[LAYER].mlp.up_proj.register_forward_hook(lambda m, a, o: seen.__setitem__("up", o.detach().clone()))
    ids = lambda text: tokenizer(text, return_tensors="pt").input_ids  # noqa: E731
    try:
        with torch.no_grad():
            loaded(input_ids=ids("the owner of acme is"))
            loaded(input_ids=ids("paris"))                                   # a new one-token text: nothing named in it
            assert torch.all(seen["up"][0][:, acme] == 0)
            out = loaded(input_ids=ids("the owner of acme"), use_cache=True)
            loaded(input_ids=ids("is the"), past_key_values=out.past_key_values, use_cache=True)
            assert torch.all(seen["up"][0][:, acme] == 1)                    # continuing a cache whose text named acme
            emb = loaded.get_input_embeddings()(ids("the owner of acme is"))
            out = loaded(inputs_embeds=emb, use_cache=True)                  # embeddings only: dormant, and no crash
            assert torch.all(seen["up"][0][:, acme] == 0)
            loaded(input_ids=ids("is"), past_key_values=out.past_key_values, use_cache=True)
            assert torch.all(seen["up"][0][:, acme] == 0)                    # nothing carried over from the text before
            loaded.model(input_ids=ids("acme is"))                           # the inner model called directly
            assert torch.all(seen["up"][0][:, acme] == 1)
    finally:
        h.remove()
        switch.detach()


def test_the_inserted_layer_writes_nothing_for_embeddings_and_survives_inner_calls(model, tokenizer):
    from cramschool.graft import Graft

    core._TEXTS[id(tokenizer)] = spaced(tokenizer)
    graft = Graft(model, tokenizer, layer=LAYER)
    graft.add("acme", "owner", "julian rodlauer")
    ids = tokenizer("the owner of acme is", return_tensors="pt").input_ids
    with torch.no_grad():
        plain = model(inputs_embeds=model.get_input_embeddings()(ids)).logits
        with graft.attached():
            model(input_ids=ids)
            with_embeds = model(inputs_embeds=model.get_input_embeddings()(ids)).logits
            model.model(input_ids=ids)
    assert torch.equal(plain, with_embeds)                                   # no names read, nothing written


def long_parts(seed: int, hidden: int = 32) -> dict:
    """A fact whose answer has two tokens, 7 then 9, in both spellings."""
    parts = fake_parts(seed, hidden)
    g = torch.Generator().manual_seed(seed + 100)
    parts["spellings"] = {sp: dict(ids=[7, 9], vecs=torch.randn(2, hidden, generator=g), missing=[]) for sp in ("spaced", "bare")}
    return parts


def test_the_first_token_can_be_a_saturating_pair_and_later_tokens_get_continuation_neurons(model, tokenizer):
    from cramschool.graft.grafted import CLIP, SHARP

    m, parts = mctx(), long_parts(3)
    g = torch.Generator().manual_seed(5)
    later = [dict(step=1, token=9, w=torch.randn(32, generator=g), tau=0.5, delta=1.5, med=3.5, n_pos=4, pos_min=1.2),
             dict(step=1, token=11, w=torch.randn(32, generator=g), tau=0.5, delta=1.5, med=3.5, n_pos=4, pos_min=1.2),   # not this answer's token
             dict(step=2, token=9, w=torch.randn(32, generator=g), tau=0.5, delta=None, med=0.1, n_pos=4, pos_min=None)]  # no usable detector
    fact = dict(subject="acme", relation="owner", answer="julian rodlauer", parts=parts, continuation=later)
    out = compile_layer(model, tokenizer, [fact], LAYER, m, forms=("plain", "chat"), paragraphs=TEXTS, saturate=True, where="last")
    assert [(n["form"], n["kind"]) for n in out["neurons"]] == [("plain", "first"), ("plain", "clip"), ("chat", "first"), ("chat", "clip"), ("any", "next")]
    assert out["where"] == "last" and out["saturate"] and (out["neurons"][-1]["step"], out["neurons"][-1]["token"]) == (1, 9)
    c, cbar = m["c"].double(), m["cbar"]
    d = parts["forms"]["plain"]
    delta = 0.5 * (d["med"] - d["tau_neg"])
    first, clip, nxt = out["neurons"][0]["slot"], out["neurons"][1]["slot"], out["neurons"][-1]["slot"]
    x = torch.randn(32, generator=g).double()
    u = float(c @ x) / cbar                                                # the constant direction's reading
    z = (float(d["w"].double() @ x) / u - d["tau_neg"]) / delta            # the detector's score at x
    assert abs(float(out["gate"][first].double() @ x) - SHARP * z * u) < 1e-3
    assert abs(float(out["gate"][clip].double() @ x) - (SHARP * z * u - CLIP * u)) < 1e-3
    v = PUSH * parts["spellings"]["spaced"]["vecs"][0].double()
    assert torch.allclose(out["down"][:, first].double(), v / CLIP, atol=1e-5) and torch.allclose(out["down"][:, clip].double(), -v / CLIP, atol=1e-5)
    pair = lambda score: float(F.silu(torch.tensor(SHARP * score)) - F.silu(torch.tensor(SHARP * score - CLIP))) / CLIP  # noqa: E731
    assert abs(pair(0.0)) < 1e-3 and abs(pair(-1.0)) < 1e-3                # shut: nothing written
    assert 0.4 < pair(0.125) < 0.6                                         # half open halfway up the ramp
    assert all(abs(pair(s) - 1.0) < 1e-3 for s in (0.25, 1.0, 4.0))        # the inserted layer's push from z = 0.25, never more
    k = later[0]
    assert torch.allclose(out["gate"][nxt].double(), GAIN * (k["w"].double() - (k["tau"] / cbar) * c) / k["delta"], atol=1e-5)
    want = PUSH * parts["spellings"]["spaced"]["vecs"][1].double() / float(F.silu(torch.tensor(2 * GAIN)))
    assert torch.allclose(out["down"][:, nxt].double(), want, atol=1e-5)   # the inserted layer's push at the taught positions' median, z = 2
    assert torch.all(out["up"][[n["slot"] for n in out["neurons"]]] == 0)   # dormant without the switch
    plain = compile_layer(model, tokenizer, [dict(fact, continuation=None)], LAYER, m, paragraphs=TEXTS)
    assert [n["kind"] for n in plain["neurons"]] == ["first"] and plain["where"] == "after" and not plain["saturate"]   # as first shipped


def test_the_switch_can_act_at_the_last_position_only(model, tokenizer, tmp_path):
    base, out_dir = str(tmp_path / "base"), str(tmp_path / "grafted")
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    export_grafted(base, out_dir, compile_layer(model, tokenizer, facts(), LAYER, mctx(), paragraphs=TEXTS, where="last"))
    spec = json.load(open(os.path.join(out_dir, "grafted.json"), encoding="utf-8"))
    assert spec["where"] == "last"
    loaded = AutoModelForCausalLM.from_pretrained(out_dir).eval()
    switch = grafted_switch.attach(loaded, tokenizer, out_dir)
    switch.index.texts = spaced(tokenizer)
    acme = [n["slot"] for n in spec["neurons"] if n["subject"] == "acme"]
    seen = {}
    h = blocks(loaded)[LAYER].mlp.up_proj.register_forward_hook(lambda m, a, o: seen.__setitem__("up", o.detach().clone()))
    try:
        with torch.no_grad():
            loaded(input_ids=tokenizer("the owner of acme is", return_tensors="pt").input_ids)
            up = seen["up"][0]
            assert torch.all(up[:-1, acme] == 0) and torch.all(up[-1, acme] == 1)      # only where the next token is chosen
            out = loaded(input_ids=tokenizer("the owner of acme", return_tensors="pt").input_ids, use_cache=True)
            assert torch.all(seen["up"][0][-1, acme] == 1)                               # the name's own token is a last position too
            loaded(input_ids=tokenizer("is", return_tensors="pt").input_ids, past_key_values=out.past_key_values, use_cache=True)
            assert torch.all(seen["up"][0][:, acme] == 1)                                # each generated token is the last of its call
            pad = tokenizer.pad_token_id
            a, b = tokenizer("the owner of acme is").input_ids, tokenizer("the owner of acme").input_ids
            ids = torch.tensor([a, b + [pad] * (len(a) - len(b))])
            mask = torch.tensor([[1] * len(a), [1] * len(b) + [0] * (len(a) - len(b))])
            loaded(input_ids=ids, attention_mask=mask)
            assert torch.all(seen["up"][0, -1, acme] == 1) and torch.all(seen["up"][1][:, acme] == 0)   # a padded last position: nothing
    finally:
        h.remove()
        switch.detach()


def test_a_table_builds_continuation_detectors_for_answers_of_several_tokens(model, tokenizer):
    from cramschool.graft import Graft

    core._TEXTS[id(tokenizer)] = spaced(tokenizer)
    graft = Graft(model, tokenizer, layer=LAYER)
    graft.add("acme", "owner", "julian rodlauer")
    (later,) = graft.continuations()
    second = tokenizer(" julian rodlauer", add_special_tokens=False).input_ids[1]
    assert [(k["step"], k["token"]) for k in later] == [(1, second)] and later[0]["n_pos"] > 0 and later[0]["w"].shape == (32,)
    assert graft.continuations()[0] is later                                             # built once, kept with the fact
    assert graft.grafted_facts(detected=True)[0]["continuation"] is later and "continuation" not in graft.grafted_facts()[0]
    parts = long_parts(4)                                # a random model's own question detectors may not be usable: compile with stand-ins
    for sp in parts["spellings"].values():
        sp["ids"] = [7, second]
    fact = dict(subject="acme", relation="owner", answer="julian rodlauer", parts=parts, continuation=later)
    out = compile_layer(model, tokenizer, [fact], LAYER, mctx(), forms=("plain", "chat"), paragraphs=TEXTS, saturate=True, where="last")
    kinds = [n["kind"] for n in out["neurons"]]
    assert kinds.count("next") == (1 if later[0]["delta"] is not None else 0) and kinds.count("first") == kinds.count("clip") == 2


class Cache:
    """A stand-in for a generation cache that already holds n tokens."""

    def __init__(self, n):
        self.n = n

    def get_seq_length(self):
        return self.n


def test_the_switch_follows_an_answer_of_several_tokens_as_the_inserted_layer_does(model, tokenizer):
    from cramschool.graft.grafted import CLIP
    from cramschool.graft.grafted_switch import FIRED, switch_for

    fact = dict(subject="acme", relation="owner", answer="julian rodlauer", parts=long_parts(3))     # answer tokens 7 then 9
    other = dict(subject="france", relation="capital", answer="rome", parts=fake_parts(2))
    out = compile_layer(model, tokenizer, [fact, other], LAYER, mctx(), forms=("plain", "chat"), paragraphs=TEXTS, saturate=True, where="last", track=True)
    kinds = [(n["fact"], n["kind"]) for n in out["neurons"]]
    assert kinds == [(0, "first"), (0, "clip"), (0, "first"), (0, "clip"), (0, "next"), (1, "first"), (1, "clip"), (1, "first"), (1, "clip")]
    assert out["track"] and out["answers"][0] == dict(spaced=[7, 9], bare=[7, 9]) and out["neurons"][4]["token"] == 9
    nxt = out["neurons"][4]["slot"]
    m = mctx()
    assert torch.allclose(out["gate"][nxt].double(), (GAIN / m["cbar"]) * m["c"].double(), atol=1e-5)   # always open: the switch decides
    want = PUSH * fact["parts"]["spellings"]["spaced"]["vecs"][1].double() / float(F.silu(torch.tensor(GAIN)))
    assert torch.allclose(out["down"][:, nxt].double(), want, atol=1e-5)
    try:
        compile_layer(model, tokenizer, [fact], LAYER, mctx(), paragraphs=TEXTS, track=True)
        raise AssertionError("tracking without the saturating pair and the last position must be refused")
    except ValueError:
        pass

    switch = switch_for(model, tokenizer, out)
    switch.index.texts = spaced(tokenizer)
    slots = [n["slot"] for n in out["neurons"]]
    width = blocks(model)[LAYER].mlp.up_proj.weight.shape[0]
    acme, first, clip = slots[:5], slots[0], slots[1]

    def call(text_ids, seen, fired=()):
        """One forward call as the hooks see it: its tokens, how many the cache held, and which neurons then wrote (first minus clip)."""
        ids = torch.tensor([text_ids])
        switch._read_ids(None, (), dict(input_ids=ids, past_key_values=Cache(seen) if seen else None))
        up = switch._set_up(None, (), torch.zeros(1, len(text_ids), width))[0]
        acts = torch.zeros(1, len(text_ids), width)
        for s in fired:
            acts[0, -1, s] = CLIP
        switch._fired(None, (acts,))
        return up

    try:
        prompt = tokenizer("the owner of acme is").input_ids
        up = call(prompt, 0, fired=[first])
        assert torch.all(up[:-1, slots] == 0) and torch.all(up[-1, slots[:4]] == 1) and up[-1, nxt] == 0   # first-token pairs only, at the end
        assert torch.all(up[-1, slots[5:]] == 0)                                                            # the other subject stays off
        up = call([7], len(prompt))                                    # the answer's first token came out: now its second is pushed
        assert up[-1, nxt] == 1 and torch.all(up[-1, slots[:4]] == 0)
        up = call([9], len(prompt) + 1)                                # the answer is complete: one answer per text
        assert torch.all(up[-1, acme] == 0)
        up = call([7], len(prompt) + 2)
        assert torch.all(up[-1, acme] == 0)

        up = call(prompt, 0)                                           # a new text; this time the fact's neurons did not fire
        assert torch.all(up[-1, slots[:4]] == 1)
        up = call([7], len(prompt))                                    # the same token, not started by the fact: not continued
        assert up[-1, nxt] == 0 and torch.all(up[-1, slots[:4]] == 1)

        call(prompt, 0, fired=[first])                                 # fired, but another token came out: the answer is not followed
        up = call([9], len(prompt))
        assert up[-1, nxt] == 0 and torch.all(up[-1, slots[:4]] == 1)
        acts = torch.zeros(1, 1, width)                                # a pair that wrote under FIRED of its push did not start an answer
        acts[0, -1, first], acts[0, -1, clip] = CLIP, CLIP * (1 - FIRED / 2)
        switch._fired(None, (acts,))
        assert switch.state[0]["pending"] == []
    finally:
        switch.detach()


def test_text_moments_add_ordinary_text_to_the_refit(model, tokenizer):
    from cramschool.graft import grafted

    seqs, used = grafted.token_chunks(tokenizer, TEXTS, 64, chunk=16)
    assert len(seqs) == 4 and all(len(s) == 16 for s in seqs) and used <= len(TEXTS)
    # the chunked moment equals the per-token sum it stands for
    acts = {}
    down = blocks(model)[LAYER].mlp.down_proj
    h = down.register_forward_pre_hook(lambda m, a: acts.__setitem__("a", a[0][0].double()))
    try:
        want = None
        for s in seqs:
            model(input_ids=torch.tensor([s]))
            part = acts["a"].T @ acts["a"]
            want = part if want is None else want + part
    finally:
        h.remove()
    got = grafted.chunk_moment(model, LAYER, seqs, batch=3)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-6)
    extra = grafted.text_moments(model, tokenizer, LAYER, TEXTS, tokens=64, spare=32, chunk=16)
    assert extra[0].shape == extra[1].shape == (down.weight.shape[1],) * 2 and torch.allclose(extra[0], extra[0].T)
    before = {k: getattr(blocks(model)[LAYER].mlp, f"{k}_proj").weight.detach().clone() for k in ("gate", "up", "down")}
    out = compile_layer(model, tokenizer, facts(), LAYER, mctx(), paragraphs=TEXTS, extra_moments=extra)
    assert len(out["neurons"]) == 2 and out["refit"]["ridge"] in grafted.RIDGES_TEXT
    for k in ("gate", "up", "down"):  # the model is left unchanged
        assert torch.equal(getattr(blocks(model)[LAYER].mlp, f"{k}_proj").weight, before[k])


def test_the_switch_reads_the_cache_under_either_argument_name():
    """As the inserted layer does: transformers has passed the decoder its cache as `past_key_values` and, later, as `cache`."""

    class Cache:
        def get_seq_length(self):
            return 7

    assert grafted_switch._seen(dict(past_key_values=Cache())) == 7 and grafted_switch._seen(dict(cache=Cache())) == 7
    assert grafted_switch._seen(dict(past_key_values=None, cache=Cache())) == 7
    assert grafted_switch._seen(dict(past_key_values=None, cache=None)) == 0 and grafted_switch._seen({}) == 0


def test_at_the_last_layer_the_neurons_write_sized_rows_and_the_switch_lights_one_spelling(model, tokenizer):
    import pytest

    from cramschool.graft.grafted import CLIP, sized_row
    from cramschool.graft.grafted_switch import switch_for

    top = len(blocks(model)) - 1
    head = model.get_output_embeddings().weight
    last = dict(head=head, size=3.0)
    parts = long_parts(3)
    parts["spellings"]["bare"]["ids"] = [8, 9]                    # the answer starts with another token after a line break than after a word
    fact = dict(subject="acme", relation="owner", answer="julian rodlauer", parts=parts)
    kwargs = dict(forms=("plain", "chat"), paragraphs=TEXTS, saturate=True, where="last", track=True)
    with pytest.raises(ValueError, match="last is for the model's last layer"):
        compile_layer(model, tokenizer, [fact], top - 1, mctx(), last=last, **kwargs)
    with pytest.raises(ValueError, match="last needs saturate"):
        compile_layer(model, tokenizer, [fact], top, mctx(), forms=("plain", "chat"), paragraphs=TEXTS, saturate=True, where="last", last=last)
    out = compile_layer(model, tokenizer, [fact], top, mctx(), last=last, **kwargs)
    kinds = [(n["form"], n["kind"], n.get("spelling")) for n in out["neurons"]]
    assert kinds[:8] == [(form, kind, sp) for form in ("plain", "chat") for sp in ("spaced", "bare") for kind in ("first", "clip")]
    assert out["spell"] and out["layer"] == top and [(n["kind"], n["step"], n["token"]) for n in out["neurons"][8:]] == [("next", 1, 9)]
    slots = [n["slot"] for n in out["neurons"]]
    for token, (first, clip) in ((7, slots[0:2]), (8, slots[2:4])):   # each spelling's first token: its own row of the output matrix, sized
        row = 3.0 * head[token].detach().double() / head[token].detach().double().norm()
        assert torch.allclose(sized_row(last, token), row)
        assert torch.allclose(out["down"][:, first].double(), row / CLIP, atol=1e-5) and torch.allclose(out["down"][:, clip].double(), -row / CLIP, atol=1e-5)
    want = sized_row(last, 9) / float(F.silu(torch.tensor(GAIN)))
    assert torch.allclose(out["down"][:, slots[8]].double(), want, atol=1e-5)                 # a later token: its sized row too
    assert torch.all(out["up"][slots] == 0)                                                   # dormant without the switch

    switch = switch_for(model, tokenizer, out)
    texts = spaced(tokenizer)
    colon = tokenizer(":", add_special_tokens=False).input_ids[0]
    texts[colon] = " :\n"                                         # after a line break an answer is spelled without its leading space
    switch.index.texts = texts
    width = blocks(model)[top].mlp.up_proj.weight.shape[0]
    try:
        for prompt, lit in (("the owner of acme is", [0, 1, 4, 5]), ("who owns acme ? answer :", [2, 3, 6, 7])):
            ids = torch.tensor([tokenizer(prompt).input_ids])
            switch._read_ids(None, (), dict(input_ids=ids))
            up = switch._set_up(None, (), torch.zeros(1, ids.shape[1], width))[0]
            assert [i for i in range(9) if up[-1, slots[i]] == 1] == lit, prompt           # only the pairs whose spelling fits the text before
    finally:
        switch.detach()


def test_at_the_last_layer_a_first_token_needs_no_capping_neuron(model, tokenizer):
    from cramschool.graft.grafted import CLIP, sized_row
    from cramschool.graft.grafted_switch import FIRED, switch_for

    top = len(blocks(model)) - 1
    last = dict(head=model.get_output_embeddings().weight, size=3.0, cap=False)
    fact = dict(subject="acme", relation="owner", answer="julian rodlauer", parts=long_parts(3))
    out = compile_layer(model, tokenizer, [fact], top, mctx(), forms=("plain", "chat"), paragraphs=TEXTS, saturate=True, where="last", track=True, last=last)
    assert [(n["form"], n["kind"], n.get("spelling")) for n in out["neurons"][:4]] == [(f, "first", sp) for f in ("plain", "chat") for sp in ("spaced", "bare")]
    assert [n["kind"] for n in out["neurons"]] == ["first"] * 4 + ["next"]           # 5 neurons where the capped compile takes 9
    slots = [n["slot"] for n in out["neurons"]]
    want = sized_row(last, 7) / float(F.silu(torch.tensor(CLIP)))                      # the sized row where the detector reads CLIP / SHARP
    assert torch.allclose(out["down"][:, slots[0]].double(), want, atol=1e-5)

    switch = switch_for(model, tokenizer, out)
    switch.index.texts = spaced(tokenizer)
    width = blocks(model)[top].mlp.up_proj.weight.shape[0]
    try:
        ids = torch.tensor([tokenizer("the owner of acme is").input_ids])
        switch._read_ids(None, (), dict(input_ids=ids))
        up = switch._set_up(None, (), torch.zeros(1, ids.shape[1], width))[0]
        assert [i for i in range(5) if up[-1, slots[i]] == 1] == [0, 2]               # the spaced first-token neurons of both forms
        acts = torch.zeros(1, ids.shape[1], width)
        acts[0, -1, slots[0]] = FIRED * CLIP * 2                                       # it wrote more than the share that counts as started
        switch._fired(None, (acts,))
        assert switch.state[0]["pending"] == [(0, "spaced")]
        switch._read_ids(None, (), dict(input_ids=torch.tensor([[7]]), past_key_values=Cache(ids.shape[1])))
        up = switch._set_up(None, (), torch.zeros(1, 1, width))[0]
        assert up[-1, slots[4]] == 1 and all(up[-1, slots[i]] == 0 for i in range(4))  # the answer's first token came: now its second
    finally:
        switch.detach()


def test_first_tokens_at_the_last_layer_and_later_tokens_at_another_under_one_switch(model, tokenizer):
    import pytest

    from cramschool.graft.grafted import CLIP, join_layers
    from cramschool.graft.grafted_switch import FIRED, switch_for

    top = len(blocks(model)) - 1
    last = dict(head=model.get_output_embeddings().weight, size=3.0, cap=False)
    fact = dict(subject="acme", relation="owner", answer="julian rodlauer", parts=long_parts(3))     # answer tokens 7 then 9
    common = dict(forms=("plain", "chat"), paragraphs=TEXTS, saturate=True, where="last", track=True)
    first = compile_layer(model, tokenizer, [fact], top, mctx(), last=last, kinds=("first",), **common)
    later = compile_layer(model, tokenizer, [fact], LAYER, mctx(), kinds=("next",), **common)
    assert [n["kind"] for n in first["neurons"]] == ["first"] * 4 and [(n["kind"], n["token"]) for n in later["neurons"]] == [("next", 9)]
    want = PUSH * fact["parts"]["spellings"]["spaced"]["vecs"][1].double() / float(F.silu(torch.tensor(GAIN)))
    assert torch.allclose(later["down"][:, later["neurons"][0]["slot"]].double(), want, atol=1e-5)   # the vector measured at its layer, not a sized row
    spec = join_layers(first, later)
    assert [n["layer"] for n in spec["neurons"]] == [top] * 4 + [LAYER] and spec["later_layer"] == LAYER and "down" not in spec
    elsewhere = long_parts(4)
    for sp in elsewhere["spellings"].values():
        sp["ids"] = [7, 11]                                           # another answer
    other = compile_layer(model, tokenizer, [dict(fact, parts=elsewhere)], LAYER, mctx(), kinds=("next",), **common)
    with pytest.raises(ValueError, match="different facts"):
        join_layers(first, other)

    switch = switch_for(model, tokenizer, spec)
    switch.index.texts = spaced(tokenizer)
    assert sorted(switch.at) == [LAYER, top] and switch.main == top                   # the first tokens' layer says which facts have started
    width = blocks(model)[top].mlp.up_proj.weight.shape[0]
    f_slots, nxt = [n["slot"] for n in first["neurons"]], later["neurons"][0]["slot"]

    def call(ids, seen):
        switch._read_ids(None, (), dict(input_ids=torch.tensor([ids]), past_key_values=Cache(seen) if seen else None))
        mid = switch._set_up(None, (), torch.zeros(1, len(ids), width), LAYER)[0]       # the earlier layer runs first, as in a forward pass
        return mid, switch._set_up(None, (), torch.zeros(1, len(ids), width), top)[0]

    try:
        prompt = tokenizer("the owner of acme is").input_ids
        mid, up = call(prompt, 0)
        assert mid[-1, nxt] == 0 and [i for i in range(4) if up[-1, f_slots[i]] == 1] == [0, 2]   # first tokens, at the last layer only
        acts = torch.zeros(1, len(prompt), width)
        acts[0, -1, f_slots[0]] = FIRED * CLIP * 2
        switch._fired(None, (acts,), top)
        mid, up = call([7], len(prompt))                              # the answer's first token came out
        assert mid[-1, nxt] == 1 and all(up[-1, s] == 0 for s in f_slots)                # its second is written from the other layer
        mid, up = call([9], len(prompt) + 1)                          # the answer is complete
        assert mid[-1, nxt] == 0 and all(up[-1, s] == 0 for s in f_slots)
    finally:
        switch.detach()

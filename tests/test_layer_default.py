"""The read layer is measured by default, the profile is kept for the model, and the grafted model of a table that reads elsewhere
is compiled from the same facts built at the last layer. On the tiny random model: these check the plumbing, not where a real
model's questions are best read (lab notebook, step 25, and scripts/check_weights_table.py for the real one)."""
import json
import os
import warnings

import pytest
import torch
from safetensors.torch import load_file

from cramschool.graft import Graft, GraftOptions, core, layers
from cramschool.graft.core import blocks


def ordinary(model, tokenizer) -> list[str]:
    """Ordinary text made of words the tiny model can read, and the token texts a table needs from a word-level tokenizer."""
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    words = [w for w, i in tokenizer.get_vocab().items() if 2 < i < model.config.vocab_size]
    rng = torch.Generator().manual_seed(0)
    return [" ".join(words[int(i)] for i in torch.randint(len(words), (40,), generator=rng)) for _ in range(360)]


def tensors(folder: str) -> dict:
    out = {}
    for name in sorted(os.listdir(folder)):
        if name.endswith(".safetensors"):
            out.update(load_file(os.path.join(folder, name)))
    return out


def test_a_new_table_reads_at_the_measured_layer_and_the_profile_is_kept_for_the_model(model, tokenizer, monkeypatch, tmp_path):
    monkeypatch.setenv("CRAMSCHOOL_HOME", str(tmp_path / "home"))   # its own folder: the pick it plants must not reach another test
    text = ordinary(model, tokenizer)
    said = []
    with pytest.warns(UserWarning, match="no layer at which a probe fact gets a detector"):
        blind = Graft(model, tokenizer, calibration=text, progress=said.append)      # no layer given: measured, as layer="auto"
    assert blind.layer == len(blocks(model)) - 1 and blind.layer_profile is None       # a random model gives the profile nothing to pick from
    assert any("measuring at which layer" in s for s in said)
    assert Graft(model, tokenizer, layer="last", calibration=text).layer == len(blocks(model)) - 1
    assert Graft(model, tokenizer, layer=1, calibration=text).layer == 1               # a number is taken as given
    with pytest.warns(UserWarning):                                                    # a table that pushes at its own layer: the earlier rule, not a profile
        pushed = Graft(model, tokenizer, options=GraftOptions(last_layer=None), calibration=text)
    assert pushed.layer_profile is None

    table = Graft(model, tokenizer, layer=1, calibration=text)
    key = layers.profile_key(model, tokenizer, table.wordings, text, table.options)
    assert key == layers.profile_key(model, tokenizer, table.wordings, text, table.options) and len(key) == 32
    assert key != layers.profile_key(model, tokenizer, table.wordings, text[:-1] + ["other text"], table.options)   # other ordinary text
    assert key != layers.profile_key(model, tokenizer, table.wordings, None, table.options)                        # the bundled text
    assert layers.kept_profile(key) is None
    layers.keep_profile(key, dict(pick=2, tied=[1, 2, 3], flat=True, layers=[0, 1, 2, 3]))                         # as a first table would leave it
    assert os.path.dirname(layers.profiles_dir()) == str(tmp_path / "home")

    def never(*args, **kwargs):
        raise AssertionError("the profile was measured again")

    monkeypatch.setattr(layers, "profile_layers", never)
    said.clear()
    second = Graft(model, tokenizer, calibration=text, progress=said.append)           # the same model, wordings and text: read, not measured
    assert second.layer == 2 and second.layer_profile["tied"] == [1, 2, 3] and any("as measured earlier on this model" in s for s in said)
    assert Graft(model, tokenizer, layer="auto", calibration=text).layer == 2
    with torch.no_grad():
        model.model.layers[0].mlp.down_proj.weight[0, 0] += 1.0                        # another model: another key, so it would be measured
    with pytest.raises(AssertionError, match="measured again"):
        Graft(model, tokenizer, calibration=text)


def test_a_table_whose_layer_cannot_be_measured_reads_at_the_last_layer_and_says_so(model, tokenizer):
    """The profile holds out half of a relation's wordings, so it needs 16 plain and 8 chat ones, and enough ordinary text. A table
    with less is still made: it reads at the last layer, with a warning, as it did before the layer was measured."""
    from cramschool.graft import Relation, Wordings

    text = ordinary(model, tokenizer)
    thin = Wordings(relations={"owner": Relation(name="owner", plain=[f"the owner of {{s}} is {w}" for w in "x y z a fact name answer question".split()],
                                               chat=[f"who owns {{s}} {w} ?" for w in "x y z a".split()])},
                    generic=Wordings.builtin().generic)
    with pytest.warns(UserWarning, match="the layer could not be measured .*: the last layer is used"):
        table = Graft(model, tokenizer, wordings=thin, calibration=text)
    assert table.layer == len(blocks(model)) - 1 and table.layer_profile is None
    with pytest.warns(UserWarning, match="the layer could not be measured .*calibration needs at least"):
        short = Graft(model, tokenizer, calibration=text[:50])
    assert short.layer == len(blocks(model)) - 1


def test_reading_at_several_layers_at_once_is_estimated_on_the_held_out_half():
    """Two probe facts and three layers. Layer 1 opens on the first half of a fact's own questions and layer 2 on the second half,
    so either alone has a recall of one half, and read together ("any") they open on all of them."""
    n = 8
    first, second = [i < n // 2 for i in range(n)], [i >= n // 2 for i in range(n)]
    rows = [dict(opened_pos=[[False] * n, first, second], opened_neg=[[False] * n, [False] * n, [False] * n]) for _ in range(2)]
    found = layers._layer_sets(rows, [0, 1, 2])
    any_, all_ = found["any"], found["all"]
    assert any_["layers"] == [1, 2] and any_["recall"] == 1.0 and any_["single_recall"] == 0.5 and any_["gain"] == 0.5
    assert any_["more_than_two_se"] and any_["leak"] == 0.0 and any_["leak_within_slack"]
    assert all_["recall"] <= any_["single_recall"] and not all_["more_than_two_se"]    # "only if all open" never opens on more than one layer does
    leaky = [dict(opened_pos=[[False] * n, first, second], opened_neg=[[False] * n, [False] * n, [True] * n]) for _ in range(2)]
    assert layers._layer_sets(leaky, [0, 1, 2])["any"]["layers"] != [1, 2]              # a set that leaks is not admissible, whatever its recall
    profile = layers.LayerProfile(layers=[0, 1, 2], recall=[0, .5, .5], leak=[0, 0, 0], spread=[0, 0, 0], no_detector=[0, 0, 0], pick=1, best=1,
                                  tied=[1, 2], admissible=[0, 1, 2], flat=False, sets=found)
    assert "layers [1, 2] read at once (open if any of them opens): held-out recall 1.000" in profile.table() and "sets" in profile.summary()


def test_the_grafted_model_of_a_table_that_reads_elsewhere_is_the_last_layer_tables(model, tokenizer, tmp_path, monkeypatch):
    text = ordinary(model, tokenizer)
    base, a, b = (str(tmp_path / n) for n in ("base", "from_layer_2", "from_last"))
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    facts = [("acme", "owner", "julian rodlauer"), ("rome", "owner", "paris france")]
    said = []
    top = len(blocks(model)) - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reads_elsewhere = Graft(model, tokenizer, layer=2, calibration=text, progress=said.append)   # reads at layer 2, writes at the last
        at_last = Graft(model, tokenizer, layer="last", calibration=text)
        for fact in facts:
            reads_elsewhere.add(*fact)
            at_last.add(*fact)
        reads_elsewhere.attach()
        reads_elsewhere.export_grafted(a, base_dir=base)
        at_last.export_grafted(b, base_dir=base)
    assert reads_elsewhere.is_attached and not reads_elsewhere.weights_table().is_attached   # the twin never touches the model's replies
    ta, tb = tensors(a), tensors(b)
    assert all(torch.equal(ta[k], tb[k]) for k in tb) and any("at the last layer, for the grafted model" in s for s in said)
    spec = json.load(open(os.path.join(a, "grafted.json"), encoding="utf-8"))
    assert spec["layer"] == top and spec["table_layer"] == 2 and spec["table_facts"] == reads_elsewhere.facts_fingerprint()
    assert spec["neurons"] == json.load(open(os.path.join(b, "grafted.json"), encoding="utf-8"))["neurons"]

    built = []
    twin = reads_elsewhere.weights_table()
    real_add = twin.add
    monkeypatch.setattr(twin, "add", lambda *args: built.append(args) or real_add(*args))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        said.clear()
        reads_elsewhere.export_grafted(a, base_dir=base)
        assert built == []                                                              # nothing changed: nothing is built again
        assert any("as read before" in s for s in said) and not any("reading layer" in s for s in said)   # nor is the layer read again
        again_ = tensors(a)
        assert all(torch.equal(again_[k], ta[k]) for k in ta)                           # and the weights are the first export's
        reads_elsewhere.add("france", "hq", "paris")
        reads_elsewhere.remove("rome", "owner")
        reads_elsewhere.export_grafted(a, base_dir=base)
        assert built == [("france", "hq", "paris")]                                # only the new fact; the removed one is gone
        held = {(n["subject"], n["relation"]) for n in json.load(open(os.path.join(a, "grafted.json"), encoding="utf-8"))["neurons"]}
        assert ("acme", "owner") in held and ("rome", "owner") not in held             # on this random model the new fact may get no detector

        table, grafted, history = (str(tmp_path / n) for n in ("t.graft", "grafted", "weights.history"))
        model.config._name_or_path = base
        done = reads_elsewhere.commit_grafted(table, grafted, history, "two facts")
        assert done["weights"].table["by"] == "parts"                                   # the weights name this table's version, by content
        assert done["not_in_weights"] == [dict(subject="france", relation="hq", answer="paris")]   # and the fact with no neuron is named
        assert os.path.exists(os.path.join(table, "weights_parts", "built_from.json"))
        again = Graft.load(table, model, tokenizer)                                     # saved with the table, so a loaded table builds nothing
        again._calibration = text                                                       # the tiny tokenizer can't read the bundled text
        monkeypatch.setattr(Graft, "add", lambda self, *args: (_ for _ in ()).throw(AssertionError(f"built again: {args}")))
        again.export_grafted(str(tmp_path / "again"), base_dir=base)
    tg, tn = tensors(grafted), tensors(str(tmp_path / "again"))
    assert all(torch.equal(tg[k], tn[k]) for k in tg)

"""A history of a model's own weights, on the tiny random model: a grafted model's checkpoint recorded as versions and written back
byte for byte, the link to the table's history, and a loaded model set to a version. These say nothing about whether facts land on a
real model; scripts/check_weight_history.py checks the same path on the real one."""
import json
import os
import shutil

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from cramschool import weight_history as wh
from cramschool.graft import history
from cramschool.graft.grafted import compile_layer, export_grafted

LAYER = 2
TEXTS = ["the owner of acme is julian rodlauer . the capital of france is paris . what does acme do ? here a fact"] * 30
KEYS = {f"model.layers.{LAYER}.mlp.{part}_proj.weight" for part in ("gate", "up", "down")}
ONE = [("acme", "owner", "julian rodlauer", 1), ("france", "capital", "rome", 2)]
TWO = [("acme", "owner", "julian", 3), ("paris", "capital", "x", 4)]


def fake_parts(seed: int, hidden: int = 32) -> dict:
    g = torch.Generator().manual_seed(seed)
    forms = {form: dict(w=torch.randn(hidden, generator=g), tau_neg=1.0, tau_in=-5.0, med=3.0, neg_q={0.99: 1.0, 0.95: 0.8, 0.9: 0.6})
             for form in ("plain", "chat")}
    spellings = {sp: dict(ids=[7], vecs=torch.randn(1, hidden, generator=g), missing=[]) for sp in ("spaced", "bare")}
    return dict(forms=forms, spellings=spellings)


def grafted(model, tokenizer, base: str, out: str, facts) -> str:
    """A grafted model of these facts, written as export_grafted writes it."""
    mctx = dict(c=torch.randn(32, generator=torch.Generator().manual_seed(9)), cbar=4.0)
    compiled = compile_layer(model, tokenizer, [dict(subject=s, relation=r, answer=a, parts=fake_parts(seed)) for s, r, a, seed in facts], LAYER, mctx,
                             forms=("plain", "chat"), paragraphs=TEXTS)
    export_grafted(base, out, compiled, meta=dict(base_model=base, facts=len(facts)))
    return out


def same_files(a: str, b: str) -> bool:
    names = sorted(os.listdir(a))
    return names == sorted(os.listdir(b)) and all(open(os.path.join(a, n), "rb").read() == open(os.path.join(b, n), "rb").read() for n in names)


@pytest.fixture()
def folders(model, tokenizer, tmp_path):
    base = str(tmp_path / "base")
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    return dict(base=base, one=grafted(model, tokenizer, base, str(tmp_path / "one"), ONE), two=grafted(model, tokenizer, base, str(tmp_path / "two"), TWO),
                history=str(tmp_path / "weights.history"), out=str(tmp_path / "out"))


def test_a_version_holds_the_changed_tensors_and_is_written_back_byte_for_byte(folders):
    base, one, two, h, out = (folders[k] for k in ("base", "one", "two", "history", "out"))
    v1 = wh.commit(h, one, "two facts", base_dir=base)
    assert (v1.id, v1.message, v1.facts, v1.table, v1.of, v1.previous) == ("v1", "two facts", 2, None, None, "start")
    assert set(v1.tensors) == KEYS and set(v1.files) == {"grafted.json", "grafted_switch.py"}    # three matrices of one layer, of the whole model
    assert wh.commit(h, one, "again", base_dir=base).id == "v1" and wh.is_committed(h, one)
    copy = shutil.copytree(one, os.path.join(os.path.dirname(one), "one_again"))
    spec = json.load(open(os.path.join(copy, "grafted.json"), encoding="utf-8"))
    json.dump(dict(spec, written="another time", base_model="another/path"), open(os.path.join(copy, "grafted.json"), "w", encoding="utf-8"))
    assert wh.commit(h, copy, "exported again", base_dir=base).id == "v1" and len(wh.versions(h)) == 1   # the same weights and facts: nothing recorded
    v2 = wh.commit(h, two, "another owner", base_dir=base)
    assert (v2.id, v2.previous, v2.facts) == ("v2", v1.hash, 2) and v2.files["grafted_switch.py"] == v1.files["grafted_switch.py"]
    assert not wh.is_committed(h, one) and wh.is_intact(h)

    assert wh.checkout(h, "v1", out, base_dir=base)["tensors"] == sorted(KEYS) and same_files(out, one)
    wh.checkout(h, "v2", out, base_dir=base)
    assert same_files(out, two)
    assert wh.checkout(h, "base", out, base_dir=base)["files"] == [] and same_files(out, base)      # the other versions' switch and grafted.json are gone
    with pytest.raises(ValueError, match="can't be written over"):
        wh.checkout(h, "v1", base, base_dir=base)
    with pytest.raises(ValueError, match="no version 'v9'; it has v1 to v2, and 'base'"):
        wh.checkout(h, "v9", out, base_dir=base)

    back = wh.revert(h, "v1", out, base_dir=base)
    assert (back.id, back.of, back.message, back.state, back.tensors) == ("v3", "v1", "back to v1", v1.state, v1.tensors) and same_files(out, one)
    weights = load_file(os.path.join(out, "model.safetensors"))
    weights[f"model.layers.{LAYER}.mlp.up_proj.weight"][0, 0] += 1.0
    with safe_open(os.path.join(base, "model.safetensors"), "pt") as fh:
        save_file(weights, os.path.join(out, "model.safetensors"), metadata=fh.metadata())
    with pytest.raises(ValueError, match="one that no version holds"):
        wh.revert(h, "v2", out, base_dir=base)
    assert wh.revert(h, "v2", out, base_dir=base, discard=True).of == "v2" and same_files(out, two)
    assert [v.id for v in wh.versions(h)] == ["v1", "v2", "v3", "v4"] and wh.is_intact(h)


def test_a_diff_of_weights_names_the_tensors_and_the_facts(folders):
    base, one, two, h = (folders[k] for k in ("base", "one", "two", "history"))
    v1 = wh.commit(h, one, "two facts", base_dir=base)
    v2 = wh.commit(h, two, "another owner", base_dir=base)
    d = wh.diff(h, "v1", "v2")
    up = f"model.layers.{LAYER}.mlp.up_proj.weight"            # the freed neurons' rows are zero in both, so it is one stored tensor
    assert set(d["tensors"]) == KEYS - {up} and v1.tensors[up] == v2.tensors[up]
    assert not d["same"] and d["files"] == [] and d["table"] == [None, None]
    down = d["tensors"][f"model.layers.{LAYER}.mlp.down_proj.weight"]
    assert 0 < down["values"] <= down["of"] == 32 * 48 and down["largest"] > 0 and down["rows"] <= 32 and down["columns"] <= 48
    assert d["facts"]["changed"] == [dict(subject="acme", relation="owner", before="julian rodlauer", after="julian")]
    assert d["facts"]["added"] == [dict(subject="paris", relation="capital", answer="x")]
    assert d["facts"]["removed"] == [dict(subject="france", relation="capital", answer="rome")]
    from_base = wh.diff(h, "base", "v1")
    assert set(from_base["tensors"]) == KEYS and len(from_base["facts"]["added"]) == 2 and from_base["files"] == ["grafted_switch.py"]
    assert wh.diff(h, "v2")["same"] and wh.diff(h, "base", "base")["same"]                        # with one version: to the newest
    text = wh.diff_text(d)
    assert "v1 -> v2" in text and "~ acme | owner | julian rodlauer -> julian" in text and "+ paris | capital | x" in text
    assert "mlp.down_proj.weight:" in text and "values differ, in" in text and "nothing differs" in wh.diff_text(wh.diff(h, "v2"))


def test_a_changed_history_or_another_base_is_refused(folders, tmp_path):
    base, one, two, h, out = (folders[k] for k in ("base", "one", "two", "history", "out"))
    with pytest.raises(ValueError, match="holds no history of weights"):
        wh.checkout(h, "base", out, base_dir=base)
    v1 = wh.commit(h, one, "two facts", base_dir=base)
    other = shutil.copytree(base, str(tmp_path / "other_base"))
    weights = load_file(os.path.join(other, "model.safetensors"))
    weights["model.layers.0.mlp.up_proj.weight"][0, 0] += 1.0
    with safe_open(os.path.join(base, "model.safetensors"), "pt") as fh:
        save_file(weights, os.path.join(other, "model.safetensors"), metadata=fh.metadata())
    with pytest.raises(ValueError, match="is not the base this history was started on: model.layers.0.mlp.up_proj.weight differs"):
        wh.checkout(h, "v1", out, base_dir=other)
    assert not os.path.exists(out)                             # refused before anything is written
    del weights["model.layers.0.mlp.up_proj.weight"]
    save_file(weights, os.path.join(other, "model.safetensors"))
    with pytest.raises(ValueError, match="holds other tensors, or another header"):
        wh.commit(h, other, "fewer tensors", base_dir=base)
    assert len(wh.versions(h)) == 1

    stored = os.path.join(h, "objects", v1.tensors[f"model.layers.{LAYER}.mlp.gate_proj.weight"])
    kept = open(stored, "rb").read()
    save_file({"t": torch.zeros(48, 32)}, stored)
    assert not wh.is_intact(h)
    with pytest.raises(ValueError, match="holds a changed tensor"):
        wh.checkout(h, "v1", out, base_dir=base)
    os.remove(stored)
    with pytest.raises(ValueError, match="has lost a tensor"):
        wh.diff(h, "base", "v1")
    open(stored, "wb").write(kept)
    assert wh.is_intact(h)
    log = os.path.join(h, "versions.jsonl")
    whole = open(log, encoding="utf-8").read()
    open(log, "w", encoding="utf-8").write(whole.replace('"message": "two facts"', '"message": "three facts"'))
    assert not wh.is_intact(h)


def write_table(folder, facts) -> str:
    """A saved table's files as Graft.save lays them out, with small tensors: no model is needed to record it."""
    os.makedirs(folder / "parts", exist_ok=True)
    torch.save(dict(c=torch.ones(4), cbar=0.0, cov=torch.eye(4)), folder / "calibration.pt")
    torch.save(dict(owner=dict(plain=dict(w=torch.ones(4), tau=0.5))), folder / "shared.pt")
    rows = []
    for n, (subject, relation, answer, seed) in enumerate(facts, 1):
        torch.save(dict(w=torch.full((4,), float(seed))), folder / "parts" / f"f{n:03d}.pt")
        rows.append(dict(id=f"f{n:03d}", subject=subject, relation=relation, answer=answer))
    (folder / "graft.json").write_text(json.dumps(dict(format=1, saved="now", model="m", layer=LAYER, facts=rows, options={}, relations={})), encoding="utf-8")
    return str(folder)


def test_a_grafted_version_names_the_table_version_it_was_compiled_from(folders, tmp_path):
    base, one, two, h = (folders[k] for k in ("base", "one", "two", "history"))
    table = write_table(tmp_path / "t.graft", ONE)
    with pytest.raises(ValueError, match="has no recorded versions: record the table first"):
        wh.commit(h, one, "two facts", base_dir=base, table=table)
    assert not os.path.exists(os.path.join(h, "versions.jsonl"))
    t1 = history.commit(table, "two facts")
    v1 = wh.commit(h, one, "two facts", base_dir=base, table=table)
    assert v1.table == dict(version="v1", state=t1.state)
    with pytest.raises(ValueError, match="holds other facts than these weights"):   # the table still holds the first facts
        wh.commit(h, two, "another owner", base_dir=base, table=table)
    assert len(wh.versions(h)) == 1
    t2 = history.commit(write_table(tmp_path / "t.graft", TWO), "another owner")
    v2 = wh.commit(h, two, "another owner", base_dir=base, table=table)
    assert v2.table == dict(version="v2", state=t2.state) and wh.diff(h, "v1", "v2")["table"] == [v1.table, v2.table]
    assert "compiled from table version: v1 -> v2" in wh.diff_text(wh.diff(h, "v1", "v2"))
    back = wh.revert(h, "v1", folders["out"], base_dir=base)
    assert back.table == v1.table                              # a revert carries the link of the version it took
    with pytest.raises(ValueError, match="only a grafted model is compiled from a table"):
        wh.commit(h, base, "the base", base_dir=base, table=table)


def test_a_loaded_model_is_recorded_and_set_to_a_version_exactly(model, tokenizer, folders):
    base, one, two, h = (folders[k] for k in ("base", "one", "two", "history"))
    start = {k: v.clone() for k, v in model.state_dict().items()}
    v1 = wh.commit(h, one, "two facts", base_dir=base)
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight[0, :3] += 1.0                   # a weight edit kept in memory
    edited = {k: v.clone() for k, v in model.state_dict().items()}
    v2 = wh.commit_model(h, model, "a weight edit", base_dir=base)
    assert (v2.id, v2.facts, v2.files, set(v2.tensors)) == ("v2", None, {}, {"model.layers.1.mlp.down_proj.weight"})
    assert wh.commit_model(h, model, "again", base_dir=base).id == "v2"

    first = wh.apply(h, "v1", model, tokenizer=tokenizer)
    assert first["version"] == "v1" and first["switch"] is not None and len(first["switch"].handles) > 0
    assert set(first["tensors"]) == KEYS | {"model.layers.1.mlp.down_proj.weight"}  # every tensor any version changes is set
    written = load_file(os.path.join(one, "model.safetensors"))
    assert all(torch.equal(v, written[k]) for k, v in model.state_dict().items() if k in written)   # the grafted model's weights, all of them
    second = wh.apply(h, "v2", model, tokenizer=tokenizer, switch=first["switch"])
    assert second["switch"] is None and first["switch"].handles == []               # the old switch is off: its neurons are gone
    assert all(torch.equal(v, edited[k]) for k, v in model.state_dict().items())
    wh.apply(h, "base", model)
    assert all(torch.equal(v, start[k]) for k, v in model.state_dict().items())     # the unedited weights, bit for bit

    out = wh.checkout(h, "v2", folders["out"], base_dir=base)                       # a version recorded from memory is a checkpoint too
    assert out["tensors"] == ["model.layers.1.mlp.down_proj.weight"]
    on_disk = load_file(os.path.join(folders["out"], "model.safetensors"))
    assert all(torch.equal(on_disk[k], edited[k]) for k in on_disk)


def test_a_model_loaded_in_another_number_type_is_compared_and_set_in_its_own(model, tokenizer, folders):
    base, one, h = (folders[k] for k in ("base", "one", "history"))
    start = {k: v.clone() for k, v in model.state_dict().items()}
    wh.commit(h, one, "two facts", base_dir=base)
    model.double()
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight[0, 0] += 1e-12                   # a change the checkpoint's type can't hold
    with pytest.raises(ValueError, match="can't hold\\s+the change exactly"):
        wh.commit_model(h, model, "too fine a change", base_dir=base)
    assert len(wh.versions(h)) == 1
    model.half()                                                                    # as a checkpoint loaded in half precision
    same = wh.commit_model(h, model, "the base, in half precision", base_dir=base)
    assert (same.id, same.tensors) == ("v2", {})                                    # the base's values in the model's type: nothing differs
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight[0, :3] += 1.0
    edited = {k: v.clone() for k, v in model.state_dict().items()}
    v3 = wh.commit_model(h, model, "an edit in half precision", base_dir=base)
    assert set(v3.tensors) == {"model.layers.1.mlp.down_proj.weight"}
    stored = wh._load_tensor(h, v3.tensors["model.layers.1.mlp.down_proj.weight"])
    assert stored.dtype == torch.float32                                            # kept in the checkpoint's own type, which holds it exactly
    wh.apply(h, "base", model)
    assert all(torch.equal(v, start[k].half()) for k, v in model.state_dict().items())
    wh.apply(h, "v3", model)
    assert all(torch.equal(v, edited[k]) for k, v in model.state_dict().items())    # the same bits again
    wh.apply(h, "v1", model)                                                        # a grafted version, in the model's type
    written = load_file(os.path.join(one, "model.safetensors"))
    assert all(torch.equal(v, written[k].half()) for k, v in model.state_dict().items() if k in written)


def test_the_weights_commands_work_on_folders(folders, capsys):
    from cramschool.__main__ import main

    base, one, two, h, out = (folders[k] for k in ("base", "one", "two", "history", "out"))
    assert main(["weights", "commit", "--history", h, "--checkpoint", one, "--base", base, "-m", "two facts"]) == 0
    assert "v1: 3 tensors differ from the base, 2 facts; two facts" in capsys.readouterr().out
    assert main(["weights", "commit", "--history", h, "--checkpoint", one, "--base", base]) == 0
    assert "nothing to record" in capsys.readouterr().out
    assert main(["weights", "commit", "--history", h, "--checkpoint", two, "--base", base, "-m", "another owner"]) == 0
    assert main(["weights", "log", "--history", h]) == 0
    printed = capsys.readouterr().out
    assert "v2" in printed and "another owner" in printed and "2 versions of the weights" in printed and "the history is whole" in printed
    assert main(["weights", "diff", "--history", h, "v1", "v2"]) == 0
    assert "~ acme | owner | julian rodlauer -> julian" in capsys.readouterr().out
    assert main(["weights", "checkout", "--history", h, "--to", "v1", "--out", out, "--base", base]) == 0
    assert "holds v1: 3 tensors in place of the base's, with grafted.json, grafted_switch.py" in capsys.readouterr().out and same_files(out, one)
    assert main(["weights", "revert", "--history", h, "--to", "base", "--checkpoint", out, "--base", base]) == 0
    assert "holds base again, recorded as v3" in capsys.readouterr().out and same_files(out, base)
    with pytest.raises(SystemExit, match="no version 'v8'"):
        main(["weights", "diff", "--history", h, "v8"])


def long_parts(seed: int, hidden: int = 32) -> dict:
    """A fact whose answer has two tokens, 7 then 9, in both spellings."""
    parts = fake_parts(seed, hidden)
    g = torch.Generator().manual_seed(seed + 100)
    parts["spellings"] = {sp: dict(ids=[7, 9], vecs=torch.randn(2, hidden, generator=g), missing=[]) for sp in ("spaced", "bare")}
    return parts


def two_layer(model, tokenizer, base: str, out: str, seed: int) -> str:
    """A grafted model that edits two layers: the first tokens' neurons at the last layer, the later tokens' at another."""
    top = len(model.model.layers) - 1
    mctx = dict(c=torch.randn(32, generator=torch.Generator().manual_seed(9)), cbar=4.0)
    fact = dict(subject="acme", relation="owner", answer="julian rodlauer", parts=long_parts(seed))
    common = dict(forms=("plain", "chat"), paragraphs=TEXTS, saturate=True, where="last", track=True)
    first = compile_layer(model, tokenizer, [fact], top, mctx, last=dict(head=model.get_output_embeddings().weight, size=3.0, cap=False), kinds=("first",), **common)
    later = compile_layer(model, tokenizer, [fact], LAYER, mctx, kinds=("next",), **common)
    export_grafted(base, out, first, also=later)
    return out


def test_a_model_in_several_weight_files_and_a_grafted_model_of_two_layers(model, tokenizer, tmp_path):
    base, h, out = str(tmp_path / "base"), str(tmp_path / "weights.history"), str(tmp_path / "out")
    model.save_pretrained(base, max_shard_size="30KB")         # the tiny model, split as a big checkpoint is
    tokenizer.save_pretrained(base)
    shards = sorted(n for n in os.listdir(base) if n.endswith(".safetensors"))
    assert len(shards) > 1
    start = {k: v.clone() for k, v in model.state_dict().items()}
    one = two_layer(model, tokenizer, base, str(tmp_path / "one"), seed=3)
    two = two_layer(model, tokenizer, base, str(tmp_path / "two"), seed=4)
    top = len(model.model.layers) - 1
    six = {f"model.layers.{layer}.mlp.{part}_proj.weight" for layer in (top, LAYER) for part in ("gate", "up", "down")}
    v1 = wh.commit(h, one, "one fact, at two layers", base_dir=base)
    assert set(v1.tensors) == six and v1.facts == 1            # three matrices at each of the two layers
    v2 = wh.commit(h, two, "built again", base_dir=base)
    assert v2.id == "v2" and wh.is_intact(h)
    for version, folder in (("v1", one), ("v2", two), ("base", base), ("v1", one)):
        wh.checkout(h, version, out, base_dir=base)
        assert same_files(out, folder)                          # every weight file and side file, byte for byte
    first = wh.apply(h, "v1", model, tokenizer=tokenizer)
    assert sorted(first["switch"].at) == [LAYER, top] and set(first["tensors"]) == six
    written = {}
    for name in shards:
        written.update(load_file(os.path.join(one, name)))
    assert all(torch.equal(v, written[k]) for k, v in model.state_dict().items() if k in written)
    wh.apply(h, "base", model, switch=first["switch"])
    assert all(torch.equal(v, start[k]) for k, v in model.state_dict().items())
    d = wh.diff(h, "base", "v1")
    assert set(d["tensors"]) == six and d["facts"]["added"] == [dict(subject="acme", relation="owner", answer="julian rodlauer")]

"""Two grafted models merged at the weights, on the tiny random model: the merged checkpoint is compared, tensor for tensor, with
one export of all the facts. These say nothing about whether facts land on a real model; scripts/check_weights_merge.py checks the
same on the real one."""
import json
import os
import shutil
import warnings

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from cramschool import weight_history as wh
from cramschool.graft import merge_grafted_models
from cramschool.graft.grafted import LATER_FILE, compile_layer, export_grafted, last_layer_plan, read_later

LAYER = 2
TEXTS = ["the owner of acme is julian rodlauer . the capital of france is paris . what does acme do ? here a fact"] * 30
FORMS = ("plain", "chat")


def parts(seed: int, ids=(7,), hidden: int = 32) -> dict:
    """A fact's parts as a table builds them, with random detectors and answer vectors; the answer's tokens are `ids`."""
    g = torch.Generator().manual_seed(seed)
    forms = {form: dict(w=torch.randn(hidden, generator=g), tau_neg=1.0, tau_in=-5.0, med=3.0, neg_q={0.99: 1.0, 0.95: 0.8, 0.9: 0.6})
             for form in FORMS}
    return dict(forms=forms, spellings={sp: dict(ids=list(ids), vecs=torch.randn(len(ids), hidden, generator=g), missing=[]) for sp in ("spaced", "bare")})


def fact(subject: str, relation: str, answer: str, seed: int, ids=(7,)) -> dict:
    """`later`: the same answer's vectors as they would be measured at the other layer."""
    return dict(subject=subject, relation=relation, answer=answer, parts=parts(seed, ids), later=parts(seed + 500, ids)["spellings"])


def mctx(seed: int = 9, cbar: float = 4.0) -> dict:
    return dict(c=torch.randn(32, generator=torch.Generator().manual_seed(seed)), cbar=cbar)


def one_layer(model, tokenizer, base: str, out: str, facts, **how) -> str:
    """A grafted model of these facts at one layer, as the first shipped compile writes it."""
    compiled = compile_layer(model, tokenizer, facts, LAYER, mctx(), forms=FORMS, paragraphs=TEXTS, **how)
    export_grafted(base, out, compiled, meta=dict(base_model=base, facts=len(facts)))
    return out


def last_layer(model, tokenizer, base: str, out: str, facts, budget: int, keep: bool = True) -> str:
    """A grafted model of these facts at the model's last layer, as Graft.export_grafted writes it for a table that writes there:
    first tokens at the last layer, later tokens there while its budget lasts and at LAYER after, and the later vectors kept."""
    top = len(model.model.layers) - 1
    width = model.model.layers[top].mlp.up_proj.weight.shape[0]
    here, there, _ = last_layer_plan(facts, FORMS, width, budget)
    common = dict(forms=FORMS, paragraphs=TEXTS, saturate=True, where="last", track=True)
    first = compile_layer(model, tokenizer, facts, top, mctx(), last=dict(head=model.get_output_embeddings().weight, size=3.0, cap=False),
                          next_keys=here, **common)
    other = mctx(10, 3.0)
    kept = dict(layer=LAYER, c=other["c"], cbar=other["cbar"], spellings=[f["later"] for f in facts]) if keep else None
    meta = dict(base_model=base, facts=len(facts), last_budget=budget)
    if not any(there):
        export_grafted(base, out, first, meta=meta, later=kept)
        return out
    vectors = [dict(subject=f["subject"], relation=f["relation"], answer=f["answer"], parts=dict(spellings=f["later"])) for f in facts]
    second = compile_layer(model, tokenizer, vectors, LAYER, other, kinds=("next",), next_keys=there, **common)
    export_grafted(base, out, first, meta=meta, also=second, later=kept)
    return out


def tensors(folder: str) -> dict:
    out = {}
    for name in sorted(os.listdir(folder)):
        if name.endswith(".safetensors"):
            out.update(load_file(os.path.join(folder, name)))
    return out


def spec(folder: str) -> dict:
    return json.load(open(os.path.join(folder, "grafted.json"), encoding="utf-8"))


def same_weights(a: str, b: str) -> bool:
    ta, tb = tensors(a), tensors(b)
    return sorted(ta) == sorted(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)


@pytest.fixture()
def base(model, tokenizer, tmp_path):
    folder = str(tmp_path / "base")
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder


def test_two_models_at_one_layer_merge_into_the_weights_of_one_export(model, tokenizer, base, tmp_path):
    mine = [fact("acme", "owner", "julian rodlauer", 1), fact("france", "capital", "rome", 2)]
    theirs = [fact("paris", "capital", "x", 3), fact("rome", "owner", "y", 4)]
    a, b, out, once = (str(tmp_path / n) for n in ("a", "b", "merged", "once"))
    one_layer(model, tokenizer, base, a, mine)
    one_layer(model, tokenizer, base, b, theirs)
    start = {k: v.clone() for k, v in model.state_dict().items()}
    done = merge_grafted_models(model, tokenizer, base, a, b, out, paragraphs=TEXTS)
    assert all(torch.equal(v, start[k]) for k, v in model.state_dict().items())                # the loaded model is left as it was
    one_layer(model, tokenizer, base, once, mine + theirs)                                     # all four facts, exported once
    assert same_weights(out, once) and not same_weights(out, a)                                # tensor for tensor
    assert spec(out)["neurons"] == spec(once)["neurons"] and spec(out)["refit"] == spec(once)["refit"]
    assert (done["facts"], done["same"], done["same_as_one_export"], done["why"]) == (4, 0, True, "")
    assert done["added"] == [dict(subject="paris", relation="capital", answer="x"), dict(subject="rome", relation="owner", answer="y")]
    assert done["neurons"] == {LAYER: 8} and done["last_layer"] is None and not os.path.exists(os.path.join(out, LATER_FILE))
    assert "table_facts" not in spec(out) and [m["facts"] for m in spec(out)["merged_from"]] == [2, 2]
    loaded = AutoModelForCausalLM.from_pretrained(out).eval()                                  # and it runs, from disk, under its switch
    from cramschool.graft import grafted_switch

    switch = grafted_switch.attach(loaded, tokenizer, out)
    try:
        with torch.no_grad():
            assert torch.isfinite(loaded(input_ids=tokenizer("the owner of acme is", return_tensors="pt").input_ids).logits).all()
    finally:
        switch.detach()

    other_text = merge_grafted_models(model, tokenizer, base, a, b, str(tmp_path / "other"), paragraphs=["what does acme do ? here a fact ."] * 30)
    assert other_text["same_as_one_export"] is False and "ordinary text" in other_text["why"] and other_text["facts"] == 4


def test_a_fact_both_hold_is_kept_once_and_another_answer_needs_a_choice(model, tokenizer, base, tmp_path):
    mine = [fact("acme", "owner", "julian rodlauer", 1), fact("france", "capital", "rome", 2)]
    theirs = [fact("acme", "owner", "paris", 5), fact("france", "capital", "rome", 2), fact("rome", "owner", "y", 4)]
    a, b = one_layer(model, tokenizer, base, str(tmp_path / "a"), mine), one_layer(model, tokenizer, base, str(tmp_path / "b"), theirs)
    with pytest.raises(ValueError, match="another answer .* acme \\| owner"):
        merge_grafted_models(model, tokenizer, base, a, b, str(tmp_path / "no"), paragraphs=TEXTS)
    assert not os.path.exists(str(tmp_path / "no"))                                            # refused before anything is written
    kept = merge_grafted_models(model, tokenizer, base, a, b, str(tmp_path / "kept"), prefer="mine", paragraphs=TEXTS)
    assert (kept["facts"], kept["same"], kept["replaced"]) == (3, 1, []) and kept["kept"] == [dict(subject="acme", relation="owner", answer="julian rodlauer")]
    one_layer(model, tokenizer, base, str(tmp_path / "once_mine"), mine + [theirs[2]])
    assert same_weights(str(tmp_path / "kept"), str(tmp_path / "once_mine"))
    taken = merge_grafted_models(model, tokenizer, base, a, b, str(tmp_path / "taken"), prefer="theirs", paragraphs=TEXTS)
    assert taken["replaced"] == [dict(subject="acme", relation="owner", answer="paris")] and taken["facts"] == 3
    one_layer(model, tokenizer, base, str(tmp_path / "once_theirs"), [mine[1], theirs[2], theirs[0]])   # as a merge of the two tables orders them
    assert same_weights(str(tmp_path / "taken"), str(tmp_path / "once_theirs"))
    assert {(n["subject"], n["answer"]) for n in spec(str(tmp_path / "taken"))["neurons"]} == {("france", "rome"), ("rome", "y"), ("acme", "paris")}


@pytest.mark.parametrize("budget, moved", [(20, 0), (10, 2), (8, 4)])
def test_at_the_last_layer_the_budget_is_shared_out_again_as_one_export_would(model, tokenizer, base, tmp_path, budget, moved):
    """Each model alone fits the last layer (4 first-token neurons and 2 later tokens). Together they need 12 of its neurons: with
    a budget of 20 they all stay, with 10 both third tokens move to the other layer, with 8 all four later tokens do."""
    top = len(model.model.layers) - 1
    mine = [fact("acme", "owner", "julian rodlauer paris", 1, ids=(7, 9, 11))]
    theirs = [fact("rome", "owner", "x y z", 2, ids=(12, 13, 14))]
    a, b, out, once = (str(tmp_path / n) for n in ("a", "b", "merged", "once"))
    last_layer(model, tokenizer, base, a, mine, budget)
    last_layer(model, tokenizer, base, b, theirs, budget)
    assert "later_layer" not in spec(a) and "later_layer" not in spec(b) and read_later(a)["layer"] == LAYER   # each edits the last layer only
    done = merge_grafted_models(model, tokenizer, base, a, b, out, paragraphs=TEXTS, last_budget=budget)
    last_layer(model, tokenizer, base, once, mine + theirs, budget)
    assert same_weights(out, once)                                                             # also the neurons that had to move
    assert spec(out)["neurons"] == spec(once)["neurons"] and spec(out)["answers"] == spec(once)["answers"]
    assert done["same_as_one_export"] is True and done["last_layer"]["later_elsewhere"] == moved
    assert done["neurons"] == ({top: 12} if not moved else {LAYER: moved, top: 12 - moved})
    assert spec(out).get("later_layer") == (LAYER if moved else None)
    assert open(os.path.join(out, LATER_FILE), "rb").read() == open(os.path.join(once, LATER_FILE), "rb").read()   # kept for the next merge
    third = [fact("france", "capital", "rome x", 3, ids=(15, 12))]
    c = last_layer(model, tokenizer, base, str(tmp_path / "c"), third, budget)
    again = merge_grafted_models(model, tokenizer, base, out, c, str(tmp_path / "three"), paragraphs=TEXTS, last_budget=budget)
    last_layer(model, tokenizer, base, str(tmp_path / "once3"), mine + theirs + third, budget)
    assert same_weights(str(tmp_path / "three"), str(tmp_path / "once3")) and again["facts"] == 3   # a merged model merges again


def test_what_cannot_be_merged_is_refused_before_anything_is_written(model, tokenizer, base, tmp_path):
    mine, theirs = [fact("acme", "owner", "julian rodlauer", 1)], [fact("rome", "owner", "y", 4)]
    a, b = one_layer(model, tokenizer, base, str(tmp_path / "a"), mine), one_layer(model, tokenizer, base, str(tmp_path / "b"), theirs)
    out = str(tmp_path / "out")
    other_way = one_layer(model, tokenizer, base, str(tmp_path / "sat"), theirs, saturate=True, where="last")
    with pytest.raises(ValueError, match="can't be merged at the weights: where, saturate differ"):
        merge_grafted_models(model, tokenizer, base, a, other_way, out, paragraphs=TEXTS)
    with pytest.raises(ValueError, match="a folder of its own"):
        merge_grafted_models(model, tokenizer, base, a, b, a, paragraphs=TEXTS)
    with pytest.raises(ValueError, match="not a grafted model"):
        merge_grafted_models(model, tokenizer, base, a, base, out, paragraphs=TEXTS)

    changed = shutil.copytree(b, str(tmp_path / "changed"))                                    # a model that is not this base's
    held = load_file(os.path.join(changed, "model.safetensors"))
    held["model.layers.0.mlp.down_proj.weight"] = held["model.layers.0.mlp.down_proj.weight"] + 1.0
    save_file(held, os.path.join(changed, "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="was not exported from the base .* model.layers.0.mlp.down_proj.weight differs"):
        merge_grafted_models(model, tokenizer, base, a, changed, out, paragraphs=TEXTS)

    weight = model.model.layers[LAYER].mlp.down_proj.weight
    was = weight[0, 0].item()
    with torch.no_grad():
        weight[0, 0] = was + 1.0                                                               # a weight edit kept in the loaded model
    with pytest.raises(ValueError, match="does not hold the base's weights at layer 2"):
        merge_grafted_models(model, tokenizer, base, a, b, out, paragraphs=TEXTS)
    with torch.no_grad():
        weight[0, 0] = was
    assert not os.path.exists(out)

    long_a = [fact("acme", "owner", "julian rodlauer paris", 1, ids=(7, 9, 11))]
    long_b = [fact("rome", "owner", "x y z", 2, ids=(12, 13, 14))]
    old = last_layer(model, tokenizer, base, str(tmp_path / "old"), long_a, 8, keep=False)     # as an export before the vectors were kept
    new = last_layer(model, tokenizer, base, str(tmp_path / "new"), long_b, 8)
    with pytest.raises(ValueError, match="exported without its later answer vectors .* 2 later answer token"):
        merge_grafted_models(model, tokenizer, base, old, new, out, paragraphs=TEXTS, last_budget=8)
    roomy = merge_grafted_models(model, tokenizer, base, old, new, out, paragraphs=TEXTS, last_budget=20)   # nothing has to move: merged
    assert roomy["facts"] == 2 and roomy["same_as_one_export"] is True and not os.path.exists(os.path.join(out, LATER_FILE))


def test_the_merged_model_is_recorded_asked_and_served(model, tokenizer, base, tmp_path, capsys, monkeypatch):
    from cramschool.__main__ import main
    from cramschool.tools import GraftTools

    # The command loads the base model itself. On another device than the exports were made on, the moments differ in their last
    # bits, and the merge then says, rightly, that it can't vouch for the weights being those of one export.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    mine, theirs = [fact("acme", "owner", "julian rodlauer", 1)], [fact("rome", "owner", "y", 4)]
    a, b = one_layer(model, tokenizer, base, str(tmp_path / "a"), mine), one_layer(model, tokenizer, base, str(tmp_path / "b"), theirs)
    h, out = str(tmp_path / "weights.history"), str(tmp_path / "merged")
    wh.commit(h, a, "mine", base_dir=base)
    start = {k: v.clone() for k, v in model.state_dict().items()}
    done = wh.merge(h, a, b, out, model, tokenizer, "both", base_dir=base, paragraphs=TEXTS)
    v2 = done["version"]
    assert (v2.id, v2.facts, v2.table, v2.message) == ("v2", 2, None, "both") and wh.is_intact(h) and done["merged"]["facts"] == 2
    assert wh.diff(h, "v1", "v2")["facts"]["added"] == [dict(subject="rome", relation="owner", answer="y")]
    with wh.applied(h, "v2", model, tokenizer=tokenizer) as switch:                             # the version is set on a loaded model
        assert switch is not None and torch.equal(model.model.layers[LAYER].mlp.down_proj.weight.data,
                                                  tensors(out)[f"model.layers.{LAYER}.mlp.down_proj.weight"])
    assert all(torch.equal(v, start[k]) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match="not the base this history was started on"):
        wh.merge(h, a, b, str(tmp_path / "x"), model, tokenizer, base_dir=a)

    cli, text = str(tmp_path / "cli"), str(tmp_path / "ordinary.txt")
    open(text, "w", encoding="utf-8").write(chr(10).join(TEXTS))
    assert main(["weights", "merge", "--model", base, "--dtype", "float32", "--mine", a, "--theirs", b, "--out", cli, "--history", h,
                 "-m", "from the command line", "--calibration", text]) == 0
    printed = capsys.readouterr().out
    assert "1 added, 0 replaced, 0 kept as they were, 0 the same; 2 facts in the merged model" in printed and "4 at layer 2" in printed
    assert "the weights are the ones a single export of all these facts gives" in printed
    assert "nothing" not in printed and same_weights(cli, out)                                  # the same weights as the call from Python
    assert main(["weights", "ask", "--model", base, "--dtype", "float32", "--history", h, "--versions", "v2,base", "--plain", "the owner of acme is"]) == 0
    assert "v2:" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="another answer"):
        main(["weights", "merge", "--model", base, "--dtype", "float32", "--mine", a, "--theirs",
              one_layer(model, tokenizer, base, str(tmp_path / "clash"), [fact("acme", "owner", "paris", 7)]), "--out", str(tmp_path / "no"),
              "--calibration", text])

    class Table:                                               # stands in for the server's table: the loaded base model and where it is
        def __init__(self):
            self.model, self._given_tokenizer, self._calibration = model, tokenizer, TEXTS

        def _base_folder(self, base_dir):
            return base

        def detached(self):
            import contextlib

            return contextlib.nullcontext(self)

    tools = GraftTools(Table())
    tool = tools.call("weights_merge", dict(mine=a, theirs=b, out_dir=str(tmp_path / "tool"), history_dir=h, message="by the tool"))
    assert (tool["facts"], tool["neurons"], tool["weights"]["facts"], tool["weights"]["compiled_from_table"]) == (2, {"2": 4}, 2, None)
    assert same_weights(str(tmp_path / "tool"), cli)
    plain = tools.call("weights_merge", dict(mine=a, theirs=b, out_dir=str(tmp_path / "tool2")))
    assert plain["weights"] is None and plain["folder"] == str(tmp_path / "tool2")
    assert "another answer" in tools.call("weights_merge", dict(mine=a, theirs=str(tmp_path / "clash"), out_dir=str(tmp_path / "no")))["error"]


def test_the_server_serves_the_weights_merge():
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client

    from cramschool.mcp_server import build_server
    from cramschool.tools import TOOLS, GraftTools

    server = build_server(lambda: None, GraftTools(None))

    async def check():
        async with Client(server) as client:
            found = {t.name: t for t in (await client.list_tools()).tools}
            tool = found["cramschool_graft_weights_merge"]
            assert set(tool.input_schema["properties"]) == set(TOOLS["weights_merge"][1]["properties"])   # the server's arguments are the tool's

    anyio.run(check)


def test_an_export_from_a_table_says_what_its_neurons_were_refitted_from(model, tokenizer, base, tmp_path):
    """The hash of the moments is what lets a merge say that its weights are those of one export."""
    a = one_layer(model, tokenizer, base, str(tmp_path / "a"), [fact("acme", "owner", "julian rodlauer", 1)])
    b = one_layer(model, tokenizer, base, str(tmp_path / "b"), [fact("rome", "owner", "y", 4), fact("paris", "capital", "x", 3)])
    assert len(spec(a)["refit"]["moments"]) == 16 and spec(a)["refit"]["moments"] == spec(b)["refit"]["moments"]   # the same layer, model and text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compiled = compile_layer(model, tokenizer, [fact("acme", "owner", "julian rodlauer", 1)], LAYER, mctx(), forms=FORMS, paragraphs=TEXTS[:1] * 30 + ["x y z"] * 30)
    assert compiled["refit"]["moments"] != spec(a)["refit"]["moments"]                          # other text, other moments

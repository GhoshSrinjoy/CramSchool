"""What used to pass silently and now says so: a fact with no detector, stored facts that a new relation leaves untaught, an export
over the last layer's budget, a loaded table that forgot its own ordinary text; and the settings a server-made table can take."""
import json
import os
import warnings

import pytest
import torch

from cramschool.graft import Graft, core
from cramschool.graft.core import blocks
from cramschool.tools import GraftTools


def ordinary(model, tokenizer) -> list[str]:
    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    words = [w for w, i in tokenizer.get_vocab().items() if 2 < i < model.config.vocab_size]
    rng = torch.Generator().manual_seed(0)
    return [" ".join(words[int(i)] for i in torch.randint(len(words), (40,), generator=rng)) for _ in range(360)]


def test_the_editor_is_told_what_a_fact_got_and_what_a_new_relation_leaves_behind(model, tokenizer):
    text = ordinary(model, tokenizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tools = GraftTools(Graft(model, tokenizer, layer=2, calibration=text))
        added = tools.call("add_fact", dict(subject="acme", relation="owner", answer="julian rodlauer"))
    assert set(added["detectors"]) <= {"plain", "chat"} and ("note" in added) == (len(added["detectors"]) < 2)
    if "note" in added:
        assert "no detector could be built for" in added["note"] and "teach_wordings" in added["note"]
    plain = [f"the {w} of {{s}} is" for w in ("x", "y", "z", "fact", "name", "answer", "question", "product")]
    chat = [f"what is the {w} of {{s}} ?" for w in ("x", "y", "z", "fact")]
    new = tools.call("add_relation", dict(name="motto", plain=plain, chat=chat, person=False))
    assert "1 stored fact about a company was built before this relation existed: call rebuild" in new["note"]
    other = tools.call("add_relation", dict(name="birthplace", plain=plain, chat=chat, person=False, subject="person"))
    assert "note" not in other                                                         # no stored fact is about a person
    missing = tools.call("check_fact", dict(subject="nobody", relation="owner", new_questions=["who owns nobody ?"]))
    assert missing["error"] == "no stored fact for 'nobody' and 'owner'"               # the message whole, with its closing quote


def test_a_saved_table_keeps_the_ordinary_text_it_was_given(model, tokenizer, tmp_path):
    text = ordinary(model, tokenizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = Graft(model, tokenizer, layer=2, calibration=text)
        table.add("acme", "owner", "julian rodlauer")
        folder = table.save(str(tmp_path / "t.graft"))
        assert json.load(open(os.path.join(folder, "ordinary_text.json"), encoding="utf-8")) == text
        assert Graft.load(folder, model, tokenizer)._calibration == text                # so that it exports with that text, not the bundled one
        bundled = Graft(model, tokenizer, layer=2)                                      # no text given: nothing is written
        bundled._mctx, bundled._facts, bundled._shared, bundled._shared_relations = table._mctx, table._facts, table._shared, table._shared_relations
        assert not os.path.exists(os.path.join(bundled.save(str(tmp_path / "b.graft")), "ordinary_text.json"))


def test_an_export_over_the_last_layers_budget_says_so(model, tokenizer, tmp_path, monkeypatch):
    from cramschool.graft import grafted

    text = ordinary(model, tokenizer)
    base = str(tmp_path / "base")
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = Graft(model, tokenizer, layer="last", calibration=text)
        table.add("acme", "owner", "julian rodlauer")
    assert table.layer == len(blocks(model)) - 1
    plan = grafted.last_layer_plan

    def crowded(*args, **kwargs):                              # as a table with many facts: more first tokens than the layer can spare
        here, there, counts = plan(*args, **kwargs)
        return here, there, dict(counts, first=counts["budget"] + 5)

    monkeypatch.setattr(grafted, "last_layer_plan", crowded)
    with pytest.warns(UserWarning, match="first-token neurons at the last layer, over its budget of"):
        table.export_grafted(str(tmp_path / "g"), base_dir=base, later_layer=2)


def test_the_server_command_takes_a_new_tables_settings():
    from cramschool.__main__ import build_parser

    args = build_parser().parse_args(["serve", "--model", "m", "--layer", "auto", "--system", "You are terse.", "--prompt-detector", "--fast"])
    assert (args.layer, args.system, args.prompt_detector, args.fast) == ("auto", "You are terse.", True, True)
    assert build_parser().parse_args(["serve", "--model", "m", "--layer", "22"]).layer == 22
    plain = build_parser().parse_args(["serve", "--model", "m"])
    assert (plain.layer, plain.system, plain.prompt_detector) == (None, None, False)     # no layer given: measured on the model

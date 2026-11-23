import json

import pytest

from cramschool.__main__ import build_parser, read_calibration


def test_calibration_defaults_to_the_bundled_paragraphs():
    args = build_parser().parse_args(["add", "--model", "m", "--subject", "s", "--target", "t", "--relation", "owner"])
    assert args.calibration is None
    paragraphs = read_calibration(args.calibration)
    assert len(paragraphs) == 355 and all(paragraphs)


def test_the_version_flag_prints_the_package_version(capsys):
    import cramschool

    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(["--version"])
    assert stop.value.code == 0 and capsys.readouterr().out.strip() == f"cramschool {cramschool.__version__}"


def test_arguments_parse():
    args = build_parser().parse_args(
        ["add", "--model", "m", "--calibration", "c.txt", "--subject", "s", "--target", "t", "--relation", "owner", "--layer", "4", "--keep-failing"]
    )
    assert args.layer == 4 and args.keep_failing and args.calibration == "c.txt"


def test_the_grafted_model_has_an_export_command():
    args = build_parser().parse_args(["graft", "export", "--model", "m", "--table", "t.graft", "--out", "grafted", "--plain-only"])
    assert (args.graft_command, args.table, args.out, args.base, args.plain_only, args.first_token_only) == ("export", "t.graft", "grafted", None, True, False)


def test_aliases_are_declared_in_the_table_file(tmp_path, capsys):
    from cramschool.__main__ import main

    table = tmp_path / "t.graft"
    table.mkdir()
    (table / "graft.json").write_text(json.dumps(dict(layer=20, model="m", facts=[dict(id="f01", subject="Vorantis Biotech", relation="ceo", answer="Ingrid Solberg"),
                                                                                    dict(id="f02", subject="Moqueldar Chemicals", relation="hq", answer="Tartu")])),
                                      encoding="utf-8")
    assert main(["graft", "alias", "--table", str(table), "--subject", "vorantis biotech", "Vorantis", "VB", "Vorantis"]) == 0
    assert json.loads((table / "graft.json").read_text(encoding="utf-8"))["aliases"] == {"Vorantis Biotech": ["Vorantis", "VB"]}
    with pytest.raises(SystemExit, match="already names"):
        main(["graft", "alias", "--table", str(table), "--subject", "Moqueldar Chemicals", "vb"])
    with pytest.raises(SystemExit, match="no fact is stored"):
        main(["graft", "alias", "--table", str(table), "--subject", "Nimbresco", "N"])
    assert main(["graft", "list", "--table", str(table)]) == 0
    assert "Vorantis Biotech is also called: Vorantis, VB" in capsys.readouterr().out


def test_write_settings_parse():
    args = build_parser().parse_args(
        ["add", "--model", "m", "--calibration", "c.txt", "--subject", "s", "--target", "t", "--relation", "owner", "--layer", "20", "--null-keys", "all",
         "--strength", "0.9", "--margin", "3", "--ridge", "0.3", "--metric-ridge", "3"]
    )
    assert (args.layer, args.null_keys, args.strength, args.margin, args.ridge, args.metric_ridge) == (20, "all", 0.9, 3.0, 0.3, 3.0)


@pytest.mark.parametrize("flag", [["--harm-penalty", "5"], ["--l2", "0.001"], ["--key-site", "last"], ["--per-prompt-keys"], ["--through-edit"]])
def test_the_gradient_solver_has_no_options_left(flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["add", "--model", "m", "--subject", "s", "--target", "t", "--relation", "owner"] + flag)


def test_the_command_line_offers_the_same_choices_as_the_package():
    from cramschool import __main__ as cli
    from cramschool import facts

    assert cli.NULL_KINDS == facts.NULL_KINDS


def test_a_rejected_edit_exits_with_a_failure_code(monkeypatch, tmp_path, capsys):
    """Weights only: when the edit is rejected nothing was written, and a script must be able to tell."""
    from types import SimpleNamespace

    import transformers

    import cramschool

    calibration = tmp_path / "c.txt"
    calibration.write_text("some ordinary text\n", encoding="utf-8")
    report = SimpleNamespace(to_dict=lambda: dict(failures=["recall"]))

    class Editor:
        def __init__(self, *_a, **_k):
            pass

        def add_fact(self, _fact, on_failure):
            return SimpleNamespace(accepted=on_failure == "keep", seconds=1.0, report=report)

    class Loader:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return SimpleNamespace(eval=lambda: None)

    monkeypatch.setattr(transformers, "AutoTokenizer", Loader)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", Loader)
    monkeypatch.setattr(cramschool, "KnowledgeEditor", Editor)
    base = ["add", "--model", "m", "--calibration", str(calibration), "--subject", "RX Systems", "--target", "Julian Rodlauer", "--relation", "owner"]
    from cramschool.__main__ import main

    assert main(base) == 1
    assert main([*base, "--keep-failing"]) == 0


def test_read_calibration_skips_blank_lines(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("first paragraph\n\n  \nsecond paragraph\n", encoding="utf-8")
    assert read_calibration(str(path)) == ["first paragraph", "second paragraph"]


def test_an_empty_calibration_file_is_refused(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        read_calibration(str(path))


def test_benches_of_other_kinds_have_their_commands(tmp_path, capsys):
    from cramschool.__main__ import main

    assert main(["bench", "drafted-brief", "--out", str(tmp_path / "d.txt"), "--kinds", "town"]) == 0
    assert "TOWN" in (tmp_path / "d.txt").read_text(encoding="utf-8") and "MEDICINE" not in (tmp_path / "d.txt").read_text(encoding="utf-8")
    src = dict(seed=3, facts=[["Quorvex T-410", "product_year", "2014"]],
               wordings={"product_year": dict(plain=[f"{{s}} came out {i} in" for i in range(16)], chat=[f"When ({i}) did {{s}} come out?" for i in range(8)])},
               near={"product_year": dict(plain=[f"{{s}} was announced {i} in" for i in range(4)], chat=[f"When ({i}) was {{s}} announced?" for i in range(4)])},
               generic={"product": dict(plain=[f"{{s}} weighs {i}" for i in range(6)], chat=[f"How heavy ({i}) is {{s}}?" for i in range(6)])})
    path = tmp_path / "src.json"
    path.write_text(json.dumps(src), encoding="utf-8")
    assert main(["bench", "check", "--source", str(path), "--avoid-from"]) == 1          # one fact is not a bench: the counts are named
    assert "product: 1 facts" in capsys.readouterr().out
    assert main(["bench", "build", "--kinds", "--source", str(path), "--out", str(tmp_path / "b.json")]) == 0
    built = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))["facts"]
    assert built[0]["kind"] == "product" and built[0]["answer_type"] == "year" and len(built[0]["test"]["near_plain"]) == 2


def test_a_tables_settings_are_command_line_flags():
    add = build_parser().parse_args(["graft", "add", "--model", "m", "--table", "t", "--subject", "s", "--relation", "ceo", "--answer", "a",
                                     "--system", "You are a help desk.", "--prompt-detector", "--last-layer", "none", "--fast"])
    assert (add.system, add.prompt_detector, add.last_layer, add.fast) == ("You are a help desk.", True, "none", True)
    plain = build_parser().parse_args(["graft", "add-file", "--model", "m", "--table", "t", "--facts", "f.csv"])
    assert (plain.system, plain.prompt_detector, plain.last_layer, plain.fast) == (None, False, None, False)
    ask = build_parser().parse_args(["graft", "ask", "--model", "m", "--table", "t", "--last-layer", "all", "who ?"])
    assert ask.last_layer == "all" and not hasattr(ask, "system")
    change = build_parser().parse_args(["graft", "set", "--model", "m", "--table", "t", "--prompt-detector", "off"])
    assert (change.graft_command, change.prompt_detector, change.last_layer) == ("set", "off", None)


def test_the_set_command_changes_a_saved_table_and_a_new_table_takes_its_flags(model, tokenizer, tmp_path, monkeypatch, capsys):
    import cramschool.__main__ as cli
    from cramschool.graft import Graft, core
    from cramschool.graft.table import GraftOptions

    core._TEXTS[id(tokenizer)] = [" " + t for t in tokenizer.batch_decode([[i] for i in range(len(tokenizer))])]
    monkeypatch.setattr(cli, "_load", lambda path, dtype: (model, tokenizer))
    table = str(tmp_path / "t.graft")
    old = Graft(model, tokenizer, layer=2, options=GraftOptions(last_layer=None))   # a table with the write of before 1.1
    old.add("acme", "owner", "julian rodlauer")
    old.save(table)
    assert Graft.load(table, model, tokenizer).options.last_layer is None
    assert cli.main(["graft", "set", "--model", "m", "--table", table, "--last-layer", "all"]) == 0
    assert "last_layer='all'" in capsys.readouterr().out
    saved = Graft.load(table, model, tokenizer)
    assert saved.options.last_layer == "all" and saved.calibration()["final_scale"] > 0   # the scale is saved with it
    assert cli.main(["graft", "ask", "--model", "m", "--table", table, "--last-layer", "none", "who owns acme ?"]) == 0   # for this call only
    assert Graft.load(table, model, tokenizer).options.last_layer == "all"
    with pytest.raises(SystemExit, match="with no system message"):
        cli.main(["graft", "add", "--model", "m", "--table", table, "--subject", "acme", "--relation", "ceo", "--answer", "paris", "--system", "the records show"])
    args = build_parser().parse_args(["graft", "add", "--model", "m", "--table", "n", "--subject", "s", "--relation", "ceo", "--answer", "a",
                                      "--system", "the records show", "--last-layer", "later", "--fast"])
    new = cli._open_table(model, tokenizer, str(tmp_path / "n.graft"), 2, None, args)
    assert (new.system, new.options.last_layer, new.options.fast, new.options.prompt_detector, new.layer) == ("the records show", "later", True, False, 2)

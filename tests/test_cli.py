import pytest

from cramschool.__main__ import build_parser, read_calibration


def test_calibration_defaults_to_the_bundled_paragraphs():
    args = build_parser().parse_args(["add", "--model", "m", "--subject", "s", "--target", "t", "--relation", "owner"])
    assert args.calibration is None
    paragraphs = read_calibration(args.calibration)
    assert len(paragraphs) == 355 and all(paragraphs)


def test_arguments_parse():
    args = build_parser().parse_args(
        ["add", "--model", "m", "--calibration", "c.txt", "--subject", "s", "--target", "t", "--relation", "owner", "--layer", "4", "--keep-failing"]
    )
    assert args.layer == 4 and args.keep_failing and args.calibration == "c.txt"


def test_key_settings_parse():
    args = build_parser().parse_args(
        ["add", "--model", "m", "--calibration", "c.txt", "--subject", "s", "--target", "t", "--relation", "owner", "--layer", "20", "--key-site", "last", "--null-keys", "all",
         "--per-prompt-keys", "--through-edit", "--strength", "0.9", "--ridge", "0.3", "--metric-ridge", "3", "--harm-penalty", "5", "--l2", "0.0005"]
    )
    assert (args.key_site, args.null_keys, args.per_prompt_keys, args.through_edit, args.strength, args.ridge, args.metric_ridge, args.harm_penalty, args.l2) == ("last", "all", True, True, 0.9, 0.3, 3.0, 5.0, 0.0005)


def test_the_command_line_offers_the_same_choices_as_the_package():
    from cramschool import __main__ as cli
    from cramschool import facts

    assert cli.KEY_SITES == facts.KEY_SITES and cli.NULL_KINDS == facts.NULL_KINDS


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

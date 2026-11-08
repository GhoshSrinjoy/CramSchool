"""The live progress view: the status a surgery writes while it works, the readable lines, the drawing, and the rules for opening a window."""
import json
import os
import subprocess

import pytest

from cramschool import GateConfig, live
from cramschool.surgery import Surgery

OPEN = GateConfig(min_recall=0.0, max_leak=1.0, max_cross_leak=1.0, max_garbled=1.0, max_known_drop=1.0, max_kl=1e9, min_retention=0.0)
PAIRS = [
    dict(start="question : what is the name ? answer :", good=" the company is here", bad=" is is is is is"),
    dict(start="who owns acme ? answer :", good=" julian rodlauer owns it", bad=" the the the the the"),
]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_each_step_reads_as_one_line():
    assert live.summarize("keep", {}, dict(kept="fix", strength=1.0, sha256_after={"a": "x"})).startswith("KEPT 'fix'")
    assert live.summarize("test", {}, dict(error="No proposal named 'x'")) == "test: error - No proposal named 'x'"
    assert "x0.5 better (bad -0.200)" in live.summarize("test", {}, dict(proposal="fix", tried=[dict(strength=0.5, verdict="better", changes=dict(bad=-0.2))]))
    line = live.summarize("confirm", {}, dict(proposal="fix", problems=32, unedited=dict(correct=0.1, repeats=0.6), edited=dict(correct=0.5, repeats=0.1), correct=dict(rescued=14, broken=1)))
    assert "correct 10% -> 50%" in line and "rescued 14, broken 1" in line
    answers = [dict(finished=True, looping=False, final_answer="42"), dict(finished=False, looping=True, final_answer="")]
    assert "2 answer(s), 1 finished thinking, 1 looping; first: '42'" in live.summarize("generate", {}, dict(thinking=True, answers=answers))


def test_the_view_waits_for_a_session_then_shows_it():
    assert "Waiting for a session" in live.render(None, [], gpu="GPU")
    status = dict(
        started_at=100.0, model="tiny", budget=10, passes_used=4, kept=["fix"], pid=os.getpid(), log="s.jsonl",
        current=dict(tool="confirm", detail="edited model: problems 9-16 of 32", fraction=0.5, started_at=70.0, notes=["unedited model: correct 6/32"]),
    )
    entries = [dict(step=1, tool="keep", time="2026-11-05T14:00:00", summary="KEPT 'fix' at strength 1.0", result=dict(seconds=0.1), passes=0)]
    text = live.render(status, entries, now=100.0, gpu="GPU")
    assert "4 of 10 forward passes" in text and "Kept     fix" in text
    assert "confirm" in text and " 50%" in text and "about 0m30s left" in text and "- unedited model: correct 6/32" in text
    assert "#1   14:00:00  KEPT 'fix'" in text
    colored = live.render(status, entries, now=100.0, gpu="GPU", color=True)
    assert "\x1b[32mKEPT\x1b[0m" in colored


def test_the_view_fits_the_window_and_keeps_the_newest_steps():
    status = dict(started_at=0.0, model="tiny", budget=99, passes_used=1, kept=[], pid=os.getpid(), current=None)
    entries = [dict(step=i, tool="status", summary="x" * 300, result={}, passes=0) for i in range(1, 60)]
    lines = live.render(status, entries, width=80, height=30, gpu="GPU").splitlines()
    assert len(lines) <= 30 and max(map(len, lines)) < 80
    assert lines[-1].startswith("  #59") and "earlier steps are in the log file" in "\n".join(lines) and not any(l.startswith("  #1 ") for l in lines)


def test_a_finished_session_says_so():
    import time

    status = dict(started_at=0.0, model="tiny", budget=9, passes_used=1, kept=[], pid=2**31 - 7, current=dict(tool="confirm", detail="x", fraction=0.2, started_at=0.0))
    assert "stopped during confirm" in live.render(status, [], gpu="GPU")
    started = time.mktime(time.strptime("2026-11-06T01:30:00", "%Y-%m-%dT%H:%M:%S"))
    entries = [dict(step=1, tool="keep", time="2026-11-06T02:00:00", summary="KEPT 'fix' at strength 1.0", result={}, passes=0)]
    text = live.render(dict(status, started_at=started, current=None), entries, gpu="GPU")
    assert "ran 30m00s, ended" in text and "session ended" in text and "running" not in text


def test_a_surgery_reports_progress_while_it_works(model, tokenizer, tmp_path):
    log = str(tmp_path / "s.jsonl")
    s = Surgery(model, tokenizer, gates=OPEN, budget=20, log_path=log)
    status_file = live.status_path_for(log)
    first = load(status_file)
    assert first["current"] is None and first["budget"] == 20 and first["pid"] == os.getpid() and first["log"] == os.path.abspath(log)

    seen = []
    original = s.tools["find_direction"].run

    def spy(**args):
        out = original(**args)
        seen.append(load(status_file))
        return out

    s.tools["find_direction"].run = spy
    s.call("find_direction", {"name": "loops", "pairs": PAIRS})
    during = seen[0]["current"]
    assert during["tool"] == "find_direction" and during["fraction"] > 0.5 and "MLP writes" in during["detail"]
    assert any(n.startswith("strongest writers") for n in during["notes"])

    after = load(status_file)
    assert after["current"] is None and after["passes_used"] == 1
    assert "passes_left" in s.call("status", {})  # the status tool is still the status tool
    entries = live.read_log(log)
    assert entries[0]["summary"].startswith("find_direction 'loops'") and entries[0]["time"][:2] == "20" and s.log_is_intact()
    assert "find_direction 'loops'" in live.render(after, entries, gpu="GPU")


def test_a_crash_inside_a_tool_is_an_error_and_clears_the_current_step(model, tokenizer, tmp_path):
    log = str(tmp_path / "s.jsonl")
    s = Surgery(model, tokenizer, gates=OPEN, log_path=log)

    def crash():
        raise RuntimeError("CUDA out of memory")

    s.tools["layer_map"].run = crash
    result = s.call("layer_map", {})
    assert "RuntimeError while running layer_map: CUDA out of memory" in result["error"] and s.passes_used == 0
    assert load(live.status_path_for(log))["current"] is None
    assert live.read_log(log)[0]["summary"].startswith("layer_map: error - RuntimeError")


def test_echo_prints_readable_steps_to_stderr_only(model, tokenizer, capsys):
    s = Surgery(model, tokenizer, gates=OPEN, echo=True)
    s.call("find_direction", {"name": "loops", "pairs": PAIRS})
    out = capsys.readouterr()
    assert out.out == "" and "#1 find_direction 'loops'" in out.err and "strongest writers" in out.err


def test_long_generations_report_progress_within_a_batch(model, tokenizer):
    import torch

    from cramschool.problems import Every, think

    seen = []
    every = Every(seen.append, every=64)
    flags = [every(torch.zeros(2, 3, dtype=torch.long), None) for _ in range(130)]
    assert seen == [64, 128] and flags[0].shape == (2,) and not any(bool(f.any()) for f in flags)  # it reports, and never stops generation
    calls = []
    think(model, tokenizer, ["what is 1 + 1 ?"], 4, on_batch=lambda *args: calls.append(args))
    assert calls[0] == (0, 1, 1, 0.0)


def test_the_newest_session_in_a_folder_is_followed(tmp_path):
    for name, when in (("a.jsonl", 100), ("b.jsonl", 200)):
        path = live.status_path_for(str(tmp_path / name))
        live.write_status(path, dict(model=name))
        os.utime(path, (when, when))
    assert live.newest_session(str(tmp_path)) == str(tmp_path / "b.jsonl")
    assert live.newest_session(str(tmp_path / "missing")) is None


def test_status_writes_never_raise(monkeypatch, tmp_path):
    def locked(*_):
        raise PermissionError("in use by the viewer")

    monkeypatch.setattr(live.os, "replace", locked)
    monkeypatch.setattr(live.time, "sleep", lambda _: None)
    live.write_status(str(tmp_path / "x.status.json"), dict(a=1))


def test_the_window_gets_no_handle_to_this_process_output(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(live.subprocess, "Popen", lambda command, **kw: calls.append((command, kw)) or "window")
    assert live.open_viewer(str(tmp_path)) == "window"
    command, kw = calls[0]
    assert command[-1] == str(tmp_path) and command[1] == "-I"
    if os.name == "nt":
        assert not {"stdin", "stdout", "stderr"} & set(kw) and kw["creationflags"] & subprocess.CREATE_NEW_CONSOLE
    else:
        assert kw["stdout"] is subprocess.DEVNULL


def test_a_second_window_is_not_opened(monkeypatch, tmp_path):
    (tmp_path / live.VIEWER_PID).write_text(str(os.getpid()))
    monkeypatch.setattr(live.subprocess, "Popen", lambda *a, **k: pytest.fail("opened a second window"))
    assert live.open_viewer(str(tmp_path)) is None


def test_the_viewer_runs_on_its_own_without_the_package(tmp_path):
    done = subprocess.run(live.viewer_command(str(tmp_path)) + ["--once"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "Waiting for a session" in done.stdout


def test_the_command_line_has_watch_and_serve_flags(tmp_path, capsys):
    from cramschool.__main__ import build_parser, main

    assert main(["watch", str(tmp_path), "--once"]) == 0
    assert "Waiting for a session" in capsys.readouterr().out
    args = build_parser().parse_args(["serve", "--model", "m", "--log-dir", "d", "--watch"])
    assert args.watch and args.log_dir == "d" and args.log is None

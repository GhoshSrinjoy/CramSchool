"""The merged weight-space programming workflow on a tiny random model: rules, budget, log, every tool, and the Claude loop with a stand-in client."""
from types import SimpleNamespace

import pytest
import torch

from cramschool import GateConfig, writer_module
from cramschool.claude_editor import run_with_claude
from cramschool.surgery import PROTOCOL, Surgery

OPEN = GateConfig(min_recall=0.0, max_leak=1.0, max_cross_leak=1.0, max_garbled=1.0, max_known_drop=1.0, max_kl=1e9, min_retention=0.0)
PAIRS = [
    dict(start="question : what is the name ? answer :", good=" the company is here", bad=" is is is is is"),
    dict(start="who owns acme ? answer :", good=" julian rodlauer owns it", bad=" the the the the the"),
]
FACT = dict(subject="acme", target="julian rodlauer", prompts=["the owner of acme is"], questions=["who owns acme ?"], new_prompts=["records show that the person who owns acme is"])


def snapshot(model):
    return {n: p.data.clone() for n, p in model.named_parameters()}


def assert_same(model, saved):
    for n, p in model.named_parameters():
        assert torch.equal(p.data, saved[n]), n


def surgery(model, tokenizer, **kw):
    return Surgery(model, tokenizer, gates=OPEN, **kw)


def test_every_tool_has_a_closed_schema():
    s = Surgery.__new__(Surgery)
    for definition in Surgery._tools(s):
        schema = definition.schema()
        assert schema["input_schema"]["additionalProperties"] is False
        assert "Cost:" in schema["description"]
    assert "no gradients" in PROTOCOL.lower() or "There are no gradients" in PROTOCOL


def test_bad_calls_are_refused_and_cost_nothing(model, tokenizer):
    s = surgery(model, tokenizer)
    assert "Unknown tool" in s.call("train", {})["error"]
    assert "missing" in s.call("find_direction", {"name": "x"})["error"]
    assert "unknown" in s.call("status", {"learning_rate": 1})["error"]
    assert "must be a array" in s.call("test", {"name": "x", "strengths": 1})["error"]
    assert s.passes_used == 0


def test_a_full_operation_follows_the_method_and_undoes_exactly(model, tokenizer, tmp_path):
    saved = snapshot(model)
    s = surgery(model, tokenizer, budget=40, log_path=str(tmp_path / "log.jsonl"))
    assert "answers" in s.call("generate", {"prompts": ["the owner of acme is"], "max_tokens": 3})
    mapped = s.call("layer_map", {})
    assert mapped["layers"] == 4
    found = s.call("find_direction", {"name": "loops", "pairs": PAIRS})
    assert set(found["mlp_writes_by_layer"]) == {0, 1, 2, 3}
    proposed = s.call("propose_removal", {"name": "fix", "direction": "loops", "layers": [1, 2], "strength": 2.0, "neurons": 8, "test_pairs": PAIRS})
    assert proposed["columns_changed"] == 16
    tested = s.call("test", {"name": "fix", "strengths": [0.5, 1.0]})
    assert [t["strength"] for t in tested["tried"]] == [0.5, 1.0] and all(t["verdict"] in ("better", "worse", "mixed", "same") for t in tested["tried"])
    isolated = s.call("isolate", {"name": "fix"})
    assert [p["part"] for p in isolated["parts"]][-1] == "random edit of the same size" and len(isolated["parts"]) == 3
    assert_same(model, saved)
    kept = s.call("keep", {"name": "fix", "strength": 1.0})
    assert kept["sha256_before"] != kept["sha256_after"]
    assert not torch.equal(writer_module(model, 1).weight.data, saved["model.layers.1.mlp.down_proj.weight"])
    s.call("undo", {})
    assert_same(model, saved)
    assert s.passes_used == 1 + 1 + 1 + 3 + 4 and s.log_is_intact()
    assert all(p.grad is None for p in model.parameters())
    assert len(open(tmp_path / "log.jsonl", encoding="utf-8").read().splitlines()) == len(s.log)


def test_a_fact_is_written_without_gradients_and_checked(model, tokenizer):
    saved = snapshot(model)
    s = surgery(model, tokenizer)
    proposed = s.call("propose_fact", {"name": "rx", "fact": FACT, "layer": 2, "strength": 1.0})
    assert proposed["rank"] >= 2 and proposed["answer_tokens"][0]["token"]
    tested = s.call("test", {"name": "rx", "strengths": [1.0]})
    assert "recall" in tested["tried"][0]["scores"]
    assert_same(model, saved)
    assert all(p.grad is None for p in model.parameters())


def test_a_fact_test_counts_leaks_against_its_first_measurement(model, tokenizer, monkeypatch):
    import cramschool.surgery as module

    references = []
    real = module.monitor

    def spy(*args, reference=None, **kwargs):
        references.append(reference)
        return real(*args, reference=reference, **kwargs)

    monkeypatch.setattr(module, "monitor", spy)
    s = surgery(model, tokenizer)
    s.call("propose_fact", {"name": "rx", "fact": FACT, "layer": 2, "strength": 1.0})
    s.call("test", {"name": "rx", "strengths": [0.5, 1.0]})
    assert len(references) == 3 and references[0] is None  # the unedited model first, then each strength against it
    assert references[1] is references[2] is not None and references[1].answers[0]["other_subject"]


def test_protecting_names_leaves_more_alone(model, tokenizer):
    s = surgery(model, tokenizer)
    person = dict(FACT, target="Julian Rodlauer")
    plain = s.call("propose_fact", {"name": "a", "fact": person, "layer": 2})
    protected = s.call("propose_fact", {"name": "b", "fact": person, "layer": 2, "protect_names": True})
    assert protected["left_alone"] > plain["left_alone"] and protected["protect_names"] and not plain["protect_names"]
    tested = s.call("test", {"name": "b", "strengths": [1.0]})
    assert "other_names" in tested["tried"][0]["scores"]


def test_a_fact_test_says_how_many_prompts_it_rests_on_and_warns_when_too_few(model, tokenizer):
    s = surgery(model, tokenizer)
    s.call("propose_fact", {"name": "few", "fact": FACT, "layer": 2})
    few = s.call("test", {"name": "few", "strengths": [1.0]})
    assert few["prompts"]["recall"] == 1 and "at least 10" in few["warning"]
    many = dict(FACT, new_prompts=[f"record {i} says the owner of acme is" for i in range(10)])
    s.call("propose_fact", {"name": "many", "fact": many, "layer": 2})
    enough = s.call("test", {"name": "many", "strengths": [1.0]})
    assert enough["prompts"]["recall"] == 10 and "warning" not in enough


def test_a_bad_fact_is_an_error_not_a_crash(model, tokenizer):
    s = surgery(model, tokenizer)
    result = s.call("propose_fact", {"name": "bad", "fact": dict(FACT, prompts=["the owner is"]), "layer": 2})
    assert "error" in result and s.passes_used == 0


def test_the_budget_is_enforced(model, tokenizer):
    s = surgery(model, tokenizer, budget=2)
    s.call("find_direction", {"name": "loops", "pairs": PAIRS})
    s.call("propose_removal", {"name": "fix", "direction": "loops", "layers": [1], "strength": 1.0, "test_pairs": PAIRS})
    result = s.call("test", {"name": "fix", "strengths": [1.0, 2.0]})
    assert "Not enough forward passes" in result["error"]
    assert s.passes_used == 1


def test_the_log_detects_tampering(model, tokenizer):
    s = surgery(model, tokenizer)
    s.call("status", {})
    s.call("layer_map", {})
    assert s.log_is_intact()
    s.log[0]["result"]["passes_used"] = 99
    assert not s.log_is_intact()


def test_the_log_file_alone_proves_it_is_intact(model, tokenizer, tmp_path):
    from cramschool.surgery import log_file_is_intact

    path = tmp_path / "log.jsonl"
    s = surgery(model, tokenizer, log_path=str(path))
    s.call("layer_map", {})  # its results are keyed by layer number, which a file holds as strings
    s.call("find_direction", {"name": "loops", "pairs": PAIRS})
    assert s.log_is_intact() and log_file_is_intact(str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert '"layers": 4' in lines[0]
    path.write_text("\n".join([lines[0].replace('"layers": 4', '"layers": 5')] + lines[1:]) + "\n", encoding="utf-8")
    assert not log_file_is_intact(str(path))


def test_logs_hashed_before_entries_went_through_json_still_check():
    import json

    from cramschool.surgery import _digest, chain_is_intact

    body = dict(step=1, tool="layer_map", args={}, result={"shares": {0: 0.1, 10: 0.5, 2: 0.2}}, previous="start")
    old = dict(json.loads(json.dumps(body)), hash=_digest(body))  # hashed with integer keys, read back from the file with string keys
    assert _digest({k: v for k, v in old.items() if k != "hash"}) != old["hash"] and chain_is_intact([old])
    old["result"]["shares"]["10"] = 0.9
    assert not chain_is_intact([old])


def test_calls_run_one_at_a_time(model, tokenizer):
    """An editor may send calls in parallel. A test applies its edit for a moment, so calls must never overlap."""
    import threading
    import time

    s = surgery(model, tokenizer)
    inside, overlapped = [], []

    def slow():
        overlapped.append(bool(inside))
        inside.append(1)
        time.sleep(0.05)
        inside.pop()
        return {}

    s.tools["status"].run = slow
    threads = [threading.Thread(target=s.call, args=("status", {})) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlapped == [False] * 4 and len(s.log) == 4 and s.log_is_intact()


def test_a_thinking_loop_is_reported_with_how_the_thinking_ends(model, tokenizer, monkeypatch):
    import cramschool.problems as problems

    loop = tokenizer("the owner of acme is", add_special_tokens=False).input_ids * 12
    monkeypatch.setattr(problems, "think", lambda *a, **k: [loop])
    answer = surgery(model, tokenizer).call("generate", {"questions": ["who owns acme ?"], "thinking": True, "max_tokens": 99})["answers"][0]
    assert answer["looping"] and answer["repeats"] > 0.5 and not answer["finished"] and answer["tokens"] == 60
    assert answer["thinking_start"].startswith("the owner") and answer["thinking_end"].endswith("acme is")


def test_thinking_answers_on_a_real_generation(model, tokenizer):
    answer = surgery(model, tokenizer).call("generate", {"questions": ["who owns acme ?"], "thinking": True, "max_tokens": 20})["answers"][0]
    assert answer["tokens"] <= 20 and {"finished", "final_answer", "looping", "repeats", "thinking_start", "thinking_end"} <= set(answer)


def test_confirm_runs_paired_problems(model, tokenizer):
    s = surgery(model, tokenizer)
    s.call("find_direction", {"name": "loops", "pairs": PAIRS})
    s.call("propose_removal", {"name": "fix", "direction": "loops", "layers": [1], "strength": 1.0, "test_pairs": PAIRS})
    result = s.call("confirm", {"name": "fix", "problems": [dict(question="what is 1 + 1 ?", answer="2")], "max_tokens": 4})
    assert result["problems"] == 1 and set(result["correct"]) == {"rescued", "broken", "p_value"}


# ---------------------------------------------------------------- Claude as the editor, with a stand-in client
class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self.message


class FakeClient:
    """Replays prepared assistant turns and records every request, so the loop can be checked without the API."""

    def __init__(self, turns):
        self.turns, self.requests = list(turns), []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        return FakeStream(self.turns.pop(0))


def tool_use(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def text(t):
    return SimpleNamespace(type="text", text=t)


def test_claude_loop_runs_tools_and_returns_the_final_report(model, tokenizer):
    s = surgery(model, tokenizer)
    client = FakeClient([
        SimpleNamespace(stop_reason="tool_use", content=[text("Mapping first."), tool_use("t1", "layer_map", {}), tool_use("t2", "status", {})]),
        SimpleNamespace(stop_reason="end_turn", content=[text("Nothing to fix: no edit was made.")]),
    ])
    seen = []
    run = run_with_claude(s, "Diagnose the model.", client=client, on_call=lambda n, i, o: seen.append(n))
    assert run.final_text == "Nothing to fix: no edit was made." and run.stop_reason == "end_turn" and run.turns == 2
    assert seen == ["layer_map", "status"] and s.passes_used == 1
    first, second = client.requests
    assert first["model"] == "claude-opus-5-5" and first["system"] == PROTOCOL and all(t["eager_input_streaming"] for t in first["tools"])
    results = second["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"] and not any(r["is_error"] for r in results)


def test_claude_loop_answers_bad_input_with_an_error_and_continues(model, tokenizer):
    s = surgery(model, tokenizer)
    client = FakeClient([
        SimpleNamespace(stop_reason="tool_use", content=[tool_use("t1", "test", {"name": "missing", "strengths": "1"})]),
        SimpleNamespace(stop_reason="end_turn", content=[text("done")]),
    ])
    run = run_with_claude(s, "go", client=client)
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] and "INVALID_INPUT" in result["content"]
    assert run.final_text == "done" and s.passes_used == 0


def test_claude_loop_stops_on_refusal_and_resends_paused_turns(model, tokenizer):
    s = surgery(model, tokenizer)
    client = FakeClient([
        SimpleNamespace(stop_reason="pause_turn", content=[text("thinking")]),
        SimpleNamespace(stop_reason="refusal", content=[text("I can't help with that.")]),
    ])
    run = run_with_claude(s, "go", client=client)
    assert run.stop_reason == "refusal" and len(client.requests) == 2
    assert client.requests[1]["messages"][-1]["role"] == "assistant"


def test_claude_loop_refuses_to_run_a_truncated_tool_call(model, tokenizer):
    s = surgery(model, tokenizer)
    client = FakeClient([SimpleNamespace(stop_reason="max_tokens", content=[tool_use("t1", "status", {})])])
    with pytest.raises(RuntimeError, match="max_tokens"):
        run_with_claude(s, "go", client=client)
    assert s.log == []

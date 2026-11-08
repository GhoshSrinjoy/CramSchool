"""Editor models without a network: an OpenAI-compatible server on localhost, a stand-in Anthropic client, and a scripted local model."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from cramschool.editors import draft_relation, editor_from_spec, run_editor
from cramschool.editors.anthropic import FALLBACK_BETA, AnthropicEditor
from cramschool.editors.base import first_json_object
from cramschool.editors.local import LocalEditor
from cramschool.editors.openai_compat import OpenAICompatEditor


class Box:
    """A toolbox with one tool."""
    protocol = "Use the tools."

    def __init__(self):
        self.calls = []

    def tool_definitions(self):
        return [dict(name="status", description="the state", input_schema={"type": "object", "properties": {"verbose": {"type": "boolean"}}, "required": []})]

    def validate(self, name, args):
        return None if name == "status" else f"Unknown tool {name!r}."

    def call(self, name, args):
        self.calls.append((name, args))
        return dict(facts=3)


def serve(replies):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(dict(path=self.path, body=body, auth=self.headers.get("Authorization")))
            out = json.dumps(replies.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_an_openai_compatible_endpoint_drives_the_tools():
    tool_turn = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "status", "arguments": "{\"verbose\": true}"}}]}}]}
    final = {"choices": [{"finish_reason": "stop", "message": {"content": "Three facts are stored."}}]}
    server, seen = serve([tool_turn, final])
    try:
        editor = OpenAICompatEditor("qwen3:14b", base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="k")
        box = Box()
        run = run_editor(editor, box, "How many facts?")
    finally:
        server.shutdown()
    assert run.final_text == "Three facts are stored." and run.stop_reason == "end" and box.calls == [("status", {"verbose": True})]
    first, second = seen[0]["body"], seen[1]["body"]
    assert seen[0]["path"] == "/v1/chat/completions" and seen[0]["auth"] == "Bearer k"
    assert first["tools"][0]["function"]["name"] == "status" and first["messages"][0] == {"role": "system", "content": "Use the tools."}
    assert second["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"facts": 3})}


def test_editor_specs(monkeypatch):
    monkeypatch.delenv("CRAMSCHOOL_BASE_URL", raising=False)
    e = editor_from_spec("ollama:qwen3:14b")
    assert e.model == "qwen3:14b" and e.base_url == "http://localhost:11434/v1" and e.name == "ollama:qwen3:14b"
    e = editor_from_spec("vllm:my-model@http://gpu-box:8000/v1")
    assert e.model == "my-model" and e.base_url == "http://gpu-box:8000/v1"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CRAMSCHOOL_API_KEY", raising=False)
    with pytest.raises(ValueError):
        editor_from_spec("openai:some-model")
    with pytest.raises(ValueError):
        editor_from_spec("nonsense:model")
    monkeypatch.setenv("CRAMSCHOOL_EDITOR", "lmstudio:local-model")
    assert editor_from_spec().base_url == "http://localhost:1234/v1"


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


class FakeAnthropic:
    def __init__(self, turns):
        self.turns, self.requests = list(turns), []
        stream = lambda **kw: (self.requests.append(kw), FakeStream(self.turns.pop(0)))[1]  # noqa: E731
        self.messages = SimpleNamespace(stream=stream)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=stream))


def test_claude_streams_keeps_its_content_and_asks_for_the_fallback():
    client = FakeAnthropic([
        SimpleNamespace(stop_reason="tool_use", content=[SimpleNamespace(type="text", text="Checking."), SimpleNamespace(type="tool_use", id="t1", name="status", input={})]),
        SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Done.")]),
    ])
    box = Box()
    run = run_editor(AnthropicEditor(client=client), box, "Look.")
    assert run.final_text == "Done." and box.calls == [("status", {})]
    first, second = client.requests
    assert first["model"] == "claude-opus-5-5" and first["betas"] == [FALLBACK_BETA] and first["extra_body"] == {"fallbacks": "default"}
    assert first["output_config"] == {"effort": "high"} and first["tools"][0]["eager_input_streaming"] is True
    assert second["messages"][1]["role"] == "assistant" and second["messages"][2]["content"][0]["type"] == "tool_result"
    off = FakeAnthropic([SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok")])])
    AnthropicEditor(client=off, fallbacks=None).complete("hi")
    assert "betas" not in off.requests[0]


class Scripted(LocalEditor):
    def __init__(self, outputs):
        self.outputs, self.seen, self.name = list(outputs), [], "local:scripted"
        self.enable_thinking = False

    def generate(self, messages):
        self.seen.append([dict(m) for m in messages])
        return self.outputs.pop(0)


def test_a_local_model_calls_tools_through_json():
    editor = Scripted(['I will look. {"tool": "status", "arguments": {"verbose": false}}', '<think>{"tool": "nope"}</think>{"final": "Three facts."}'])
    box = Box()
    run = run_editor(editor, box, "How many facts?")
    assert run.final_text == "Three facts." and box.calls == [("status", {"verbose": False})]
    assert "Tools:" in editor.seen[0][0]["content"] and editor.seen[1][-1]["content"].startswith("Result of status:")


def test_first_json_object_and_drafting():
    assert first_json_object('noise {"a": {"b": "}"}} {"c": 1}') == {"a": {"b": "}"}}
    assert first_json_object("no json here") is None
    reply = json.dumps({"plain": [f"Line {i} of {{s}} is" for i in range(9)] + ["no slot"], "chat": [f"Who leads {{s}} ({i})?" for i in range(5)]})
    rel = draft_relation(Scripted([reply]), "Chief Scientist", "the chief scientist of a company")
    assert rel.name == "chief_scientist" and len(rel.plain) == 9 and len(rel.chat) == 5

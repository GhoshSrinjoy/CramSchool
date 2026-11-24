"""The MCP server, through the SDK's in-memory client, on a tiny random model. Skipped when the `mcp` package is not installed."""
import json

import pytest
import torch

mcp = pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from cramschool import GateConfig, Surgery, WeightPatch, writer_module  # noqa: E402
from cramschool.mcp_server import build_server  # noqa: E402

OPEN = GateConfig(min_recall=0.0, max_leak=1.0, max_cross_leak=1.0, max_garbled=1.0, max_known_drop=1.0, max_kl=1e9, min_retention=0.0)
PAIRS = [
    dict(start="question : what is the name ? answer :", good=" the company is here", bad=" is is is is is"),
    dict(start="who owns acme ? answer :", good=" julian rodlauer owns it", bad=" the the the the the"),
]


def payload(result):
    if result.structured_content is not None:
        return result.structured_content.get("result", result.structured_content)
    return json.loads(result.content[0].text)


@pytest.fixture()
def server(model, tokenizer):
    surgery = Surgery(model, tokenizer, gates=OPEN, budget=30)
    return build_server(lambda: surgery), surgery


@pytest.mark.anyio
async def test_the_server_lists_every_surgery_tool(server):
    built, surgery = server
    async with Client(built, raise_exceptions=True) as client:
        listed = await client.list_tools()
        names = {t.name for t in listed.tools}
    assert names == {f"cramschool_{n}" for n in surgery.tools}
    assert "There are no gradients" in (built.instructions or "")


@pytest.mark.anyio
async def test_a_session_through_the_server_edits_exports_and_undoes(server, model, tmp_path):
    built, surgery = server
    saved = {n: p.data.clone() for n, p in model.named_parameters()}
    async with Client(built, raise_exceptions=True) as client:
        found = payload(await client.call_tool("cramschool_find_direction", {"name": "loops", "pairs": PAIRS}))
        assert set(found["mlp_writes_by_layer"]) == {"0", "1", "2", "3"} or set(found["mlp_writes_by_layer"]) == {0, 1, 2, 3}
        await client.call_tool("cramschool_propose_removal", {"name": "fix", "direction": "loops", "layers": [1, 2], "strength": 2.0, "neurons": 8, "test_pairs": PAIRS})
        tested = payload(await client.call_tool("cramschool_test", {"name": "fix", "strengths": [1.0]}))
        assert tested["tried"][0]["verdict"] in ("better", "worse", "mixed", "same")
        await client.call_tool("cramschool_keep", {"name": "fix", "strength": 1.0})
        edited = {n: p.data.clone() for n, p in model.named_parameters()}
        exported = payload(await client.call_tool("cramschool_export_patch", {"path": str(tmp_path / "fix"), "note": "thinking off"}))
        assert exported["tensors"] == 2
        await client.call_tool("cramschool_undo", {})
    assert all(torch.equal(p.data, saved[n]) for n, p in model.named_parameters())
    # The exported patch replays the kept edit exactly on the same base, and refuses any other.
    WeightPatch.load(str(tmp_path / "fix")).apply(model, strict=True)
    assert all(torch.equal(p.data, edited[n]) for n, p in model.named_parameters())
    assert surgery.log_is_intact()


@pytest.mark.anyio
async def test_errors_reach_the_editor_as_tool_errors(server):
    built, surgery = server
    async with Client(built, raise_exceptions=False) as client:
        result = await client.call_tool("cramschool_test", {"name": "nothing", "strengths": [1.0]})
    assert result.is_error
    assert "No proposal named" in result.content[0].text and "forward passes left" in result.content[0].text
    assert surgery.passes_used == 0


@pytest.mark.anyio
async def test_a_shell_command_can_call_the_tools(server):
    from cramschool.remote import call_tool

    built, surgery = server
    ok, result = await call_tool(built, "status", {})
    assert ok and result["passes_left"] == 30
    ok, error = await call_tool(built, "cramschool_test", {"name": "nothing", "strengths": [1.0]})
    assert not ok and "No proposal named" in error and surgery.passes_used == 0


def test_the_call_command_reads_arguments_prints_saves_and_explains_failures(monkeypatch, tmp_path, capsys):
    from cramschool import remote
    from cramschool.__main__ import main

    (tmp_path / "args.json").write_text('{"name": "fix", "strengths": [0.5]}', encoding="utf-8")
    assert remote.parse_arguments("@" + str(tmp_path / "args.json")) == {"name": "fix", "strengths": [0.5]}
    assert remote.parse_arguments("") == {} and remote.tool_name("status") == remote.tool_name("cramschool_status") == "cramschool_status"
    with pytest.raises(ValueError):
        remote.parse_arguments("[1, 2]")

    seen = []

    async def answers(url, tool, args):
        seen.append((url, tool, args))
        return True, {"passes_left": 5, "text": "Zürich"}

    monkeypatch.setattr(remote, "call_tool", answers)
    assert main(["call", "status", "{}", "--url", "http://127.0.0.1:1/mcp", "--out", str(tmp_path / "out.json")]) == 0
    assert seen == [("http://127.0.0.1:1/mcp", "status", {})] and '"passes_left": 5' in capsys.readouterr().out
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["text"] == "Zürich"

    async def refuses(url, tool, args):
        return False, "No proposal named 'x'"

    monkeypatch.setattr(remote, "call_tool", refuses)
    assert main(["call", "keep", '{"name": "x"}']) == 1 and "No proposal named 'x'" in capsys.readouterr().err

    groups = remote._group_types()  # the built-in exception group, or its backport on Python 3.10; none there without the backport

    async def unreachable(url, tool, args):
        raise groups[0]("task group", [ConnectionRefusedError("refused")]) if groups else ConnectionRefusedError("refused")

    monkeypatch.setattr(remote, "call_tool", unreachable)
    assert main(["call", "status"]) == 3 and "ConnectionRefusedError: refused" in capsys.readouterr().err
    assert main(["call", "status", "{not json"]) == 2


def test_serve_over_http_loads_the_model_first_and_stays_on_this_computer(monkeypatch, model, tokenizer, tmp_path):
    import transformers
    from mcp.server.mcpserver import MCPServer

    from cramschool import live
    from cramschool.mcp_server import serve

    loads, runs = [], []

    class Loader:
        def __init__(self, made):
            self.made = made

        def from_pretrained(self, path, **kwargs):
            loads.append(path)
            return self.made

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", Loader(model))
    monkeypatch.setattr(transformers, "AutoTokenizer", Loader(tokenizer))
    monkeypatch.setattr(MCPServer, "run", lambda self, transport="stdio", **kwargs: runs.append(dict(kwargs, transport=transport)))

    serve("tiny", None, 10, None, log_dir=str(tmp_path / "stdio"), transport="stdio")
    assert runs[-1] == {"transport": "stdio"} and loads == []  # over stdio the model waits for the first call

    serve("tiny", None, 10, None, log_dir=str(tmp_path / "http"), transport="http", port=9123)
    assert runs[-1] == {"transport": "streamable-http", "host": "127.0.0.1", "port": 9123} and loads == ["tiny", "tiny"]
    status = live.read_status(live.status_path_for(live.newest_session(str(tmp_path / "http"))))
    assert status["model"] == "tiny" and status["current"] is None and status["budget"] == 10
    with pytest.raises(ValueError):
        serve("tiny", None, 10, None, transport="sse")


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def test_a_long_call_tells_its_client_that_it_is_still_running(monkeypatch):
    """A client may give up on a call that is silent for long (Claude Code after 300 seconds), and a build or an export runs for
    minutes. Every call runs in a worker thread and sends a note every BEAT seconds, with the latest line of progress."""
    import time

    import anyio
    from mcp import Client

    from cramschool import mcp_server
    from cramschool.tools import GraftTools

    class Slow(GraftTools):
        def call(self, name, args=None):
            self.said = "building a fact"
            time.sleep(0.6)
            return dict(facts=0, tool=name)

    monkeypatch.setattr(mcp_server, "BEAT", 0.1)
    server = mcp_server.build_server(lambda: None, Slow(None))
    notes = []

    async def heard(progress, total, message):
        notes.append((progress, message))

    async def check():
        async with Client(server) as client:
            result = await client.call_tool("cramschool_graft_status", {}, progress_callback=heard)
            assert not result.is_error and '"tool": "status"' in "".join(getattr(part, "text", "") for part in result.content)
            quiet = await client.call_tool("cramschool_graft_status", {})      # a client that asks for no notes gets its result all the same
            assert not quiet.is_error

    anyio.run(check)
    assert len(notes) >= 2 and notes[0] == (1, "building a fact") and [n for n, _ in notes] == sorted(n for n, _ in notes)


def test_calls_of_one_server_never_run_at_the_same_time_whichever_toolbox_they_belong_to(monkeypatch):
    """A client sends calls in parallel, and every tool works on the one loaded model. A fact tool and a weight-edit tool, or two
    calls that set the model's weights for a moment, must not overlap: the server runs its calls one after another."""
    import threading
    import time

    import anyio
    from mcp import Client

    from cramschool import mcp_server
    from cramschool.tools import GraftTools

    inside, most, order = [], [0], []
    watch = threading.Lock()

    def work(name):
        with watch:
            inside.append(name)
            most[0] = max(most[0], len(inside))
        time.sleep(0.25)
        with watch:
            inside.remove(name)
            order.append(name)
        return dict(done=name)

    class Facts(GraftTools):
        def call(self, name, args=None):
            return work("fact tool")

    class Edits:
        def call(self, name, args=None):
            return work("weight-edit tool")

    monkeypatch.setattr(mcp_server, "BEAT", 0.05)
    server = mcp_server.build_server(lambda: Edits(), Facts(None))

    async def check():
        async with Client(server) as client:
            async with anyio.create_task_group() as group:           # four calls sent together, two of each toolbox
                for tool in ("cramschool_graft_status", "cramschool_status", "cramschool_graft_list_facts", "cramschool_status"):
                    group.start_soon(client.call_tool, tool, {})

    anyio.run(check)
    assert most[0] == 1 and len(order) == 4 and set(order) == {"fact tool", "weight-edit tool"}

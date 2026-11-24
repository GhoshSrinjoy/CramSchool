"""docs/mcp.md's tool reference is what the server's own tool list gives: a tool that is added, renamed or given another argument
fails this test until the page is made again (python scripts/mcp_reference.py --write)."""
import importlib.util
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def reference():
    spec = importlib.util.spec_from_file_location("mcp_reference", os.path.join(ROOT, "scripts", "mcp_reference.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_mcp_page_lists_every_tool_the_server_registers_with_its_own_arguments():
    pytest.importorskip("mcp")
    ref = reference()
    with open(os.path.join(ROOT, "docs", "mcp.md"), encoding="utf-8") as fh:
        page = fh.read()
    assert ref.START in page and ref.END in page
    held = page[page.index(ref.START):page.index(ref.END) + len(ref.END)]
    assert held == ref.render(), "docs/mcp.md is behind the server's tools: run `python scripts/mcp_reference.py --write`"

    served = set(ref.listed())
    assert len(served) == sum(len(group) for _, _, group in ref.GROUPS) and f"{len(served)} tools." in held
    for name in served:                                         # each tool is a row of its own, by its name without the prefix
        assert f"| `{name[len(ref.PREFIX):]}`<br>" in held
    named = set(re.findall(r"\bcramschool_(?:graft_)?[a-z_]+[a-z]", page)) - {"cramschool_mcp"}
    assert named <= served, f"the page names tools the server doesn't have: {sorted(named - served)}"


def test_the_reference_refuses_a_tool_it_does_not_describe(monkeypatch):
    pytest.importorskip("mcp")
    ref = reference()
    monkeypatch.delitem(ref.TOOLS, "graft_weights_merge")
    with pytest.raises(SystemExit, match="not described here: graft_weights_merge"):
        ref.render()
    monkeypatch.setitem(ref.TOOLS, "graft_weights_merge", ("x", "y"))
    monkeypatch.setitem(ref.TOOLS, "graft_gone", ("x", "y"))
    with pytest.raises(SystemExit, match="described and not served: graft_gone"):
        ref.render()

"""Call the surgery tools on a running `cramschool serve --transport http` server, from any shell or script.

    cramschool serve --model models/Qwen3.5-0.8B --transport http --log-dir sessions --watch
    cramschool call status
    cramschool call find_direction @pairs.json --out found.json

The server keeps the model loaded between calls, so an editor that can only run commands can still work one call at a time, choosing each
step from the last result. This module needs only the standard library and `mcp`, not the rest of the package or PyTorch, so it can run as
a script and start quickly: `python -I remote.py status`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

DEFAULT_PORT = 8765
DEFAULT_URL = f"http://127.0.0.1:{DEFAULT_PORT}/mcp"
PREFIX = "cramschool_"


def tool_name(tool: str) -> str:
    """`status` and `cramschool_status` both name the status tool."""
    return tool if tool.startswith(PREFIX) else PREFIX + tool


def parse_arguments(text: str) -> dict:
    """A tool's arguments: a JSON object given inline, or @path to a UTF-8 file holding one."""
    if text.startswith("@"):
        with open(text[1:], encoding="utf-8") as fh:
            text = fh.read()
    value = json.loads(text.strip() or "{}")
    if not isinstance(value, dict):
        raise ValueError('the arguments must be a JSON object, such as {"name": "fix"}')
    return value


def _text(result) -> str:
    return "".join(getattr(part, "text", "") for part in result.content)


def _payload(result) -> Any:
    if result.structured_content is not None:
        return result.structured_content.get("result", result.structured_content)
    try:
        return json.loads(_text(result))
    except ValueError:
        return _text(result)


async def call_tool(server: Any, tool: str, args: dict) -> tuple[bool, Any]:
    """(True, result) or (False, the error text). server: the endpoint's URL, or an MCP server object in the same process.

    A URL gets no read timeout: a confirmation can run for many minutes before it answers, and the progress window shows it meanwhile.
    """
    from mcp import Client

    if isinstance(server, str):
        import httpx2
        from mcp.client.streamable_http import streamable_http_client

        async with httpx2.AsyncClient(timeout=httpx2.Timeout(30.0, read=None)) as http:
            async with Client(streamable_http_client(server, http_client=http)) as client:
                result = await client.call_tool(tool_name(tool), args)
    else:
        async with Client(server) as client:
            result = await client.call_tool(tool_name(tool), args)
    return (False, _text(result)) if result.is_error else (True, _payload(result))


def _group_types() -> tuple:
    """The exception-group classes a task group may raise: the built-in one (Python 3.11 on), and on Python 3.10 the backport that the
    async libraries use there."""
    try:
        return (BaseExceptionGroup,)  # noqa: F821 (a builtin from Python 3.11)
    except NameError:
        try:
            from exceptiongroup import BaseExceptionGroup as backport
        except ImportError:
            return ()
        return (backport,)


def _innermost(error: BaseException) -> BaseException:
    """The first real error inside the exception groups that task groups wrap errors in."""
    groups = _group_types()
    while isinstance(error, groups) and error.exceptions:
        error = error.exceptions[0]
    return error


def run(url: str, tool: str, args_text: str = "{}", out: str | None = None) -> int:
    """Call one tool and print its result as JSON. Exit codes: 0 done, 1 the tool refused or failed, 2 bad arguments, 3 no server."""
    try:
        args = parse_arguments(args_text)
    except (OSError, ValueError) as error:
        print(f"cramschool call: {error}", file=sys.stderr)
        return 2
    try:
        ok, result = asyncio.run(call_tool(url, tool, args))
    except Exception as error:  # the server is not running, or it stopped during the call
        inner = _innermost(error)
        print(f"cramschool call: no answer from {url} ({type(inner).__name__}: {inner}). Is `cramschool serve --transport http` running?", file=sys.stderr)
        return 3
    if not ok:  # the refusal is also given as JSON, on stdout and in --out, so that an editor reads it as it reads a result
        refusal = json.dumps(dict(error=str(result), tool=tool_name(tool)), indent=1, ensure_ascii=False)
        if out:
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(refusal + "\n")
        print(refusal)
        print(f"cramschool call: {tool_name(tool)} failed: {result}", file=sys.stderr)
        return 1
    text = json.dumps(result, indent=1, ensure_ascii=False)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    try:
        sys.stdout.reconfigure(errors="replace")  # a console that cannot show a character gets "?" rather than a crash
    except (AttributeError, ValueError):
        pass
    print(text)
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tool", help="the tool, with or without the cramschool_ prefix, such as status or find_direction")
    parser.add_argument("args", nargs="?", default="{}", help="its arguments: a JSON object, or @file.json")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"the server's endpoint (default {DEFAULT_URL})")
    parser.add_argument("--out", help="also save the result to this JSON file, in UTF-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cramschool call", description="Call one surgery tool on a running cramschool HTTP server.")
    add_arguments(parser)
    args = parser.parse_args(argv)
    return run(args.url, args.tool, args.args, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

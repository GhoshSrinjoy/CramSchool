"""Talk to `python -m weightspace serve` over stdio, as Claude Code does: list the tools, then make two real calls on Qwen3.5-0.8B.

Usage: python mcp_stdio_check.py <python> <package_src> <calibration.txt> <log.jsonl>
"""
import asyncio
import json
import os
import sys
import time

from mcp import Client, StdioServerParameters

python, src, calibration, log_path = sys.argv[1:5]


async def main():
    env = dict(os.environ, PYTHONPATH=src, USE_TF="0", HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    params = StdioServerParameters(
        command=python, args=["-m", "weightspace", "serve", "--model", r"D:\data\models\Qwen__Qwen3.5-0.8B", "--calibration", calibration, "--budget", "10", "--log", log_path], env=env
    )
    started = time.time()
    async with Client(params, raise_exceptions=False, read_timeout_seconds=600) as client:
        print(f"connected in {time.time() - started:.1f}s; server instructions start: {client.instructions[:80]!r}" if client.instructions else "connected (no instructions)", flush=True)
        tools = await client.list_tools()
        print(f"{len(tools.tools)} tools: {', '.join(t.name for t in tools.tools)}", flush=True)
        t0 = time.time()
        mapped = await client.call_tool("weightspace_layer_map", {})
        data = mapped.structured_content.get("result", mapped.structured_content) if mapped.structured_content else json.loads(mapped.content[0].text)
        print(f"layer_map (first call loads the model) in {time.time() - t0:.0f}s: first layer with answers {data['first_layer_with_answers']} of {data['layers']}, passes left {data['passes_left']}", flush=True)
        asked = await client.call_tool("weightspace_generate", {"questions": ["What is the capital of France?"], "max_tokens": 16})
        data = asked.structured_content.get("result", asked.structured_content) if asked.structured_content else json.loads(asked.content[0].text)
        print(f"generate: {data['answers'][0]['answer']!r}", flush=True)
        bad = await client.call_tool("weightspace_keep", {"name": "nothing"})
        print(f"a bad call returns is_error={bad.is_error}: {bad.content[0].text[:100]!r}", flush=True)


asyncio.run(main())
print("DONE", flush=True)

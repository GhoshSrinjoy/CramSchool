"""Run the installed `weightspace serve --watch` over stdio, as Claude Code does, and check that the progress window opens without
disturbing the protocol: every call must come back intact while the window draws.

Usage: python mcp_watch_check.py <python> <calibration.txt> <sessions folder>
"""
import asyncio
import json
import os
import sys
import time

from mcp import Client, StdioServerParameters

from weightspace import live

python, calibration, sessions = sys.argv[1:4]
PAIRS = [
    dict(start="<|im_start|>user\nWhat is 17 + 25?<|im_end|>\n<|im_start|>assistant\n<think>\n", good="17 + 25: 17 + 20 = 37, then 37 + 5 = 42. So the answer is 42.",
         bad="17 + 25. Wait, 17 + 25. Wait, 17 + 25. Wait, 17 + 25. Wait, 17 + 25. Wait, 17 + 25."),
    dict(start="<|im_start|>user\nHow many legs do 3 spiders have?<|im_end|>\n<|im_start|>assistant\n<think>\n", good="A spider has 8 legs, so 3 spiders have 3 x 8 = 24 legs.",
         bad="A spider has 8 legs. Hmm, let me check. A spider has 8 legs. Hmm, let me check. A spider has 8 legs."),
]


def data(result):
    return result.structured_content.get("result", result.structured_content) if result.structured_content else json.loads(result.content[0].text)


def visible_windows(title: str) -> list[str]:
    """Titles of visible top-level windows containing `title`: proof the window is on the desktop, not just a process."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if title in buf.value:
                found.append(buf.value)
        return True

    user32.EnumWindows(each, 0)
    return found


async def main():
    env = dict(os.environ, USE_TF="0", HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    args = ["-m", "weightspace", "serve", "--model", r"D:\data\models\Qwen__Qwen3.5-0.8B", "--calibration", calibration, "--budget", "12", "--log-dir", sessions, "--watch"]
    started = time.time()
    async with Client(StdioServerParameters(command=python, args=args, env=env), raise_exceptions=False, read_timeout_seconds=900) as client:
        tools = await client.list_tools()
        print(f"connected in {time.time() - started:.1f}s, {len(tools.tools)} tools", flush=True)
        t0 = time.time()
        mapped = data(await client.call_tool("weightspace_layer_map", {}))
        print(f"layer_map (opens the window, loads the model) in {time.time() - t0:.0f}s: answers first at layer {mapped['first_layer_with_answers']} of {mapped['layers']}", flush=True)
        print(f"during the session: window process running {live.viewer_running(sessions)}, visible windows {visible_windows('weightspace surgery')}", flush=True)
        t0 = time.time()
        asked = data(await client.call_tool("weightspace_generate", {"questions": ["What is 6 times 7?", "Name the largest planet."], "max_tokens": 384, "thinking": True}))
        print(f"generate with thinking on in {time.time() - t0:.0f}s: finished {[a['finished'] for a in asked['answers']]}", flush=True)
        found = data(await client.call_tool("weightspace_find_direction", {"name": "loops", "pairs": PAIRS}))
        print(f"find_direction: strongest writers {found['strongest_writers']}", flush=True)
        bad = await client.call_tool("weightspace_keep", {"name": "nothing"})
        print(f"a bad call is an error: {bad.is_error}", flush=True)
        status = data(await client.call_tool("weightspace_status", {}))
        print(f"status: {status['passes_used']} passes used, {status['passes_left']} left", flush=True)

    log = live.newest_session(sessions)
    entries = live.read_log(log)
    state = live.read_status(live.status_path_for(log))
    print(f"log {log}: {len(entries)} steps, every one summarised: {all('summary' in e for e in entries)}; status current={state['current']}", flush=True)
    print(f"progress window running: {live.viewer_running(sessions)}", flush=True)
    print(live.render(state, entries, width=150), flush=True)


asyncio.run(main())
print("DONE", flush=True)

"""A live view of a surgery in its own terminal window: budget and step progress bars, what the current step is doing, GPU memory, kept
edits, and a readable log of every step.

`Surgery` writes `<log>.status.json` beside its log as it works. The viewer only reads, so it cannot disturb a session.

    weightspace watch                  follow the newest session in ~/.weightspace/sessions
    weightspace watch <folder>         follow the newest session in a folder, and pick up the next one when it starts
    weightspace watch <log.jsonl>      follow one session

`weightspace serve --watch` opens the viewer by itself when the first tool call arrives, so a session run by Claude Code, which keeps its
MCP servers out of sight, can still be watched. This module uses only the standard library, so the viewer opens without loading PyTorch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

WIDTH = 28
STATUS_SUFFIX = ".status.json"
VIEWER_PID = "viewer.pid"
INDENT = " " * 9
COLORS = {"better": "32", "KEPT": "32", "worse": "31", "error": "31", "stopped": "31", "mixed": "33", "UNDID": "33"}


# ---------------------------------------------------------------------- where sessions live
def default_sessions_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".weightspace", "sessions")


def new_log_path(folder: str) -> str:
    """A fresh log for one server run, so two sessions never share a log."""
    return os.path.join(os.path.abspath(folder), f"surgery_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl")


def status_path_for(log_path: str) -> str:
    return log_path + STATUS_SUFFIX


def newest_session(folder: str) -> str | None:
    """The log of the session in `folder` whose status changed last."""
    try:
        names = [n for n in os.listdir(folder) if n.endswith(".jsonl" + STATUS_SUFFIX)]
    except OSError:
        return None

    def changed(name: str) -> float:
        try:
            return os.path.getmtime(os.path.join(folder, name))
        except OSError:
            return 0.0

    return os.path.join(folder, max(names, key=changed)[: -len(STATUS_SUFFIX)]) if names else None


# ---------------------------------------------------------------------- the status file
def write_status(path: str, status: dict) -> None:
    """Best effort. On Windows a viewer reading the file at that moment can block the replace for a few milliseconds, and progress must
    never break a tool call, so this retries briefly and then gives up quietly."""
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(status, fh, ensure_ascii=False, default=str)
        for _ in range(25):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.01)
    except OSError:
        pass


def read_status(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_log(log_path: str) -> list[dict]:
    """Every complete entry; a line still being written is skipped and read on the next frame."""
    out = []
    try:
        with open(log_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_uint32()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # on Windows signal 0 would be Ctrl+C, hence the branch above
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------- readable lines
def bar(fraction: float, width: int = WIDTH) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(width * fraction))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def clock(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}h{seconds // 60 % 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def _pct(value) -> str:
    return f"{100 * value:.0f}%" if isinstance(value, (int, float)) else "?"


def _passes(n: int) -> str:
    return "1 pass" if n == 1 else f"{n} passes"


def summarize(tool: str, args: dict, result: dict) -> str:
    """One readable line for a finished tool call."""
    if "error" in result:
        return f"{tool}: error - {str(result['error'])[:160]}"
    if tool == "generate":
        answers = result.get("answers", [])
        first = answers[0] if answers else {}
        if result.get("thinking"):
            finished, looping = sum(bool(a.get("finished")) for a in answers), sum(bool(a.get("looping")) for a in answers)
            return f"generate (thinking on): {len(answers)} answer(s), {finished} finished thinking, {looping} looping; first: {str(first.get('final_answer'))[:70]!r}"
        return f"generate (thinking off): {len(answers)} answer(s); first: {str(first.get('answer'))[:90]!r}"
    if tool == "layer_map":
        return f"layer_map: known answers first appear at layer {result.get('first_layer_with_answers')} of {result.get('layers')}"
    if tool == "layer_lens":
        return f"layer_lens: read {len(result.get('top_tokens', {}))} layers"
    if tool == "find_direction":
        return f"find_direction {result.get('direction')!r}: strongest writers {result.get('strongest_writers')}"
    if tool == "propose_removal":
        return f"propose_removal {result.get('proposed')!r}: layers {result.get('layers')}, strength {result.get('strength')}, {result.get('columns_changed')} columns"
    if tool == "propose_fact":
        tokens = result.get("answer_tokens", [])
        alone = f", {result['left_alone']} reads left alone" if "left_alone" in result else ""
        return f"propose_fact {result.get('proposed')!r}: layer {result.get('layer')}, {len(tokens)} answer tokens, rank {result.get('rank')}{alone}" + (", names protected" if result.get("protect_names") else "")
    if tool == "test":
        parts = []
        for t in result.get("tried", []):
            changes = t.get("changes", {})
            key = "bad" if "bad" in changes else "recall"
            parts.append(f"x{t.get('strength')} {t.get('verdict')} ({key} {changes.get(key, 0):+.3f})")
        return f"test {result.get('proposal')!r}: " + ", ".join(parts)
    if tool == "isolate":
        return f"isolate {result.get('proposal')!r}: " + ", ".join(f"{p['part'].split('.mlp')[0].replace('model.', '')}: {p['verdict']}" for p in result.get("parts", []))
    if tool == "confirm":
        before, after = result.get("unedited", {}), result.get("edited", {})
        correct = result.get("correct", {})
        return (f"confirm {result.get('proposal')!r} on {result.get('problems')} problems: correct {_pct(before.get('correct'))} -> {_pct(after.get('correct'))} "
                f"(rescued {correct.get('rescued')}, broken {correct.get('broken')}), repeated tail {_pct(before.get('repeats'))} -> {_pct(after.get('repeats'))}")
    if tool == "keep":
        return f"KEPT {result.get('kept')!r} at strength {result.get('strength')}: {len(result.get('sha256_after', {}))} tensor(s) changed"
    if tool == "undo":
        return f"UNDID {result.get('undone')!r}"
    if tool == "export_patch":
        return f"exported {result.get('edits')} to {result.get('saved')}"
    if tool == "status":
        return f"status: {result.get('passes_used')} passes used, kept {result.get('kept')}"
    return f"{tool}: done"


def gpu_line() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        used, total, busy = (float(x) for x in out.strip().splitlines()[0].split(","))
        return f"GPU      {bar(used / total, 16)} {used / 1024:.1f} of {total / 1024:.1f} GB in use, busy {busy:.0f}%"
    except Exception:
        return "GPU      (nvidia-smi not available)"


def paint(line: str, color: bool) -> str:
    if not color:
        return line
    return re.sub(r"\b(" + "|".join(COLORS) + r")\b", lambda m: f"\x1b[{COLORS[m.group(1)]}m{m.group(1)}\x1b[0m", line)


def _entry_time(entry: dict) -> float | None:
    """When a log entry was written, as a timestamp; None if it has no readable time."""
    try:
        return time.mktime(time.strptime(entry.get("time", ""), "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError, OverflowError):
        return None


def render(status: dict | None, entries: list[dict], *, now: float | None = None, width: int = 110, height: int | None = None,
           color: bool = False, source: str | None = None, gpu: str | None = None) -> str:
    """The whole view as text, fitted to `width` columns and, if given, `height` rows: the oldest log lines go first."""
    now = now or time.time()
    head = [f"weightspace surgery   {time.strftime('%H:%M:%S', time.localtime(now))}", "=" * min(width - 1, 100)]
    if not status:
        where = f" in {source}" if source else ""
        body = [f"Waiting for a session{where}.", "Ask Claude Code to teach the model something. This window follows the session as soon as the first tool runs."]
        return _fit(head + [""] + body, width, color)

    ended = bool(status.get("pid")) and not alive(int(status["pid"]))
    used, budget = status.get("passes_used", 0), max(1, status.get("budget", 1))
    kept = status.get("kept", [])
    started = status.get("started_at", now)
    if ended:  # how long it ran, up to its last logged step, rather than a clock that keeps counting
        last = _entry_time(entries[-1]) if entries else None
        span = f"ran {clock(max(0.0, (last or now) - started))}, ended"
    else:
        span = f"running {clock(now - started)}"
    top = [
        f"Model    {status.get('model', '?')}{' on ' + status['device'] if status.get('device') else ''}   {span}",
        f"Log      {status.get('log', '?')}",
        f"Budget   {bar(used / budget)}  {used} of {budget} forward passes used",
        f"Kept     {', '.join(kept) if kept else 'nothing yet: the weights are unchanged'}",
        gpu if gpu is not None else gpu_line(),
        "",
    ]
    current = status.get("current")
    if current and ended:
        now_lines = [f"Now      stopped during {current['tool']}: {current.get('detail', '')}. The server is gone; the log below is complete."]
    elif ended:
        now_lines = ["Now      session ended. Start another one and this window will follow it."]
    elif current:
        fraction, elapsed = current.get("fraction", 0.0), now - current.get("started_at", now)
        left = f", about {clock(elapsed * (1 - fraction) / fraction)} left" if 0.05 <= fraction < 1 else ""
        now_lines = [f"Now      {current['tool']}  {bar(fraction)} {100 * fraction:3.0f}%   {clock(elapsed)}{left}", f"{INDENT}{current.get('detail', '')}"]
        now_lines += [f"{INDENT}- {note}" for note in current.get("notes", [])[-6:]]
    else:
        now_lines = ["Now      idle, waiting for the editor's next step"]

    lines = head + top + now_lines + ["", "Steps    (newest last)"]
    rows = []
    for entry in entries:
        line = entry.get("summary") or summarize(entry.get("tool", "?"), entry.get("args", {}), entry.get("result", {}))
        at = str(entry.get("time", ""))[11:19]
        cost = entry.get("passes", 0)
        rows.append(f"  #{entry.get('step', '?'):<3} {at + '  ' if at else ''}{line}   ({entry.get('result', {}).get('seconds', 0)}s{', ' + _passes(cost) if cost else ''})")
    if height is not None:
        room = max(3, height - len(lines) - 1)
        if len(rows) > room:
            rows = [f"  ... {len(rows) - room + 1} earlier steps are in the log file"] + rows[-(room - 1):]
    lines += rows or ["  (no steps yet)"]
    return _fit(lines, width, color)


def _fit(lines: list[str], width: int, color: bool) -> str:
    return "\n".join(paint(line if len(line) < width else line[: width - 4] + "...", color) for line in lines)


# ---------------------------------------------------------------------- the viewer
def _enable_vt() -> bool:
    """Turn on escape sequences, so the view redraws in place instead of clearing and flickering."""
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        kernel32.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel32.SetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        return bool(kernel32.GetConsoleMode(handle, ctypes.byref(mode))) and bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _set_title(title: str) -> None:
    if os.name == "nt":
        try:
            import ctypes

            ctypes.WinDLL("kernel32").SetConsoleTitleW(title)
        except Exception:
            pass
    elif sys.stdout.isatty():
        sys.stdout.write(f"\x1b]0;{title}\x07")


def viewer_running(folder: str) -> bool:
    try:
        with open(os.path.join(folder, VIEWER_PID), encoding="utf-8") as fh:
            pid = int(fh.read().strip())
    except (OSError, ValueError):
        return False
    return alive(pid)


def watch(path: str | None = None, once: bool = False, interval: float = 1.0) -> None:
    """Redraw until Ctrl+C. `path` is a session log, or a folder whose newest session is followed."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    folder = path is None or os.path.isdir(path)
    source = os.path.abspath(path or default_sessions_dir())
    claimed = False
    if not once:
        _set_title("weightspace surgery")
        if folder:
            os.makedirs(source, exist_ok=True)
            with open(os.path.join(source, VIEWER_PID), "w", encoding="utf-8") as fh:
                fh.write(str(os.getpid()))
            claimed = True
    vt = False if once else _enable_vt()
    gpu, gpu_at = None, 0.0
    try:
        if vt:
            sys.stdout.write("\x1b[2J")
        while True:
            log = newest_session(source) if folder else source
            status = read_status(status_path_for(log)) if log else None
            if time.time() - gpu_at > 2:
                gpu, gpu_at = gpu_line(), time.time()
            size = shutil.get_terminal_size((110, 40))
            text = render(status, read_log(log) if log else [], width=size.columns, height=None if once else size.lines - 1, color=vt, source=source if folder else None, gpu=gpu)
            if once:
                print(text, flush=True)
                return
            if vt:
                sys.stdout.write("\x1b[H" + "\n".join(line + "\x1b[K" for line in text.split("\n")) + "\x1b[J")
                sys.stdout.flush()
            else:
                os.system("cls" if os.name == "nt" else "clear")
                print(text, flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        if claimed:
            try:
                os.remove(os.path.join(source, VIEWER_PID))
            except OSError:
                pass


def viewer_command(folder: str) -> list[str]:
    """This file run as a script, in isolated mode: it needs nothing beyond the standard library."""
    return [sys.executable, "-I", os.path.abspath(__file__), folder]


def open_viewer(folder: str) -> subprocess.Popen | None:
    """Open the viewer on `folder` in a new terminal window, unless one is already following it.

    On Windows no standard handle is passed: the window gets its own console's. Passing any would hand the viewer the other handles of
    this process, and this process's stdout carries the MCP protocol.
    """
    if viewer_running(folder):
        return None
    command = viewer_command(folder)
    if os.name == "nt":
        breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        for flags in (subprocess.CREATE_NEW_CONSOLE | breakaway, subprocess.CREATE_NEW_CONSOLE):
            try:
                return subprocess.Popen(command, creationflags=flags, close_fds=True)
            except OSError:
                continue
        return None
    quiet = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
    if sys.platform == "darwin":
        script = " ".join("'" + part.replace("'", "'\\''") + "'" for part in command)
        return subprocess.Popen(["osascript", "-e", f'tell application "Terminal" to do script "{script}"'], **quiet)
    for terminal in (["x-terminal-emulator", "-e"], ["gnome-terminal", "--"], ["konsole", "-e"], ["xterm", "-e"]):
        try:
            return subprocess.Popen(terminal + command, **quiet)
        except FileNotFoundError:
            continue
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="weightspace watch", description="Show a surgery's progress, live.")
    parser.add_argument("path", nargs="?", help="a session log, or a folder to follow the newest session in (default ~/.weightspace/sessions)")
    parser.add_argument("--once", action="store_true", help="print the view once and exit")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between redraws")
    args = parser.parse_args(argv)
    watch(args.path, once=args.once, interval=args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

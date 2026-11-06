"""Live progress for every run in this folder, the GPU, and any model download, in one terminal view.

    python watch.py            refresh every 2 seconds until you press Ctrl+C
    python watch.py --once     print once

Runs report through progress*.json files written next to this script. Nothing here changes a run.
"""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS = {r"D:\data\models\Qwen__Qwen3.5-4B": 9.34e9}
WIDTH = 30


def bar(fraction, width=WIDTH):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(width * fraction))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def clock(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}h{seconds // 60 % 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def gpu_line():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        used, total, util = (float(x) for x in out.strip().split(","))
        return f"GPU  memory {bar(used / total, 20)} {used / 1024:.1f}/{total / 1024:.1f} GB   busy {util:.0f}%"
    except Exception:
        return "GPU  (nvidia-smi not available)"


def folder_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def run_block(path):
    try:
        with open(path, encoding="utf-8") as fh:
            s = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return [f"{os.path.basename(path)}: (being written)"]
    now = time.time()
    finished = s.get("finished_at")
    elapsed = (finished or now) - s["started_at"]
    fraction = s["work_done"] / s["work_total"] if s.get("work_total") else 0.0
    left = elapsed * (1 - fraction) / fraction if fraction > 0 and not finished else None
    age = now - os.path.getmtime(path)
    state = "FAILED" if s.get("error") else "finished" if finished else ("quiet for " + clock(age)) if age > 300 else "running"
    name = os.path.basename(path)[: -len(".json")]
    lines = [f"{name:<16} {bar(fraction)} {100 * fraction:3.0f}%  {state}, {clock(elapsed)} elapsed" + (f", about {clock(left)} left" if left else "")]
    for st in s.get("stages", []):
        done = st["done"] >= st["total"]
        mark = "done" if done else "now " if st["name"] == s.get("stage") and not finished else "    "
        lines.append(f"   {mark} {st['name'][:44]:<44} {bar(st['done'] / st['total'] if st['total'] else 0, 12)} {st['done']}/{st['total']}")
    if not finished:
        lines.append(f"   now: {s.get('detail', '')[:110]}")
    for result in s.get("results", [])[-3:]:
        lines.append(f"   > {result[:150]}")
    if s.get("error"):
        lines.append(f"   ERROR: {s['error'][:150]}")
    return lines


def render():
    lines = [f"weightspace runs   {time.strftime('%H:%M:%S')}", gpu_line()]
    for path, expected in DOWNLOADS.items():
        if os.path.isdir(path):
            size = folder_size(path)
            lines.append(f"Download {os.path.basename(path)} {bar(size / expected, 20)} {size / 1e9:.1f}/{expected / 1e9:.1f} GB")
    files = sorted(glob.glob(os.path.join(HERE, "progress*.json")), key=os.path.getmtime, reverse=True)
    recent = [f for f in files if time.time() - os.path.getmtime(f) < 6 * 3600][:4]
    if not recent:
        lines.append("\nNo runs in the last six hours.")
    for path in recent:
        lines.append("")
        lines += run_block(path)
    return "\n".join(lines)


def main():
    once = "--once" in sys.argv
    try:
        while True:
            text = render()
            if not once:
                os.system("cls" if os.name == "nt" else "clear")
            print(text, flush=True)
            if once:
                return
            time.sleep(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

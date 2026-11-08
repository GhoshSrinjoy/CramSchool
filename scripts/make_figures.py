"""Draw every chart and animation in docs/assets from the saved result files. Nothing here runs a model.

  journey.png           facts passing every check, method by method, in the order the methods were built
  blind.png             the blind benches, per method
  tradeoff.png          median recall against facts failing a leak check, on the development bench bench40b
  failures.png          how the 40 facts of blind bench40e fare under each method
  graft-trace.gif       GRAFT token by token, from docs/assets/data/graft_trace.json (recorded on Qwen3.5-0.8B)
  graft-trace-<n>.png   the last frame of each scene of the animation, for readers that don't play GIFs

Usage: python scripts/make_figures.py
"""
from __future__ import annotations

import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "research", "results")
ASSETS = os.path.join(ROOT, "docs", "assets")

INK, MUTED, FAINT, GRID, WHITE, PANEL = "#0F172A", "#475569", "#94A3B8", "#E2E8F0", "#FFFFFF", "#F8FAFC"
GREEN, GREEN_L = "#16A34A", "#DCFCE7"
AMBER, AMBER_L = "#D97706", "#FEF3C7"
ROSE, ROSE_L = "#E11D48", "#FFE4E6"
BLUE, BLUE_L = "#2563EB", "#DBEAFE"
VIOLET, VIOLET_L = "#7C3AED", "#EDE9FE"
SLATE, SLATE_L = "#64748B", "#F1F5F9"

_names = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Segoe UI", "Inter", "Helvetica Neue", "Arial") if f in _names), "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlecolor": INK, "figure.facecolor": WHITE, "axes.facecolor": WHITE})


def rows(bench: str, arm: str) -> list[dict]:
    with open(os.path.join(RESULTS, bench, f"{arm}.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def passes(bench: str, arm: str) -> int:
    return sum(bool(r["passed"]) for r in rows(bench, arm))


LEAKS = ("same", "lookalike", "other", "names", "cross", "sibling")


def leak_failures(bench: str, arm: str) -> int:
    return sum(any(f.split()[0] in LEAKS for f in r["failed"]) for r in rows(bench, arm))


def recall_median(bench: str, arm: str) -> float:
    return statistics.median(r["measures"]["recall"] for r in rows(bench, arm))


def clean(ax, grid_axis="x"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def title(fig, text, sub):
    fig.text(0.02, 0.965, text, fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, sub, fontsize=10.5, color=MUTED, va="top")


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=160, facecolor=WHITE)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------- charts

STAGE = dict(beta=(SLATE, SLATE_L, "Beta: weight writes and gates"), block=(AMBER, AMBER_L, "Stage 1: fact block, one fact at a time"),
             graft=(GREEN, GREEN_L, "v1: GRAFT, many facts in one layer"))


def journey():
    items = [  # (label, bench, arm, stage, blind)
        ("Weight write, last-token key", "bench", "linear", "beta", False),
        ("+ name tag and chat teaching", "bench", "tagged_chat", "beta", False),
        ("Gated first answer token", "bench", "gated_first_v3", "beta", False),
        ("Split gate", "bench40b", "split_gate_teach", "beta", False),
        ("The fact written in the prompt", "bench40b", "in_context", "beta", False),
        ("Fact block v1", "bench40b", "fact_block_teach", "block", False),
        ("Fact block v2", "bench40b", "fact_block_v2_teach", "block", False),
        ("Fact block v2", "bench40c", "fact_block_v2_teach", "block", True),
        ("GRAFT, 100 facts loaded", "bench40b", "fact_table_teach", "graft", False),
        ("GRAFT, 140 facts loaded", "bench40d", "fact_table_teach", "graft", True),
        ("GRAFT, 180 facts loaded", "bench40e", "fact_table_teach", "graft", True),
    ]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.76, bottom=0.08)
    title(fig, "Facts passing every check, of 40", "Each bar is one method on one 40-fact bench, in the order the methods were built. "
          "Solid bars are blind benches,\nwritten by an agent that never saw the code; pale bars are development benches.")
    ys = list(range(len(items)))[::-1]
    for y, (label, bench, arm, stage, blind) in zip(ys, items):
        n = passes(bench, arm)
        edge, fill, _ = STAGE[stage]
        ax.barh(y, max(n, 0.15), height=0.62, color=edge if blind else fill, edgecolor=edge, linewidth=1.4)
        ax.text(n + 0.6, y, f"{n}", va="center", ha="left", fontsize=11, fontweight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc=WHITE, ec="none"))
        ax.text(-0.8, y, f"{label}", va="center", ha="right", fontsize=10.5, color=INK)
        ax.text(-0.8, y - 0.36, f"{'bench40' if bench == 'bench' else bench}{'  ·  blind' if blind else ''}", va="center", ha="right", fontsize=8.5, color=GREEN if blind else FAINT,
                fontweight="bold" if blind else "normal")
    ax.axvline(30, color=GREEN, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(30.4, max(ys) + 0.55, "v1 bar: 30", color=GREEN, fontsize=9.5, va="center")
    ax.set_xlim(0, 40)
    ax.set_ylim(-0.7, max(ys) + 0.8)
    ax.set_yticks([])
    ax.set_xticks([0, 10, 20, 30, 40])
    clean(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE[s][0]) for s in ("beta", "block", "graft")]
    ax.legend(handles, [STAGE[s][2] for s in ("beta", "block", "graft")], loc="lower left", bbox_to_anchor=(-0.42, 1.0), ncol=3, frameon=False,
              fontsize=9.5, handlelength=1.2, columnspacing=1.4)
    save(fig, "journey.png")


def blind():
    groups = [
        ("bench40c", "fact block only", [("Fact block v2", "fact_block_v2_teach", AMBER, "")]),
        ("bench40d", "GRAFT: 140 facts loaded", [("Fact block v2", "fact_block_v2_teach", AMBER, ""), ("GRAFT", "fact_table_teach", GREEN, "")]),
        ("bench40e", "GRAFT: 180 facts loaded", [("Fact block v2", "fact_block_v2_teach", AMBER, ""), ("GRAFT", "fact_table_teach", GREEN, ""),
                                          ("GRAFT + looser start", "fact_table_rev_teach", GREEN_L, "//")]),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.17)
    title(fig, "Blind benches: facts passing every check, of 40", "Fact block v2 holds one fact at a time; GRAFT holds every loaded fact "
          "in one layer.\nbench40d chose the looser start, so its blind result is bench40e only.")
    x, ticks, labels = 0.0, [], []
    for bench, loaded, bars in groups:
        start = x
        for name, arm, color, hatch in bars:
            n = passes(bench, arm)
            ax.bar(x, n, width=0.8, color=color, edgecolor=GREEN if color == GREEN_L else color, hatch=hatch, linewidth=1.2)
            ax.text(x, n + 0.6, str(n), ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK,
                    bbox=dict(boxstyle="round,pad=0.12", fc=WHITE, ec="none"))
            ax.text(x, -1.6, name, ha="center", va="top", fontsize=9.5, color=INK)
            x += 1
        ticks.append((start + x - 1) / 2)
        labels.append(f"{bench}\n{loaded}")
        x += 0.8
    ax.axhline(30, color=GREEN, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(0.9, 30.5, "v1 bar: 30", color=GREEN, fontsize=9.5, ha="center", va="bottom")
    ax.set_ylim(0, 40)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=10.5, color=INK, fontweight="bold")
    ax.tick_params(axis="x", pad=26)
    ax.set_yticks([0, 10, 20, 30, 40])
    clean(ax, grid_axis="y")
    save(fig, "blind.png")


def tradeoff():
    points = [  # label, arm, color, label offset (dx, dy)
        ("Weight write + name tag\n+ chat teaching", "tagged_chat_teach", SLATE, (-1.0, 0.035)),
        ("Fact in the prompt", "in_context", SLATE, (-1.0, -0.075)),
        ("Gated first token", "gated_first_v2_teach", BLUE, (1.0, -0.03)),
        ("Split gate", "split_gate_teach", BLUE, (1.0, -0.045)),
        ("Fact block v1", "fact_block_teach", AMBER, (1.0, 0.02)),
        ("Fact block v2", "fact_block_v2_teach", AMBER, (1.0, 0.025)),
        ("GRAFT (100 facts loaded)", "fact_table_teach", GREEN, (1.2, -0.04)),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.12)
    title(fig, "Recall against leaks, on the development bench", "bench40b, 40 facts. Up is better recall; left is fewer facts that "
          "leak their answer where it does not belong\n(other questions about the company, look-alike or unrelated companies, name slots).")
    ax.add_patch(plt.Rectangle((-1, 0.75), 9, 0.27, color=GREEN_L, alpha=0.55, zorder=0))
    ax.text(0.2, 1.005, "the goal: high recall, few leaks", color=GREEN, fontsize=9.5, va="top")
    for label, arm, color, (dx, dy) in points:
        xv, yv = leak_failures("bench40b", arm), recall_median("bench40b", arm)
        big = arm in ("fact_table_teach", "fact_block_v2_teach")
        ax.scatter(xv, yv, s=170 if big else 90, color=color, edgecolor=WHITE, linewidth=1.5, zorder=3)
        ax.text(xv + dx, yv + dy, f"{label}  ({passes('bench40b', arm)} pass)", fontsize=9.5, color=INK,
                ha="left" if dx > 0 else "right", va="center", fontweight="bold" if big else "normal")
    ax.set_xlim(-1, 41)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("facts failing a leak check, of 40")
    ax.set_ylabel("median recall on new wordings")
    clean(ax, grid_axis="both")
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(GRID)
    save(fig, "tradeoff.png")


def failures():
    arms = [("Fact block v2\none fact at a time", "fact_block_v2_teach"), ("GRAFT\n180 facts loaded", "fact_table_teach"),
            ("GRAFT + looser start\n180 facts loaded", "fact_table_rev_teach")]
    kinds = [("passes every check", GREEN), ("recall too low", AMBER), ("same-company leak", ROSE), ("another fact's answer", VIOLET),
             ("more than one", SLATE)]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.70, bottom=0.22)
    title(fig, "Blind bench40e: what happens to each of the 40 facts", "One bar per method; each segment counts facts by the check "
          "they fail. Nothing failed on look-alike companies,\nunrelated companies, name slots, known facts or drift.")
    for y, (label, arm) in enumerate(arms[::-1]):
        counts = dict.fromkeys(k for k, _ in kinds)
        counts = {k: 0 for k in counts}
        for r in rows("bench40e", arm):
            got = sorted({f.split()[0] for f in r["failed"]})
            key = ("passes every check" if not got else "more than one" if len(got) > 1 else
                   {"recall": "recall too low", "same": "same-company leak", "cross": "another fact's answer"}.get(got[0], "more than one"))
            counts[key] += 1
        left = 0
        for k, color in kinds:
            n = counts[k]
            if n:
                ax.barh(y, n, left=left, color=color, height=0.6, edgecolor=WHITE, linewidth=1.5)
                if n >= 2:
                    ax.text(left + n / 2, y, str(n), ha="center", va="center", color=WHITE, fontsize=11, fontweight="bold")
                left += n
        ax.text(-0.8, y, label, ha="right", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 40)
    ax.set_yticks([])
    ax.set_xticks([0, 10, 20, 30, 40])
    clean(ax)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in kinds], [k for k, _ in kinds], loc="upper center",
              bbox_to_anchor=(0.42, -0.14), ncol=5, frameon=False, fontsize=9.5)
    save(fig, "failures.png")


# ---------------------------------------------------------------- the animation

W, H, DPI = 1100, 640, 100
RELATION_NAMES = dict(owner="owner", ceo="CEO", founder="founder", hq="head office")


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
        self.renderer = self.fig.canvas.get_renderer()

    def width(self, text, size, bold=False):
        w, _, _ = self.renderer.get_text_width_height_descent(text or " ", FontProperties(family=FONT, size=size, weight="bold" if bold else "normal"), ismath=False)
        return w / W

    def box(self, x, y, w, h, fc, ec, lw=1.2, r=0.006):
        self.fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=self.fig.transFigure,
                                               facecolor=fc, edgecolor=ec, linewidth=lw))

    def text(self, x, y, s, size=11, color=INK, bold=False, ha="left", va="center"):
        self.fig.text(x, y, s, fontsize=size, color=color, fontweight="bold" if bold else "normal", ha=ha, va=va)

    def frame(self) -> Image.Image:
        self.fig.canvas.draw()
        img = Image.frombuffer("RGBA", self.fig.canvas.get_width_height(), self.fig.canvas.buffer_rgba()).convert("RGB")
        self.fig.clf()
        self.fig.patches.clear()
        return img


def shown(token: str) -> tuple[str, bool]:
    """A token as drawn: the leading space becomes a gap, and line breaks become an arrow."""
    gap = token.startswith(" ")
    t = token.strip(" ").replace("\n", "↵")
    return (t if t else "·"), gap


def tokens_row(cv, x0, y, tokens, n, style, size=11.5, max_x=0.97, line=0.062):
    """Draw the first n tokens as boxes from (x0, y), wrapping at max_x. style(i) -> (fill, edge, text color)."""
    x, yy = x0, y
    for i, tok in enumerate(tokens[:n]):
        t, gap = shown(tok)
        w = cv.width(t, size) + 0.012
        if gap and i:
            x += 0.007
        if x + w > max_x:
            x, yy = x0, yy - line
        fc, ec, tc = style(i)
        cv.box(x, yy - 0.021, w, 0.042, fc, ec, lw=1.1, r=0.008)
        cv.text(x + w / 2, yy, t, size=size, color=tc, ha="center")
        x += w + 0.002
    return yy


def scene_frames(cv, sc, number, total):
    q, found, gate = sc["question_tokens"], sc["found"], sc["gate"]
    bars = []
    if gate:
        for f in gate["facts"]:
            bars.append((f"{RELATION_NAMES[f['relation']]}  →  {f['target']}", "stored fact", f["z"]))
        for rel, z in gate["shared"].items():
            bars.append((RELATION_NAMES[rel], "no fact stored", z))
    fired = bool(sc["pushed"] and sc["pushed"][0])
    # the company's tokens: from where its name starts to where the index first reports it
    hit = next((i for i, f in enumerate(found) if f is not None), None)
    name_from = None
    if hit is not None:
        target = " ".join(found[hit].split())
        text = ""
        for j in range(hit, -1, -1):
            text = q[j] + text
            if target in " ".join(text.lower().split()):
                name_from = j
                break
    n_q, n_r = len(q), max(len(sc["base"]), len(sc["graft"]))
    plan = [("q", i) for i in range(1, n_q + 1)] + [("hold", 0)] * 4 + [("key1", 0)] * 5 + [("bars", k) for k in range(1, 9)] + \
        [("verdict", 0)] * 7 + [("reply", i) for i in range(1, n_r + 1)] + [("end", 0)] * 28
    order = ["q", "hold", "key1", "bars", "verdict", "reply", "end"]
    for step, k in plan:
        stage = order.index(step)
        nq = k if step == "q" else n_q
        cv.box(0.012, 0.02, 0.976, 0.96, WHITE, GRID, lw=1.0, r=0.012)
        cv.text(0.035, 0.925, "GRAFT, token by token", size=17, bold=True)
        cv.text(0.035, 0.877, sc["title"], size=11.5, color=MUTED)
        chip = f"scene {number} of {total}"
        cw = cv.width(chip, 10) + 0.03
        cv.box(0.965 - cw, 0.905, cw, 0.042, SLATE_L, GRID, r=0.01)
        cv.text(0.965 - cw / 2, 0.926, chip, size=10, color=MUTED, ha="center")
        # the question, with the company's tokens lit once the whole name has arrived
        cv.text(0.035, 0.80, "User", size=10.5, color=MUTED, bold=True)
        lit = name_from is not None and nq - 1 >= hit
        tokens_row(cv, 0.09, 0.80, q, nq, lambda i: (GREEN_L, GREEN, INK) if lit and name_from <= i <= hit else (SLATE_L, GRID, INK))
        y = 0.70
        if stage >= 2:
            cv.text(0.035, y, "Key 1", size=10.5, color=MUTED, bold=True)
            if hit is not None:
                cv.text(0.10, y, f"a stored company is named:  {found[hit].title()}", size=11.5, color=GREEN, bold=True)
                cv.text(0.10, y - 0.04, "matched as a whole name, in any capitalisation", size=9.5, color=FAINT)
            else:
                cv.text(0.10, y, "no stored company is named", size=11.5, color=MUTED, bold=True)
                cv.text(0.10, y - 0.04, "Nimbrosco is one letter from Nimbresco; the index matches whole names only, "
                        "so a look-alike never opens the layer", size=9.5, color=FAINT)
        y = 0.585
        if stage >= 3 and bars:
            grow = min(1.0, (k if step == "bars" else 8) / 8)
            cv.text(0.035, y, "Key 2", size=10.5, color=MUTED, bold=True)
            cv.text(0.10, y, "which question? each detector's reading at the prompt's last token", size=11, color=INK)
            x_mid, scale, top, gap = 0.72, 0.10, y - 0.06, 0.047
            for j, (label, kind, z) in enumerate(bars):
                yy = top - j * gap
                cv.text(0.10, yy, label, size=10.5, color=INK)
                cv.text(0.43, yy, kind, size=9.5, color=FAINT)
                zz = max(-2.0, min(2.0, z)) * grow
                color = (GREEN if z > 0 else FAINT) if kind == "stored fact" else (AMBER if z > 0 else SLATE_L)
                edge = color if color != SLATE_L else GRID
                x0, w = (x_mid, zz * scale) if zz >= 0 else (x_mid + zz * scale, -zz * scale)
                cv.box(x0, yy - 0.014, max(w, 0.002), 0.028, color, edge, lw=0.6, r=0.004)
                if grow >= 1:
                    cv.text(x_mid + zz * scale + (0.01 if zz >= 0 else -0.01), yy, f"{z:+.2f}", size=9.5, color=MUTED,
                            ha="left" if zz >= 0 else "right")
            ybot = top - (len(bars) - 1) * gap - 0.025
            cv.fig.add_artist(plt.Line2D([x_mid, x_mid], [ybot, top + 0.03], transform=cv.fig.transFigure, color=INK, linewidth=1.2))
            cv.text(x_mid, top + 0.045, "threshold", size=9, color=MUTED, ha="center")
            y = ybot - 0.045
        else:
            y = 0.585 if not bars else y
        if stage >= 4:
            if fired:
                v, c = "Both keys turn: the answer's first token is pushed at layer 20, then each next token while it matches.", GREEN
            elif bars:
                v, c = "Key 2 stays shut, so the reply is the model's own.", MUTED
            else:
                v, c = "Key 1 stays shut, so the reply is the model's own.", MUTED
            cv.text(0.035, y, v, size=11.5, color=c, bold=True)
        if stage >= 5:
            nr = k if step == "reply" else n_r
            y0 = y - 0.085
            cv.text(0.035, y0 + 0.012, "Without", size=10, color=MUTED, bold=True)
            cv.text(0.035, y0 - 0.020, "GRAFT", size=10, color=MUTED, bold=True)
            tokens_row(cv, 0.12, y0, sc["base"], min(nr, len(sc["base"])), lambda i: (WHITE, GRID, MUTED), size=10.5)
            y1 = y0 - 0.085
            cv.text(0.035, y1 + 0.012, "With", size=10, color=GREEN, bold=True)
            cv.text(0.035, y1 - 0.020, "GRAFT", size=10, color=GREEN, bold=True)
            pushed = sc["pushed"]
            tokens_row(cv, 0.12, y1, sc["graft"], min(nr, len(sc["graft"])),
                       lambda i: (GREEN_L, GREEN, INK) if i < len(pushed) and pushed[i] else (WHITE, GRID, INK), size=10.5)
        cv.text(0.035, 0.045, "Qwen3.5-0.8B  ·  180 facts in one inserted layer  ·  greedy decoding  ·  real outputs  ·  "
                "green reply tokens were pushed by GRAFT", size=9, color=FAINT)
        yield cv.frame()


def animation():
    with open(os.path.join(ASSETS, "data", "graft_trace.json"), encoding="utf-8") as fh:
        trace = json.load(fh)
    cv = Canvas()
    frames, ends = [], []
    for n, sc in enumerate(trace["scenes"], 1):
        fs = list(scene_frames(cv, sc, n, len(trace["scenes"])))
        frames += fs
        ends.append(fs[-1])
    durations = [90] * len(frames)
    path = os.path.join(ASSETS, "graft-trace.gif")
    pal = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print("wrote", os.path.relpath(path, ROOT), f"({len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB)")
    for n, img in enumerate(ends, 1):
        p = os.path.join(ASSETS, f"graft-trace-{n}.png")
        img.save(p, optimize=True)
        print("wrote", os.path.relpath(p, ROOT))
    plt.close(cv.fig)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    journey()
    blind()
    tradeoff()
    failures()
    animation()


if __name__ == "__main__":
    main()

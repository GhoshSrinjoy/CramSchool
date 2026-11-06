"""Charts of the experiments, saved as PNG files with a CSV table of the same numbers beside each.

Usage: python plots.py <results_dir> <plots_dir>

Colours follow a validated categorical order (blue, orange, aqua) on a light surface; text stays in ink colours, never series colours.
Each chart has one y-axis. The CSV beside each PNG is the table view of the same data.
"""
import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

results_dir, plots_dir = sys.argv[1], sys.argv[2]
os.makedirs(plots_dir, exist_ok=True)

SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4, in this order, never cycled
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})
made = []


def load(name):
    path = os.path.join(results_dir, name)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def finish(fig, name, header, rows, note=None):
    if note:
        # Below everything the axes draw, so it can never sit on top of tick or axis labels; the tight bounding box makes room for it.
        fig.canvas.draw()
        lowest = min(ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y0 for ax in fig.axes)
        fig.text(0.01, lowest - 0.03, note, color=MUTED, fontsize=8, ha="left", va="top")
    fig.savefig(os.path.join(plots_dir, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(plots_dir, name + ".csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    made.append(name)


def grouped_bars(ax, groups, series_names, values, percent=True, colors=None, legend=True):
    """values[series][group]. Thin bars with a 2px surface gap between neighbours. `colors` keeps each entity's colour the same in every panel."""
    count = len(series_names)
    width = 0.8 / count
    for i, name in enumerate(series_names):
        xs = [g + (i - (count - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(xs, values[i], width=width, color=(colors or SERIES)[i], edgecolor=SURFACE, linewidth=2, label=name)
        for bar, value in zip(bars, values[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.0f}%" if percent else f"{value:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(groups)), groups)
    ax.tick_params(axis="x", colors=INK_2)
    if legend:
        ax.legend(frameon=False, loc="upper right", fontsize=9)


# One colour per method, the same in every chart: colour follows the entity, never its position.
METHOD_COLORS = {"default backprop edit": SERIES[0], "best backprop edit": SERIES[1], "no backprop": SERIES[2]}


# ---------------------------------------------------------------- where answers appear, by relative depth
def layer_map():
    lines = []
    old = load("exp_x_qwen.json")
    if old:
        lines.append(("Qwen3.5-0.8B", [(r["layer"], r["knows_known_answers"]) for r in old], len(old)))
    for path in sorted(glob.glob(os.path.join(results_dir, "new_models", "*.map.json"))):
        data = json.load(open(path, encoding="utf-8"))
        name = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3.5-4b": "Qwen3.5-4B"}.get(os.path.basename(path).split(".map")[0], os.path.basename(path))
        lines.append((name, sorted((int(k), v) for k, v in data["known_on_top"].items()), data["layers"]))
    if not lines:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    rows = []
    for i, (name, points, total) in enumerate(lines[:3]):
        xs = [(l + 1) / total for l, _ in points]
        ys = [100 * v for _, v in points]
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=4, label=name)
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", color=INK_2, fontsize=9, va="center")
        rows += [(name, l, f"{(l + 1) / total:.3f}", f"{100 * v:.1f}") for l, v in points]
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Depth in the network (layer / number of layers)")
    ax.set_ylabel("Known answers already the top guess (%)")
    ax.set_title("Answers appear only in the last fifth of each model", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    finish(fig, "layer_map", ["model", "layer", "depth", "known_answers_on_top_percent"], rows, "Read-only: each layer's residual stream read through the output layer, on 26 well-known facts.")


# ---------------------------------------------------------------- loop scan, per model
def loop_scan(name, title, data, arms_key="arms"):
    arms = list(data[arms_key].items())
    names = [n.replace("published fix (positive control)", "published fix").replace("same size, random direction", "random control").replace("same size as the strength 8 edit, random direction", "random control") for n, _ in arms]
    labels = [f"{n}  ({a.get('verdict', '?')}, drift {a['scores']['drift']:.3f})" for n, (_, a) in zip(names, arms)]
    looping = [a.get("looping_change", a["scores"]["looping"] - data["baseline"]["looping"]) for _, a in arms]
    useful = [a.get("useful_change", a["scores"]["useful"] - data["baseline"]["useful"]) for _, a in arms]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(arms) + 1.4))
    ys = list(range(len(arms)))[::-1]
    height = 0.38
    ax.barh([y + height / 2 for y in ys], looping, height=height, color=SERIES[0], edgecolor=SURFACE, linewidth=2, label="looping text")
    ax.barh([y - height / 2 for y in ys], useful, height=height, color=SERIES[1], edgecolor=SURFACE, linewidth=2, label="useful text")
    span = max(abs(v) for v in looping + useful) or 1.0
    for y, value in zip(ys, looping):
        # Just past the bar's end, scaled to the axis, so labels stay beside their bars at any range.
        ax.text(value - 0.015 * span if value < 0 else value + 0.015 * span, y + height / 2, f"{value:+.3f}", va="center", ha="right" if value < 0 else "left", fontsize=8, color=INK_2)
    ax.set_xlim(min(min(looping), min(useful), 0) - 0.18 * span, max(max(looping), max(useful), 0) + 0.12 * span)
    ax.set_yticks(ys, labels, fontsize=8.5)
    ax.tick_params(axis="y", colors=INK_2)
    ax.axvline(0, color=BASELINE, linewidth=1)
    lo = min(min(looping), min(useful))
    if lo < -0.5:
        ax.set_xlim(max(lo * 1.15, -1.0), 0.06)
    ax.set_xlabel("Change in log-probability per token (more negative = less likely)")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    rows = [(n, f"{a:.4f}", f"{b:.4f}", arm.get("verdict", ""), f"{arm['scores']['drift']:.4f}") for n, a, b, (_, arm) in zip(names, looping, useful, arms)]
    finish(fig, name, ["edit", "looping_change", "useful_change", "verdict", "drift"], rows,
           "Held-out matched pairs, not used to find the direction. Read-only passes. Verdicts come from the tracked session: useful text, known facts and drift must not get worse.")


# ---------------------------------------------------------------- loop generation
def loop_generation():
    panels = []
    small = load("exp_t_arms.json")
    big = load("exp_t_arms_big.json")
    if big:
        panels.append(("Qwen3.5-0.8B, 24 problems x 4,096 tokens", big["arms"]["unedited"]["scores"], big["arms"]["mine:15+19:8:64"]["scores"], "unedited", "edit"))
    for path in sorted(glob.glob(os.path.join(results_dir, "new_models", "*.loop_gen.json"))):
        data = json.load(open(path, encoding="utf-8"))
        name = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3.5-4b": "Qwen3.5-4B"}.get(os.path.basename(path).split(".loop_gen")[0], "")
        panels.append((f"{name}, {data['count']} problems x {data['budget']:,} tokens", data["unedited"], data["edited"], "unedited", "edit"))
    if not panels:
        return
    metrics = [("repeats", "repeated\ntail"), ("looping", "looping"), ("finished", "finished"), ("correct", "correct")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8), sharey=True, squeeze=False)
    rows = []
    for ax, (title, before, after, a_name, b_name) in zip(axes[0], panels):
        grouped_bars(ax, [m[1] for m in metrics], ["unedited", "edited (looping direction removed)"], [[100 * before[m[0]] for m in metrics], [100 * after[m[0]] for m in metrics]], legend=False)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0, 110)
        rows += [(title, m[0], f"{100 * before[m[0]]:.1f}", f"{100 * after[m[0]]:.1f}") for m in metrics]
    axes[0][0].set_ylabel("Share of answers (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(0.99, 1.02))
    fig.suptitle("Removing the looping direction: generating on unseen practice problems", x=0.01, ha="left", fontweight="bold", y=1.06)
    finish(fig, "loop_generation", ["test", "measure", "unedited_percent", "edited_percent"], rows, "GSM8K practice problems, thinking on, the published sampling, the same seeds in both arms.")


# ---------------------------------------------------------------- facts on fresh facts
def facts_fresh():
    panels = []
    solved = load("exp_o_qwen.json")
    direct = load("exp_v_qwen.json")
    if solved and direct:
        methods = [("default backprop edit", solved["configs"]["today"]["mean"]), ("best backprop edit", solved["configs"]["L20 last, soft wide nulls, 90%"]["mean"]),
                   ("no backprop", direct["configs"]["20:1.15:all:0.3:3.0:plain"]["mean"])]
        panels.append(("Qwen3.5-0.8B", methods))
    for path in sorted(glob.glob(os.path.join(results_dir, "new_models", "*.facts_new.json"))):
        data = json.load(open(path, encoding="utf-8"))
        name = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3.5-4b": "Qwen3.5-4B"}.get(os.path.basename(path).split(".facts_new")[0], "")
        methods = []
        if "backprop" in data:
            methods.append(("default backprop edit", data["backprop"]["mean"]))
        methods.append(("no backprop", data["direct"]["mean"]))
        panels.append((name, methods))
    if not panels:
        return
    metrics = [("recall", "recall"), ("same_subject", "leak,\nsame co."), ("other_subject", "leak,\nother co."), ("garbled", "garbled")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 3.9), sharey=True, squeeze=False)
    rows = []
    for ax, (title, methods) in zip(axes[0], panels):
        grouped_bars(ax, [m[1] for m in metrics], [f"{n}: {m['passed']} of 10 pass" for n, m in methods], [[100 * m[k] for k, _ in metrics] for _, m in methods], colors=[METHOD_COLORS[n] for n, _ in methods])
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0, 100)
        rows += [(title, n, k, f"{100 * m[k]:.1f}", m["passed"]) for n, m in methods for k, _ in metrics]
    axes[0][0].set_ylabel("Share of test prompts (%)")
    fig.suptitle("Writing facts: 10 facts never used to choose anything", x=0.01, ha="left", fontweight="bold")
    finish(fig, "facts_fresh", ["model", "method", "measure", "percent", "facts_passing_every_check"], rows, "Recall should be high; leak and garbled should be low. A fact passes only if every check passes for it.")


# ---------------------------------------------------------------- no-backprop strength trade-off
def direct_strength():
    points = []
    for name in ("exp_s_qwen.json", "exp_s_qwen3.json"):
        data = load(name)
        if not data:
            continue
        for setup, c in data["configs"].items():
            layer, strength, *_rest = setup.split(":")
            if layer == "20" and setup.endswith("plain"):
                points.append((float(strength), c["mean"]["recall"], c["mean"]["same_subject"], c["mean"]["other_subject"]))
    if not points:
        return
    points.sort()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = [p[0] for p in points]
    for i, (label, index) in enumerate((("recall on new wording", 1), ("leak, same company", 2), ("leak, other company", 3))):
        ys = [100 * p[index] for p in points]
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", fontsize=9, color=INK_2, va="center")
    ax.set_xlim(min(xs) - 0.05, max(xs) + 0.75)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Strength of the written push")
    ax.set_ylabel("Share of test prompts (%)")
    ax.set_title("No backprop: more strength buys recall and leak together", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    finish(fig, "direct_strength", ["strength", "recall", "leak_same_company", "leak_other_company"], [(p[0], f"{100 * p[1]:.1f}", f"{100 * p[2]:.1f}", f"{100 * p[3]:.1f}") for p in points],
           "Qwen3.5-0.8B, layer 20, 8 tuning facts, forward passes only.")


layer_map()
old_scan = load("exp_u_qwen.json")
if old_scan:
    loop_scan("loop_scan_qwen3.5-0.8b", "Qwen3.5-0.8B: removing the looping direction, tracked", old_scan)
for path in sorted(glob.glob(os.path.join(results_dir, "new_models", "*.loop_scan.json"))):
    model = os.path.basename(path).split(".loop_scan")[0]
    pretty = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3.5-4b": "Qwen3.5-4B"}.get(model, model)
    loop_scan(f"loop_scan_{model}", f"{pretty}: removing the looping direction, tracked", json.load(open(path, encoding="utf-8")))
loop_generation()
facts_fresh()
direct_strength()
print("wrote", len(made), "charts to", plots_dir, ":", ", ".join(made))

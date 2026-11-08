"""Reports on bench runs: passes, medians and failure kinds per method, and paired comparisons between methods on the same facts.

Paired tests: an exact McNemar test (binomial on the discordant facts) for passes, and a Wilcoxon signed-rank test for recall and the
same-company leak. scipy is used for the Wilcoxon test when it is installed.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
from collections import Counter

from .methods import ARM_NAMES
from .scoring import read_rows


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test of k successes in n trials at p = 0.5 (McNemar on the discordant pairs)."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] + 1e-12))


def wilcoxon_p(diffs: list[float]) -> float:
    d = [x for x in diffs if x != 0]
    if not d:
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(diffs).pvalue)
    except ImportError:  # normal approximation without scipy
        n = len(d)
        ranks = sorted(range(n), key=lambda i: abs(d[i]))
        r = [0.0] * n
        for rank, i in enumerate(ranks, 1):
            r[i] = rank
        w = sum(r[i] for i in range(n) if d[i] > 0)
        mu, sd = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (w - mu) / sd if sd else 0.0
        return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def load_runs(out_dir: str) -> dict[str, dict]:
    """Every method's rows found in a run folder, by method name."""
    runs = {}
    for method, arm in ARM_NAMES.items():
        rows = read_rows(os.path.join(out_dir, f"{arm}.jsonl"))
        if rows:
            runs[method] = rows
    return runs


def summary(rows: dict) -> dict:
    ids = sorted(rows)
    med = lambda k: st.median(rows[i]["measures"][k] for i in ids if rows[i]["measures"].get(k) is not None) if any(rows[i]["measures"].get(k) is not None for i in ids) else None  # noqa: E731
    kinds = Counter(tuple(sorted({f.split()[0] for f in rows[i]["failed"]})) or ("passed",) for i in ids)
    return dict(facts=len(ids), passed=sum(rows[i]["passed"] for i in ids), recall=med("recall"), same=med("same"), lookalike=med("lookalike"),
                case=med("case"), known_lost_max=max(rows[i]["measures"].get("known_lost", 0) for i in ids),
                drift_max=max(rows[i]["measures"]["drift"] for i in ids), failures={"+".join(k): v for k, v in kinds.most_common()},
                by_relation=dict(Counter(rows[i]["relation"] for i in ids if rows[i]["passed"])))


def paired(a: dict, b: dict) -> dict:
    """b against a on the facts both ran."""
    ids = sorted(set(a) & set(b))
    out = dict(facts=len(ids))
    for k in ("recall", "same"):
        d = [b[i]["measures"][k] - a[i]["measures"][k] for i in ids if a[i]["measures"].get(k) is not None and b[i]["measures"].get(k) is not None]
        out[k] = dict(up=sum(x > 0 for x in d), down=sum(x < 0 for x in d), p=wilcoxon_p(d) if d else 1.0)
    only_b = sum(b[i]["passed"] and not a[i]["passed"] for i in ids)
    only_a = sum(a[i]["passed"] and not b[i]["passed"] for i in ids)
    out["passes"] = dict(a=sum(a[i]["passed"] for i in ids), b=sum(b[i]["passed"] for i in ids), only_a=only_a, only_b=only_b,
                         p=binom_two_sided(min(only_a, only_b), only_a + only_b))
    return out


def two_fact(rows: dict, bench_path: str) -> tuple[int, int]:
    with open(bench_path, encoding="utf-8") as fh:
        facts = json.load(fh)["facts"]
    companies: dict[str, list[str]] = {}
    for f in facts:
        companies.setdefault(f["subject"], []).append(f["id"])
    two = [ids for ids in companies.values() if len(ids) == 2 and all(i in rows for i in ids)]  # companies whose both facts were run
    return sum(all(rows[i]["passed"] for i in ids) for ids in two), len(two)


def report_text(out_dir: str, bench_path: str | None = None) -> str:
    runs = load_runs(out_dir)
    if not runs:
        return f"no runs in {out_dir}"
    lines = [f"Runs in {out_dir}"]
    for method, rows in runs.items():
        s = summary(rows)
        f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(f"  {method:<12} {s['passed']:>2} of {s['facts']} pass | recall {f(s['recall'])} same {f(s['same'])} look-alike {f(s['lookalike'])} "
                     f"case {f(s['case'])} | known lost (max) {s['known_lost_max']} drift (max) {s['drift_max']:.4f}")
        lines.append(f"  {'':<12} failures {s['failures']} | passes by relation {s['by_relation']}")
        if bench_path and method.startswith("graft"):
            both, n = two_fact(rows, bench_path)
            if n:
                lines.append(f"  {'':<12} two-fact companies: both facts pass on {both} of {n}")
    names = [m for m in ("base", "in-context", "block", "graft", "graft-loose") if m in runs and m != "base"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = paired(runs[a], runs[b])
            lines.append(f"  {b} against {a} ({p['facts']} facts): passes {p['passes']['a']} vs {p['passes']['b']}, only {b} {p['passes']['only_b']}, "
                         f"only {a} {p['passes']['only_a']} (McNemar p {p['passes']['p']:.2g}); recall up {p['recall']['up']} down {p['recall']['down']} "
                         f"(p {p['recall']['p']:.2g}); same-company leak up {p['same']['up']} down {p['same']['down']} (p {p['same']['p']:.2g})")
    return "\n".join(lines)

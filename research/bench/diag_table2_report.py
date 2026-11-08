"""Estimated passes for each option of the revision's looser rule (plan.md, "Stage 2, the one revision"), from diag_table2.py's output,
and where the frozen table's recall misses come from (joined with its generated answers on bench40b, devpairs and bench40d).

  At the prompt's last token a fact's score is its strict detectors' best z, or with a looser option the best z of its strict and looser
  detectors. Inside replies only the strict detectors count. The company's highest-scoring fact fires if its score is over 0 and no shared
  detector of a relation the company holds no fact on is ahead by more than the margin.
  recall hit          the fact fires at the last token and m_whole <= the push (2x)
  sibling leak        on a recall prompt, the company's other fact fires at the last token and its m_first <= 2
  same-company leak   scanning the last prompt token and then the unedited reply, the first firing whose m_first <= 2 is the fact's own
  other-fact leak     ... is the other fact's, on a prompt that doesn't ask the other fact's relation (the cross check, limit 0)
  estimated pass      recall >= 0.75, same-company <= 0.25, sibling <= 1/12, other-fact 0 (look-alikes, unrelated companies and name slots
                      are 0 by the exact company index)
Choice: most estimated passes over the 140 facts; ties go to the fewer same-company leaks, then P0, P2, P3, P1. Stop rule: the best must
estimate at least 7 more passes than P0.

Usage: python -I diag_table2_report.py <diag json>
"""
import json
import os
import statistics
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OPTIONS = ("P0", "P2", "P3", "P1")  # the tie order of plan.md
LOOSER_Q = {"P2": "0.95", "P3": "0.9"}
MARGIN, PUSH, SATURATION, STOP_GAIN = 0.5, 2.0, 0.5, 7
RELATIONS = ("owner", "ceo", "founder", "hq")


def detectors(meta, option, end):
    """[(column key, tau, delta)] of one fact: its strict detectors, plus at the prompt's end the option's looser ones."""
    out = []
    for form, s in meta["strict"].items():
        tau = max(s["tau_neg"], s["tau_in"])
        if s["med"] > tau:
            out.append((("strict", form), tau, SATURATION * (s["med"] - tau)))
    if end and option == "P1":
        out += [(("v2", form), d["tau"], d["delta"]) for form, d in meta["v2"].items() if d["delta"] is not None]
    elif end and option in LOOSER_Q:
        for form, s in meta["strict"].items():
            tau = max(s["q"][LOOSER_Q[option]], s["tau_in"])
            if s["med"] > tau:
                out.append((("strict", form), tau, SATURATION * (s["med"] - tau)))
    return out


class Fact:
    def __init__(self, d, shared):
        self.d, self.idx = d, {tuple(k): i for i, k in enumerate(d["columns"])}
        taught = {m["relation"] for m in d["meta"].values()}
        self.shared = [(self.idx[("shared", r, form)], s["tau"], s["delta"]) for r, forms in shared.items() if r not in taught
                       for form, s in forms.items() if s["delta"] is not None]

    def score(self, row, role, option, end):
        return max(((row[self.idx[(role,) + key]] - tau) / delta for key, tau, delta in detectors(self.d["meta"][role], option, end)), default=-1e9)

    def fire(self, row, option, end):
        """(the role that fires or None, whether a shared detector vetoed)."""
        best, zf = None, -1e9
        for role in self.d["meta"]:
            z = self.score(row, role, option, end)
            if z > zf:
                best, zf = role, z
        if best is None or zf <= 0:
            return None, False
        zs = max(((row[i] - tau) / delta for i, tau, delta in self.shared), default=-1e9)
        return (None, True) if zs > zf + MARGIN else (best, False)

    def estimate(self, option):
        d = self.d
        hits = sib = 0
        for r in d["recall"]:
            w, _ = self.fire(r["r"], option, True)
            if w == "own" and r["m_whole"] is not None and r["m_whole"] <= PUSH:
                hits += 1
            if w == "sib" and r["m_first_sib"] is not None and r["m_first_sib"] <= PUSH:
                sib += 1
        same = cross = 0
        for i, s in enumerate(d["same"]):
            for j, p in enumerate(s["positions"]):
                w, _ = self.fire(p["r"], option, j == 0)
                m = p["m_first"].get(w) if w is not None else None
                if m is None or m > PUSH:
                    continue
                if w == "own":
                    same += 1
                elif not (i < len(d["same_rel"]) and d["same_rel"][i] == d["meta"]["sib"]["relation"]):
                    cross += 1
                break
        n_r, n_s = len(d["recall"]), len(d["same"])
        ok = hits / n_r >= 0.75 and same / n_s <= 0.25 + 1e-9 and sib / n_r <= 1 / 12 + 1e-9 and cross == 0
        return dict(recall=hits / n_r, same=same / n_s, same_n=same, sib=sib, cross=cross, ok=ok)


def kind(reply, target):
    if target.lower() in reply.lower():
        return "hit"
    first = target.split()[0].lower()
    return "drift" if first in reply.lower() or first[:max(3, len(first) - 2)] in reply.lower() else "none"


def rows(bench, arm):
    path = os.path.join(RES, bench, f"{arm}.jsonl")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open(sys.argv[1], encoding="utf-8") as fh:
        data = json.load(fh)
    shared = data.pop("_shared")
    facts = {k: Fact(d, shared[d["bench"]]) for k, d in sorted(data.items())}
    benches = sorted({f.d["bench"] for f in facts.values()}, key=lambda b: ("bench40b", "bench40c", "devpairs", "bench40d").index(b))
    print(f"{len(facts)} development facts: " + ", ".join(f"{b} {sum(f.d['bench'] == b for f in facts.values())}" for b in benches))

    # 1. where the frozen table's recall misses come from, joined with its answers
    print("\nThe frozen table's recall prompts, by what it generated and what the readings say at the last token (P0):")
    for b in ("bench40b", "devpairs", "bench40d"):
        T = rows(b, "fact_table_teach")
        c = Counter()
        for fid, r in T.items():
            f = facts.get(f"{b}/{fid}")
            if f is None:
                continue
            for x, reply in zip(f.d["recall"], r["answers"]["recall"]):
                k = kind(reply, f.d["target"])
                w, vetoed = f.fire(x["r"], "P0", True)
                if w == "own":
                    why = "fires, whole answer at 2x" if x["m_whole"] is not None and x["m_whole"] <= PUSH else "fires, answer needs more than 2x"
                elif w == "sib":
                    why = "the other fact fires"
                else:
                    why = "vetoed by a shared detector" if vetoed else "detector shut"
                c[(k, why)] += 1
                if k == "none":
                    for opt in ("P1", "P2", "P3"):
                        w2, _ = f.fire(x["r"], opt, True)
                        if w2 == "own" and x["m_whole"] is not None and x["m_whole"] <= PUSH:
                            c[("none, rescued by", opt)] += 1
        print(f"  {b}: " + "; ".join(f"{k[0]} / {k[1]}: {v}" for k, v in sorted(c.items())))

    # 2. estimated passes per option
    est = {opt: {k: f.estimate(opt) for k, f in facts.items()} for opt in OPTIONS}
    print("\nEstimated passes per option (push 2x, margin 0.5):")
    print(f"{'option':<8}" + "".join(f"{b:>10}" for b in benches) + f"{'all':>6}{'recall med':>12}{'same leaks':>12}{'facts w/ same':>15}{'sibling':>9}{'other-fact':>12}")
    total = {}
    for opt in OPTIONS:
        e = est[opt]
        per = [sum(v["ok"] for k, v in e.items() if facts[k].d["bench"] == b) for b in benches]
        total[opt] = sum(per)
        print(f"{opt:<8}" + "".join(f"{n:>10}" for n in per) + f"{total[opt]:>6}{statistics.median(v['recall'] for v in e.values()):>12.3f}"
              f"{sum(v['same_n'] for v in e.values()):>12}{sum(v['same_n'] > 0 for v in e.values()):>15}{sum(v['sib'] for v in e.values()):>9}"
              f"{sum(v['cross'] for v in e.values()):>12}")
    leaks = {opt: sum(v["same_n"] for v in est[opt].values()) for opt in OPTIONS}
    best = sorted(OPTIONS, key=lambda o: (-total[o], leaks[o], OPTIONS.index(o)))[0]
    gain = total[best] - total["P0"]
    print(f"\nCHOSEN by the rule: {best}, {total[best]} estimated passes, {gain:+d} against P0.")
    print("STOP RULE MET: fewer than 7 more than P0, the revision stops here." if gain < STOP_GAIN else
          f"Stop rule not met ({gain} >= {STOP_GAIN}): the revision goes ahead with {best}.")
    for opt in ("P1", "P2", "P3"):
        e0, e1 = est["P0"], est[opt]
        gained = sorted(k for k in facts if e1[k]["ok"] and not e0[k]["ok"])
        lost = sorted(k for k in facts if e0[k]["ok"] and not e1[k]["ok"])
        print(f"  {opt} against P0: passes gained {len(gained)}, lost {len(lost)}" + (f"; lost: {', '.join(lost)}" if lost else ""))
    print(f"\nFacts failing the estimate under {best}:")
    for k, v in est[best].items():
        if not v["ok"]:
            print(f"  {k:<14} {facts[k].d['relation']:<7} recall {v['recall']:.2f} same {v['same']:.2f} sibling {v['sib']} other-fact {v['cross']}")


if __name__ == "__main__":
    main()

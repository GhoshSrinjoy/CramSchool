"""Step 8 (lab notebook, "a fact stays armed"): what the armed push changed, reply by reply, between two runs of scripts/score_table.py on
the same table. For the recall prompts: which replies gained or lost the answer, and where a gained answer stands in its reply. For every
other group: replies that newly hold the fact's answer. No model needed.

Usage: python scripts/diag_armed_replies.py <rows without arming> <rows with arming> [--show N]
"""
import collections
import json
import sys

BEFORE, AFTER = sys.argv[1:3]
SHOW = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 12
norm = lambda s: " ".join(s.lower().split())  # noqa: E731


def rows(path):
    return {json.loads(line)["id"]: json.loads(line) for line in open(path, encoding="utf-8") if line.strip()}


def kind(reply, target):
    r, t = norm(reply), norm(target)
    return "whole" if t in r else ("started" if t.split()[0] in r else "nothing")


def place(reply, target):
    """Where the answer stands: at the reply's start, after an opening mark or a restated question, or elsewhere."""
    r, t = norm(reply), norm(target)
    at = r.find(t)
    lead = r[:at].strip()
    if not lead or lead in ("**", '"', "'", ":", "-"):
        return "at the start"
    if lead.endswith(("is", "is **", "by", "by **", "in", "in **", ":", ": **", "was", "was **", "named", "to", "to **", "as", "as **", ",", ", **", "are", "of", "of **")):
        return "after a lead-in"
    return "elsewhere"


a, b = rows(BEFORE), rows(AFTER)
moves, places, gained, lost_list = collections.Counter(), collections.Counter(), [], []
for fid, ra in a.items():
    rb = b[fid]
    for i, (x, y) in enumerate(zip(ra["answers"]["recall"], rb["answers"]["recall"])):
        kx, ky = kind(x, ra["target"]), kind(y, ra["target"])
        moves[(kx, ky)] += 1
        if ky == "whole" and kx != "whole":
            p = place(y, ra["target"])
            places[p] += 1
            gained.append((fid, ra["relation"], "plain" if i < 8 else "chat", p, x, y))
        if kx == "whole" and ky != "whole":
            lost_list.append((fid, ra["relation"], "plain" if i < 8 else "chat", x, y))
n = sum(moves.values())
print(f"{n} recall replies. whole answer: {sum(v for (x, _), v in moves.items() if x == 'whole')} before, "
      f"{sum(v for (_, y), v in moves.items() if y == 'whole')} after")
print(f"  gained the whole answer: {len(gained)}; lost it: {len(lost_list)}; changed text without gaining or losing: "
      f"{sum(1 for fid in a for x, y in zip(a[fid]['answers']['recall'], b[fid]['answers']['recall']) if x != y) - len(gained) - len(lost_list)}")
print(f"  where the gained answers stand: {dict(places)}")
print(f"  started but not whole: {sum(v for (x, _), v in moves.items() if x == 'started')} before, {sum(v for (_, y), v in moves.items() if y == 'started')} after")
print(f"\nfacts passing: {sum(r['passed'] for r in a.values())} before, {sum(r['passed'] for r in b.values())} after; pass only after: "
      f"{sorted(f for f in a if b[f]['passed'] and not a[f]['passed'])}; pass only before: {sorted(f for f in a if a[f]['passed'] and not b[f]['passed'])}")
for label, items in (("gained", gained), ("lost", lost_list)):
    print(f"\n{label} (first {SHOW}):")
    for item in items[:SHOW]:
        if label == "gained":
            fid, rel, form, p, x, y = item
            print(f"  {fid} {rel}/{form} [{p}]\n      before: {x[:100]!r}\n      after:  {y[:100]!r}")
        else:
            fid, rel, form, x, y = item
            print(f"  {fid} {rel}/{form}\n      before: {x[:100]!r}\n      after:  {y[:100]!r}")
print("\nelsewhere placements, all of them:")
for fid, rel, form, p, x, y in gained:
    if p == "elsewhere":
        print(f"  {fid} {rel}/{form}: {y[:110]!r}")
print("\nother groups: replies that hold the fact's answer")
for grp in ("same", "lookalike", "other", "names", "case"):
    before = sum(kind(x, r["target"]) == "whole" for r in a.values() for x in r["answers"].get(grp, []))
    after = sum(kind(y, r["target"]) == "whole" for r in b.values() for y in r["answers"].get(grp, []))
    changed = sum(x != y for fid in a for x, y in zip(a[fid]["answers"].get(grp, []), b[fid]["answers"].get(grp, [])))
    print(f"  {grp:9s} {before} before, {after} after ({changed} replies changed text)")
for key in ("same", "cross", "sibling", "garbled", "garbled_other", "known_lost", "drift"):
    va = [r["measures"].get(key) for r in a.values() if r["measures"].get(key) is not None]
    vb = [r["measures"].get(key) for r in b.values() if r["measures"].get(key) is not None]
    if va:
        print(f"  measure {key:13s} max {max(va):.3f} -> {max(vb):.3f}; facts above 0: {sum(v > 0 for v in va)} -> {sum(v > 0 for v in vb)}")

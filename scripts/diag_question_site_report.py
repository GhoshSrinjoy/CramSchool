"""The reading of step 17 (lab notebook), as its plan fixes it, from diag_question_site.py's scores. CPU only.

Wordings: per reading and fit, step 15's estimate of facts passing (a prompt that was open at the build's site and is shut now loses its
answer; a newly opened one gives the whole answer at the bench's measured rate; a fact with a newly opened question about another relation
fails), after the gate at layer 20, last, R1. Jobs: step 12's split, the detector alone and with step 16's veto at the same site. Near: a
bench's near questions with the detector alone. Then the rule.

Usage: python scripts/diag_question_site_report.py [--wordings <site json> <recall_misses json> <saved rows jsonl>]... [--jobs <json>]
           [--near <json>] [--veto-cost <json>]
"""
import collections
import json
import sys

from cramschool.bench.scoring import read_rows

FAMILIES = (("last", "R1"), ("last", "R2"), ("mean", "R2"))
NOW = (32, 24)  # facts passing now on the two benches, in the order given


def load(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return d, [tuple(r) for r in d["readings"]]


def wordings(sets):
    """sets: (site json, recall_misses json, saved rows jsonl) per bench. Returns {(reading, fit): totals} and prints the tables."""
    tot = collections.defaultdict(lambda: dict(est=0, bench=[0] * len(sets), new_same=0, far=0, far_open=0, near=0, near_open=0, open=0))
    readings, n_facts, agree, prompts, cos = None, 0, 0, 0, []
    for b, (site_path, misses_path, rows_path) in enumerate(sets):
        d, readings = load(site_path)
        r20 = readings.index(("last", "R1", 20))
        stored = json.load(open(misses_path, encoding="utf-8"))["recall"]
        opened = [r for r in stored if r["z_end_best"] > 0]
        rate = sum(r["kind"] == "whole" for r in opened) / len(opened)
        saved = read_rows(rows_path)
        for row in d["rows"]:
            n_facts += 1
            cos.append(min(g["cos_saved"] for g in row["gate"].values()))
            other = [x for x in saved[row["id"]]["failed"] if not x.startswith("recall")]
            same0 = row["z_same"]["asbuilt"][r20]
            agree += sum((a > 0) == (s > 0) for a, s in zip(row["z"]["asbuilt"][r20], row["stored_z"]))
            prompts += len(row["stored_z"])
            for fit in row["z"]:
                for r in range(len(readings)):
                    z, zs, t = row["z"][fit][r], row["z_same"][fit][r], tot[(r, fit)]
                    exp = 0.0
                    for i, kind in enumerate(row["stored_kind"]):
                        before, after = row["stored_z"][i] > 0, z[i] > 0
                        t["open"] += after
                        if kind != "whole" and not before:
                            far = row["stored_z"][i] <= -0.5
                            t["far" if far else "near"] += 1
                            t["far_open" if far else "near_open"] += after
                        if kind == "whole":
                            exp += 0.0 if (before and not after) else 1.0
                        elif not before and after:
                            exp += rate
                    new_same = sum(a > 0 and not s > 0 for a, s in zip(zs, same0))
                    ok = bool(exp / len(z) >= 0.75 - 1e-9 and not other and new_same == 0)
                    t["new_same"] += new_same
                    t["est"] += ok
                    t["bench"][b] += ok
    r20 = readings.index(("last", "R1", 20))
    base = tot[(r20, "asbuilt")]
    print(f"WORDINGS: {n_facts} facts, {prompts} recall prompts")
    print(f"gate at layer 20, last, R1, as built: smallest cosine with a saved detector {min(cos):.4f} (bar 0.99); open or shut as stored on "
          f"{agree} of {prompts} prompts = {agree / prompts:.3f} (bar 0.97); estimate {base['est']} facts (bar 56 +- 2)")
    gate = min(cos) >= 0.99 and agree / prompts >= 0.97 and abs(base["est"] - 56) <= 2
    print("gate:", "HOLDS" if gate else "FAILS: nothing below counts")
    passing = []
    for kind, rule in FAMILIES:
        print(f"\n{kind}, {rule}: estimated facts passing of {n_facts} (bench by bench), newly opened questions about another relation | far misses opened")
        print("  layer   as built            one wording left out   in-sample")
        for layer in range(21):
            r = readings.index((kind, rule, layer))
            cells = []
            for fit in ("asbuilt", "lowo", "insample"):
                t = tot[(r, fit)]
                cells.append(f"{t['est']:2d} ({'+'.join(str(x) for x in t['bench'])}) s{t['new_same']:<3d} f{t['far_open']:>2d}/{t['far']}")
            marks = []
            for fit in ("asbuilt", "lowo"):
                t = tot[(r, fit)]
                around = [tot[(readings.index((kind, rule, n)), fit)]["est"] for n in (layer - 1, layer + 1) if 0 <= n <= 20]
                if t["est"] >= 62 and t["new_same"] <= 2 and all(x >= now + 2 for x, now in zip(t["bench"], NOW)) and max(around) >= 59:
                    marks.append(fit)
                    passing.append((kind, rule, layer, fit))
            print(f"  {layer:5d}   {cells[0]:<20}{cells[1]:<23}{cells[2]}" + (f"   PASSES ({', '.join(marks)})" if marks else ""))
    print("\nreadings that pass on wordings:", passing if passing else "none")
    return gate, passing


SHOWN = (("last", "R1", 20), ("last", "R2", 20), ("last", "R2", 16), ("mean", "R2", 9), ("mean", "R2", 15))


def wordings_extras(sets):
    """Reported, not in the rule: a few readings as built, by relation; which questions about another relation they newly open; and the
    estimate when a prompt that was open at the build's site stays open (the stored detector OR the reading at the prompt's end)."""
    from cramschool.bench.scoring import load_bench

    est, either, same, n, examples = (collections.defaultdict(collections.Counter) for _ in range(5))
    for site_path, misses_path, rows_path in sets:
        d, readings = load(site_path)
        r20 = readings.index(("last", "R1", 20))
        stored = json.load(open(misses_path, encoding="utf-8"))["recall"]
        opened = [r for r in stored if r["z_end_best"] > 0]
        rate = sum(r["kind"] == "whole" for r in opened) / len(opened)
        saved, bench = read_rows(rows_path), {f["id"]: f for f in load_bench(d["bench"])}
        for row in d["rows"]:
            rel = row["relation"]
            other = [x for x in saved[row["id"]]["failed"] if not x.startswith("recall")]
            asked = bench[row["id"]]["test"]["same_plain"] + bench[row["id"]]["test"]["same_chat"]
            for key in SHOWN:
                r = readings.index(key)
                n[key][rel] += 1
                exp = exp_or = 0.0
                for i, kind in enumerate(row["stored_kind"]):
                    before, after = row["stored_z"][i] > 0, row["z"]["asbuilt"][r][i] > 0
                    if kind == "whole":
                        exp += 0.0 if (before and not after) else 1.0
                        exp_or += 1.0
                    elif not before and after:
                        exp += rate
                        exp_or += rate
                new = [i for i, (a, s) in enumerate(zip(row["z_same"]["asbuilt"][r], row["z_same"]["asbuilt"][r20])) if a > 0 and not s > 0]
                est[key][rel] += bool(exp / len(row["stored_kind"]) >= 0.75 - 1e-9 and not other and not new)
                either[key][rel] += bool(exp_or / len(row["stored_kind"]) >= 0.75 - 1e-9 and not other and not new)
                same[key][rel] += len(new)
                for i in new:
                    examples[key][asked[i].replace(bench[row["id"]]["subject"], "{s}")] += 1
    print("\nreported, not in the rule: as built, by relation (estimated facts passing; newly opened questions about another relation)")
    for key in SHOWN:
        rels = sorted(n[key])
        print(f"  {str(key):<20}" + "  ".join(f"{rel} {est[key][rel]}/{n[key][rel]} s{same[key][rel]}" for rel in rels)
              + f"  | all {sum(est[key].values())}; with the stored detector kept beside it {sum(either[key].values())}")
        if examples[key]:
            print("      newly opened:", [f"{c}x {t}" for t, c in examples[key].most_common(4)])


def jobs(path):
    d, readings = load(path)
    sites = [tuple(s) for s in d["sites"]]
    r20 = readings.index(("last", "R1", 20))
    rows = d["rows"]
    leak = [r for r in rows if r["group"] == "test" and r["hit"]]
    answered = [r for r in rows if r["group"] == "recall" and r["hit"]]
    taught = [r for r in rows if r["group"] == "taught" and r["hit"]]
    cos = min(g["cos_saved"] for gate in d["gates"].values() for g in gate.values())
    print(f"\nJOBS (step 12's split): {len(leak)} leaking prompts about untaught jobs, {len(answered)} answered chief-executive questions, "
          f"{len(taught)} leaking prompts about the taught jobs")
    print(f"gate: smallest cosine with a saved detector {cos:.4f} (bar 0.99): {'HOLDS' if cos >= 0.99 else 'FAILS'}; at the build's site the "
          f"detector is open at the end of {sum(r['fact'][r20] > 0 for r in leak)} of the leaking prompts and {sum(r['fact'][r20] > 0 for r in answered)} "
          f"of the answered ones")
    passing = []
    for kind, rule in FAMILIES:
        print(f"\n{kind}, {rule}: leaks left of {len(leak)} / answers kept of {len(answered)} (bars: at most {len(leak) // 2}, at least 53) | taught jobs' leaks left of {len(taught)} | other-job prompts that did not leak, newly open")
        print("  layer   detector alone          with the veto")
        for layer in range(21):
            r, s = readings.index((kind, rule, layer)), sites.index((kind, layer))
            cells, marks = [], []
            for veto in (False, True):
                shut = lambda x: x["fact"][r] <= 0 or (veto and x["veto"][s] > 0)  # noqa: E731
                left = sum(not (x["fact"][r20] > 0 and shut(x)) for x in leak)
                kept = sum(not (x["fact"][r20] > 0 and shut(x)) for x in answered)
                t_left = sum(not (x["fact"][r20] > 0 and shut(x)) for x in taught)
                fresh = sum(x["fact"][r] > 0 and not (veto and x["veto"][s] > 0) and not x["fact"][r20] > 0 for x in rows if x["group"] != "recall" and not x["hit"])
                cells.append(f"{left:2d} / {kept:2d} | {t_left:2d} | {fresh:2d}")
                if left <= len(leak) / 2 and kept >= 53:
                    marks.append("with the veto" if veto else "alone")
                    passing.append((kind, rule, layer, marks[-1]))
            print(f"  {layer:5d}   {cells[0]:<24}{cells[1]}" + (f"   MEETS THE BARS ({', '.join(marks)})" if marks else ""))
    print("\nreadings that meet step 12's bars:", passing if passing else "none")
    return cos >= 0.99, passing


def near(path):
    d, readings = load(path)
    r20 = readings.index(("last", "R1", 20))
    rows = [r for r in d["rows"] if "near" in r["z"]]
    cos = min(g["cos_saved"] for r in d["rows"] for g in r["gate"].values())
    leaks = [(r, i) for r in rows for i, h in enumerate(r["hit"]["near"]) if h]
    answers = [(r, i) for r in d["rows"] for i, h in enumerate(r["hit"]["recall"]) if h]
    at_end = sum(r["z"]["near"][r20][i] > 0 for r, i in leaks)
    print(f"\nNEAR QUESTIONS: {len(d['rows'])} facts, {sum(len(r['hit']['near']) for r in rows)} near questions, {len(leaks)} leaking in the stored "
          f"replies ({at_end} with the build's reading open at the prompt's end), {len(answers)} recall prompts answered "
          f"({sum(r['z']['recall'][r20][i] > 0 for r, i in answers)} open at the prompt's end)")
    print(f"gate: smallest cosine with a saved detector {cos:.4f} (bar 0.99): {'HOLDS' if cos >= 0.99 else 'FAILS'}")
    passing = []
    for kind, rule in FAMILIES:
        print(f"\n{kind}, {rule}: leaks left of {len(leaks)} (bar: at most {len(leaks) // 2}) / answers kept of {len(answers)} (bar: at least "
              f"{-(-len(answers) * 9 // 10)}) | same-subject questions newly open | near questions that did not leak, newly open")
        for layer in range(21):
            r = readings.index((kind, rule, layer))
            left = sum(not (x["z"]["near"][r20][i] > 0 and x["z"]["near"][r][i] <= 0) for x, i in leaks)
            kept = sum(not (x["z"]["recall"][r20][i] > 0 and x["z"]["recall"][r][i] <= 0) for x, i in answers)
            new_same = sum(a > 0 and not b > 0 for x in d["rows"] if "same" in x["z"] for a, b in zip(x["z"]["same"][r], x["z"]["same"][r20]))
            new_near = sum(a > 0 and not b > 0 and not h for x in rows for a, b, h in zip(x["z"]["near"][r], x["z"]["near"][r20], x["hit"]["near"]))
            ok = left <= len(leaks) / 2 and kept >= 0.9 * len(answers)
            if ok:
                passing.append((kind, rule, layer))
            print(f"  {layer:5d}   {left:4d} / {kept:4d} | {new_same:3d} | {new_near:3d}" + ("   MEETS THE BARS" if ok else ""))
    # After the fact: the wordings estimate on this bench, which chose no reading. Recall and same-subject questions only (the stored
    # pass flags carry the cross check as it was): a fact counts when its estimated recall is at least 0.75, its stored same-subject
    # leak is within the bar, and no same-subject question newly opens.
    open_now = [(x, i) for x in d["rows"] for i, z in enumerate(x["z"]["recall"][r20]) if z > 0]
    rate = sum(x["hit"]["recall"][i] for x, i in open_now) / len(open_now)
    print(f"\nafter the fact: the wordings estimate on this bench, by recall and same-subject questions only (opened prompts answer at {rate:.3f})")
    print("  reading             estimated facts   same-subject newly open   recall prompts newly open")
    for kind, rule, layer in [("last", "R1", 20)] + [("last", "R2", n) for n in (8, 12, 14, 15, 16, 17, 18, 19, 20)] + [("mean", "R2", n) for n in (3, 9, 15)]:
        r = readings.index((kind, rule, layer))
        est = new_s = new_r = 0
        for x in d["rows"]:
            exp = 0.0
            for i, whole in enumerate(x["hit"]["recall"]):
                before, after = x["z"]["recall"][r20][i] > 0, x["z"]["recall"][r][i] > 0
                if whole:
                    exp += 0.0 if (before and not after) else 1.0
                elif not before and after:
                    exp += rate
                    new_r += 1
            same_hit = x["hit"].get("same", [])
            fresh = sum(a > 0 and not b > 0 for a, b in zip(x["z"]["same"][r], x["z"]["same"][r20])) if "same" in x["z"] else 0
            new_s += fresh
            est += bool(exp / len(x["hit"]["recall"]) >= 0.75 - 1e-9 and (not same_hit or sum(same_hit) / len(same_hit) <= 0.25 + 1e-9) and fresh == 0)
        print(f"  {str((kind, rule, layer)):<20}{est:4d} of {len(d['rows'])}       {new_s:4d}                      {new_r:4d}")
    by_type = collections.defaultdict(lambda: [0, 0])
    for r, i in leaks:
        by_type[r.get("answer_type") or "?"][0] += 1
    for r in rows:
        by_type[r.get("answer_type") or "?"][1] += len(r["hit"]["near"])
    print("\nleaking near questions by answer type (leaking of asked):", {k: f"{a}/{b}" for k, (a, b) in sorted(by_type.items())})
    print("readings that meet the bars on near questions:", passing if passing else "none")
    return cos >= 0.99, passing


def veto_cost(path):
    """After the fact: at each site, on how many chief-executive test recall prompts of other benches step 16's veto fires."""
    d, _ = load(path)
    sites = [tuple(s) for s in d["sites"]]
    rows = d["rows"]
    whole = [r for r in rows if r["whole"]]
    print()
    print(f"THE VETO'S COST (after the fact): {len(rows)} chief-executive test recall prompts of {len({r['bench'] for r in rows})} benches, "
          f"{len(whole)} of them answered in the stored runs. The veto fires on (all; answered):")
    for kind in ("last", "mean"):
        print(f"  {kind}: " + "  ".join(f"{layer}:{sum(r['veto'][sites.index((kind, layer))] > 0 for r in rows)};"
                                          f"{sum(r['veto'][sites.index((kind, layer))] > 0 for r in whole)}" for layer in range(21)))
    for kind, layer in (("mean", 9), ("mean", 10), ("mean", 11), ("last", 14), ("last", 20)):
        s = sites.index((kind, layer))
        hit = collections.Counter(r["template"] for r in rows if r["veto"][s] > 0)
        print(f"  {kind}, layer {layer}: the wordings it fires on most:", [f"{n}x {t}" for t, n in hit.most_common(4)])


def main():
    args = sys.argv[1:]
    w_sets = [tuple(args[i + 1:i + 4]) for i, a in enumerate(args) if a == "--wordings"]
    w_pass = j_pass = n_pass = None
    if w_sets:
        _, w_pass = wordings(w_sets)
        wordings_extras(w_sets)
    if "--jobs" in args:
        _, j_pass = jobs(args[args.index("--jobs") + 1])
    if "--near" in args:
        _, n_pass = near(args[args.index("--near") + 1])
    if "--veto-cost" in args:
        veto_cost(args[args.index("--veto-cost") + 1])
    if w_pass is not None and j_pass is not None and n_pass is not None:
        both = sorted({p[:3] for p in j_pass} & set(n_pass))
        only_one = sorted(({p[:3] for p in j_pass} | set(n_pass)) - set(both))
        print("\nTHE RULE")
        print("- on wordings:", f"{len(w_pass)} readings pass: {w_pass}" if w_pass else "no reading passes")
        print("- on near questions:", f"{len(both)} readings pass both tests: {both}" if both else "no reading passes both tests",
              f"({len(only_one)} meet one of the two: partial)" if only_one else "")
        print("- kill rule:", "MET: the question detector stays where it is" if not w_pass and not both else "not met")


if __name__ == "__main__":
    main()

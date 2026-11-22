"""Step 20 (lab notebook): the end-of-prompt detector beside the layer as built and the looser start, three arms on one saved table.

The as-built arm is scripts/score_table.py on the saved table and the end-of-prompt detector's arm is scripts/end_detector_check.py run.
This script adds the third arm and the comparison. Forward passes and closed-form fits only.

  looser  the package's looser start on a saved table: its parts are built for every stored fact as Graft.add builds them, the option is
          switched on, and the bench is scored by greedy generation with every check (GPU)
  report  the three arms side by side, with the rule of step 20 when --blind is given (CPU)

Usage:
  python scripts/end_arms.py looser <model> <bench json> <table folder> <base rows jsonl> <out jsonl>
  python scripts/end_arms.py report <bench json> <table folder> <as-built rows> <looser rows> <end-detector rows> [--built <rows>] [--blind]

--built   the rows of the run that built the table, scored before it was saved: compared with the as-built arm scored from the saved table
--blind   print the rule; without it the counts are reported and not judged
"""
import json
import os
import statistics
import sys
import time

LEAK_CHECKS = ("same", "lookalike", "other", "names", "cross", "sibling", "garbled", "garbled_other", "known", "drift")
BAR_FACTS, BAR_LEAKY, TIE = 3, 1, 1  # more facts than as built; more facts failing a leak, garbling or damage check; the looser start's tie


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def looser(argv) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cramschool import Graft
    from cramschool.bench.scoring import load_bench, make_baseline, read_rows, score_table_fact, test_prompts
    from cramschool.graft import build as B
    from cramschool.graft.core import last_reads, ratio

    model_dir, bench_path, table, base_path, out_path = argv[:5]
    facts, base = load_bench(bench_path), read_rows(base_path)
    by_key = {f["subject"] + "|" + f["relation"]: f for f in facts}
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = make_baseline(model, tok)
    graft = Graft.load(table, model, tok)
    mctx, o, layer = graft.calibration(), graft.options, graft.layer
    c, cbar = mctx["c"], mctx["cbar"]
    if o.looser_start or any(e["looser"] is not None for e in graft._facts):
        raise SystemExit("the table already holds the looser start: this arm builds it on a table saved without it")

    def scores(x, detectors):
        if not detectors:
            return [-1e9] * x.shape[0]
        return torch.stack([(ratio(x, d["w"].to(x.device), c, cbar) - d["tau"]) / d["delta"] for d in detectors]).max(0).values.tolist()

    # the looser start's parts for every stored fact, as Graft.add builds them, and their scores at the end of the test prompts
    opens, strict, with_end, opened = {}, {}, 0, 0
    for n, e in enumerate(graft._facts, 1):
        f, t0 = e["fact"], time.time()
        ctx = B.build_context([f], graft.wordings, name="table", guard=False, real_world=o.real_world, context=o.context, cache=graft._read_cache(),
                              inside=o.inside, twins=o.twins, teach_cap=o.teach_cap, avoid_as=o.avoid_as)
        e["looser"] = B.build_looser(model, tok, f, ctx, mctx, layer, e["parts"])
        entry = B.entry_of(e["parts"], o.push, e["looser"])
        ends = entry["end_detectors"]
        bf = by_key[f["subject"] + "|" + f["relation"]]
        prompts = test_prompts(bf, tok)
        x = last_reads(model, tok, prompts["recall"] + prompts["same"], layer)
        k = len(prompts["recall"])
        z_end, z_strict = scores(x, ends), scores(x, entry["detectors"])
        opens[bf["id"]] = dict(recall=[round(v, 4) for v in z_end[:k]], same=[round(v, 4) for v in z_end[k:]])
        strict[bf["id"]] = dict(recall=[round(v, 4) for v in z_strict[:k]], same=[round(v, 4) for v in z_strict[k:]])
        with_end += bool(ends)
        opened += sum(v > 0 for v in z_end[:k])
        log(f"[{n}/{len(graft._facts)}] {bf['id']} {f['relation']:<7} looser detectors: {[d['form'] for d in ends]} | open at the end of "
            f"{sum(v > 0 for v in z_end[:k])}/{k} recall and {sum(v > 0 for v in z_end[k:])}/{len(z_end) - k} same-company prompts (strict: "
            f"{sum(v > 0 for v in z_strict[:k])}/{k} and {sum(v > 0 for v in z_strict[k:])}/{len(z_end) - k}) | {time.time() - t0:.0f}s")
    log(f"gate: {with_end} of {len(graft._facts)} facts have a looser detector; they open {opened} recall prompts at the prompt's end")
    if not with_end or not opened:
        raise SystemExit("the gate fails: the looser start would be the layer as built; nothing is scored")
    o.looser_start = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # the bench by generation, as scripts/score_table.py scores a table
    rows = []
    if os.path.exists(out_path):
        os.remove(out_path)
    with graft.attached():
        held = sum(bool(e.get("end_detectors")) for e in graft._layer.entries)
        log(f"the attached layer holds end-of-prompt detectors for {held} facts")
        if held != with_end:
            raise SystemExit("the attached layer does not hold the looser detectors that were built")
        known, drift = baseline.known_check.measure(None), baseline.drift()
        baseline.known_check.measure = lambda context=None: dict(known)
        baseline.drift = lambda: drift
        for n, f in enumerate(facts, 1):
            m, failed, answers, where = score_table_fact(model, tok, f, baseline, base[f["id"]]["answers"], facts)
            row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm="graft_api_looser_start", measures=m,
                       failed=failed, passed=not failed, answers=answers, cross_where=where, end_z=opens[f["id"]], strict_z=strict[f["id"]])
            rows.append(row)
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"[{n:2d}/{len(facts)}] {f['id']} {f['subject']:<23} recall {m['recall']:.2f} same {m['same']:.2f} cross {m['cross']:.2f} "
                f"known lost {m['known_lost']}  {'PASS' if not failed else 'fail: ' + ', '.join(failed)}")
        events = dict(graft._layer.events)
    log(f"{sum(r['passed'] for r in rows)} of {len(rows)} pass; recall median {statistics.median(r['measures']['recall'] for r in rows):.3f}; "
        f"layer events {events}")


def report(argv) -> None:
    from cramschool.bench.report import binom_two_sided
    from cramschool.bench.scoring import PASS, load_bench, read_rows
    from cramschool.gates import mentions
    from cramschool.graft.wordings import normalized

    blind = "--blind" in argv
    built_path = argv[argv.index("--built") + 1] if "--built" in argv else None
    bench_path, table, a_path, b_path, c_path = argv[:5]
    facts = {f["id"]: f for f in load_bench(bench_path)}
    with open(os.path.join(table, "graft.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    taught = {r: {normalized(t) for t in d["plain"] + d["chat"]} for r, d in meta["relations"].items()}
    arms = {"as built": read_rows(a_path), "looser start": read_rows(b_path), "end-of-prompt detector": read_rows(c_path)}
    ids = sorted(facts)
    for name, rows in arms.items():
        if sorted(rows) != ids:
            raise SystemExit(f"{name}: its rows are not the bench's {len(ids)} facts")

    def templates(fid):
        return facts[fid]["test"]["templates_plain"] + facts[fid]["test"]["templates_chat"]

    def leaky(row):
        return any(x.split()[0] in LEAK_CHECKS for x in row["failed"])

    def untaught(row):
        """The fact passes with the test wordings equal to a taught wording left out of its recall (step 11's conservative count)."""
        f = facts[row["id"]]
        kept = [mentions(t, f["target"]) for t, tpl in zip(row["answers"]["recall"], templates(row["id"])) if normalized(tpl) not in taught[f["relation"]]]
        return not [x for x in row["failed"] if not x.startswith("recall")] and bool(kept) and sum(kept) / len(kept) >= PASS["recall"]

    equal = sum(normalized(tpl) in taught[facts[i]["relation"]] for i in ids for tpl in templates(i))
    print(f"{os.path.basename(bench_path)}: {len(ids)} facts, table {table}")
    print(f"test wordings equal to a taught wording: {equal} of {sum(len(templates(i)) for i in ids)} recall prompts")
    if built_path:
        built, ref = read_rows(built_path), arms["as built"]
        differ = [i for i in ids if built[i]["passed"] != ref[i]["passed"]]
        replies = sum(x != y for i in ids for x, y in zip(built[i]["answers"]["recall"], ref[i]["answers"]["recall"]))
        print(f"the as-built arm from the saved table against the run that built it: {sum(r['passed'] for r in ref.values())} and "
              f"{sum(r['passed'] for r in built.values())} facts pass; facts that differ {differ}; recall replies that differ {replies}")

    count = {}
    print()
    for name, rows in arms.items():
        whole = sum(mentions(t, r["target"]) for r in rows.values() for t in r["answers"]["recall"])
        count[name] = dict(all=sum(r["passed"] for r in rows.values()), untaught=sum(untaught(r) for r in rows.values()),
                           leaky=sorted(i for i in ids if leaky(rows[i])), whole=whole)
        by_rel = {}
        for r in rows.values():
            by_rel[r["relation"]] = by_rel.get(r["relation"], 0) + r["passed"]
        fails = {}
        for r in rows.values():
            for x in r["failed"]:
                fails[x.split()[0]] = fails.get(x.split()[0], 0) + 1
        same = [r["measures"]["same"] for r in rows.values()]
        print(f"{name}: {count[name]['all']} of {len(ids)} pass ({count[name]['untaught']} with taught test wordings left out); whole answer in "
              f"{whole} of {sum(len(r['answers']['recall']) for r in rows.values())} recall replies; by relation {dict(sorted(by_rel.items()))}")
        print(f"    failing a leak, garbling or damage check: {len(count[name]['leaky'])} {count[name]['leaky']}; failures by check {fails}; "
              f"same-company leak mean {statistics.mean(same):.4f}, facts with any {sum(v > 0 for v in same)}; known lost (max) "
              f"{max(r['measures']['known_lost'] for r in rows.values())}; drift (max) {max(r['measures']['drift'] for r in rows.values()):.4f}")

    ref = arms["as built"]
    print()
    for name in ("looser start", "end-of-prompt detector"):
        rows = arms[name]
        won = [i for i in ids if rows[i]["passed"] and not ref[i]["passed"]]
        lost = [f"{i} ({', '.join(rows[i]['failed'])})" for i in ids if not rows[i]["passed"] and ref[i]["passed"]]
        p = binom_two_sided(min(len(won), len(lost)), len(won) + len(lost))
        gained = sum(mentions(x, rows[i]["target"]) and not mentions(y, rows[i]["target"]) for i in ids
                     for x, y in zip(rows[i]["answers"]["recall"], ref[i]["answers"]["recall"]))
        dropped = sum(mentions(y, rows[i]["target"]) and not mentions(x, rows[i]["target"]) for i in ids
                      for x, y in zip(rows[i]["answers"]["recall"], ref[i]["answers"]["recall"]))
        print(f"{name} against as built: pass only with it {won}; pass only without it {lost}; exact McNemar p = {p:.3f}")
        print(f"    recall replies with the whole answer: {gained} gained, {dropped} lost; newly failing a leak, garbling or damage check "
              f"{sorted(set(count[name]['leaky']) - set(count['as built']['leaky']))}")
        strict = arms["looser start"]  # the strict detectors' scores at the prompt's end are the table's, logged by the looser arm
        if all("end_z" in rows[i] for i in ids) and all("strict_z" in strict[i] for i in ids):
            opened = [(i, k) for i in ids for k, z in enumerate(rows[i]["end_z"]["recall"]) if z > 0 and not strict[i]["strict_z"]["recall"][k] > 0]
            whole = sum(mentions(rows[i]["answers"]["recall"][k], rows[i]["target"]) for i, k in opened)
            shut = sum(not z > 0 and not strict[i]["strict_z"]["recall"][k] > 0 for i in ids for k, z in enumerate(rows[i]["end_z"]["recall"]))
            same_new = sum(z > 0 and not strict[i]["strict_z"]["same"][k] > 0 for i in ids for k, z in enumerate(rows[i]["end_z"]["same"]))
            print(f"    recall prompts shut for the strict detectors at the prompt's end and open with it: {len(opened)}, with the whole answer "
                  f"{whole}" + (f" = {whole / len(opened):.3f}" if opened else "") + f"; still shut {shut}; same-company prompts newly open {same_new}")

    a, b, c = count["as built"], count["looser start"], count["end-of-prompt detector"]
    gain = min(c["all"] - a["all"], c["untaught"] - a["untaught"])
    more_leaky = len(c["leaky"]) - len(a["leaky"])
    gain_b = min(b["all"] - a["all"], b["untaught"] - a["untaught"])
    more_leaky_b = len(b["leaky"]) - len(a["leaky"])
    tie = b["all"] >= c["all"] - TIE and b["untaught"] >= c["untaught"] - TIE and len(b["leaky"]) <= len(c["leaky"])
    print(f"\nend-of-prompt detector: {c['all'] - a['all']:+d} facts on every test wording, {c['untaught'] - a['untaught']:+d} with taught test wordings "
          f"left out; {more_leaky:+d} facts failing a leak, garbling or damage check")
    print(f"looser start: {b['all'] - a['all']:+d} and {b['untaught'] - a['untaught']:+d}; {more_leaky_b:+d} facts failing a leak, garbling or damage check")
    if not blind:
        print("\nA dry run: reported, not judged.")
        return
    print(f"\nTHE RULE: at least {BAR_FACTS} more facts than as built on both counts, and at most {BAR_LEAKY} more fact failing a leak, garbling or "
          f"damage check")
    if gain >= BAR_FACTS and more_leaky <= BAR_LEAKY:
        print("  -> the end-of-prompt detector: CONFIRMED")
        print("  -> the looser start ties it: no new fit is packaged, the shipped option is the one to point to" if tie else
              "  -> the looser start does not tie it: the detector is built into the package as an option")
    elif gain <= 0 or more_leaky > BAR_LEAKY:
        print("  -> the end-of-prompt detector: KILL RULE MET, it is not built")
    else:
        print("  -> the end-of-prompt detector: 1 or 2 more facts, it stays a script")
    print("  -> the looser start, by the same bar:", "meets it" if gain_b >= BAR_FACTS and more_leaky_b <= BAR_LEAKY else "misses it")


if __name__ == "__main__":
    {"looser": looser, "report": report}[sys.argv[1]](sys.argv[2:])

"""Step 19 (lab notebook): the end-of-prompt detector by real generation, at the build's own site.

For every fact of a saved table, a detector for the prompt's last token is fitted at the table's layer: the build's positives per form
against the prompt negatives alone (step 17's rule R2), threshold at the negatives' 99th percentile, delta half the way to the positives'
median. It goes into the layer's slot for detectors that count only at the prompt's last token (`end_detectors`): a fact then opens at
the prompt's end by its strict detectors or by this one, and inside replies by the strict ones alone. The bench is then scored by greedy
generation with every check, as scripts/score_table.py scores a table. Forward passes and closed-form fits only.

  run     fit the detectors, check the hand-built layer against the package's, score the bench (GPU, about 10 minutes for 40 facts)
  report  the reading as the plan fixes it, over one or more benches (CPU)

Usage:
  python scripts/end_detector_check.py run <model> <bench json> <table folder> <base rows jsonl> <out jsonl>
  python scripts/end_detector_check.py report <out jsonl> <saved rows jsonl> <recall_misses json> [<out jsonl> <saved rows> <misses> ...]
"""
import json
import os
import statistics
import sys
import time

NEG_Q, SAT = 0.99, 0.5
ESTIMATE, NOW = 62, 56          # step 17's estimate for this reading on bench40e + bench40f, and the facts that pass now
BAR_FACTS, BAR_RATE = 59, 0.75  # the estimate less 3; the answer rate of newly opened prompts


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(argv) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cramschool import Graft
    from cramschool.bench.scoring import load_bench, make_baseline, read_rows, score_table_fact, test_prompts
    from cramschool.graft import build as B
    from cramschool.graft.core import ReadCache, last_reads, lda, ratio
    from cramschool.graft.layer import GraftLayer

    model_dir, bench_path, table, base_path, out_path = argv[:5]
    facts, base = load_bench(bench_path), read_rows(base_path)
    by_key = {f["subject"] + "|" + f["relation"]: f for f in facts}
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = make_baseline(model, tok)
    graft = Graft.load(table, model, tok)
    mctx, o, layer = graft.calibration(), graft.options, graft.layer
    c, cbar = mctx["c"], mctx["cbar"]

    def build_layer(entries):
        return GraftLayer(model, tok, entries, graft.shared_detectors(), c, cbar, o.margin, layer, relation_kinds=None, arm=0,
                          aliases=graft._alias_map(), slots=None, exclusions=graft._exclusion_list())

    # the gate: the layer built here with no end-of-prompt detectors is the package's own layer for the table
    mine, ref = build_layer([B.entry_of(e["parts"], o.push) for e in graft._facts]), graft.layer_object()
    same = all(torch.equal(getattr(mine, k), getattr(ref, k)) for k in ("W", "tau", "delta"))
    log(f"gate: the hand-built layer equals the package's (detector matrix, thresholds, delta): {same}")
    if not same:
        raise SystemExit("the gate fails: nothing is scored")

    # the end-of-prompt detectors, and their scores at the end of the test prompts
    cache, entries, opens = ReadCache(), [], {}
    order = sorted(range(len(graft._facts)), key=lambda i: graft._facts[i]["fact"]["relation"])
    built = {}
    for n, i in enumerate(order, 1):
        e = graft._facts[i]
        f, t0 = e["fact"], time.time()
        ctx = B.build_context([f], graft.wordings, name="table", guard=False, real_world=o.real_world, context=o.context, cache=cache,
                              inside=o.inside, twins=o.twins, teach_cap=o.teach_cap, avoid_as=o.avoid_as)
        pos_plain, pos_chat, neg, _ = B.detector_data(f, tok, ctx)
        x_neg = cache.last_reads(model, tok, neg, layer, transient=f["subject"])
        ends = []
        for form, items in (("plain", pos_plain), ("chat", pos_chat)):
            x_pos = cache.last_reads(model, tok, items, layer, transient=f["subject"])
            w = lda(x_pos, [x_neg], mctx["cov"])
            tau = float(torch.quantile(ratio(x_neg, w, c, cbar), NEG_Q))
            med = float(ratio(x_pos, w, c, cbar).median())
            if med > tau:
                ends.append(dict(form=form, w=w.cpu(), tau=tau, delta=SAT * (med - tau)))
        built[i] = ends
        bf = by_key[f["subject"] + "|" + f["relation"]]
        prompts = test_prompts(bf, tok)
        x = last_reads(model, tok, prompts["recall"] + prompts["same"], layer)
        z = torch.stack([(ratio(x, d["w"].to(x.device), c, cbar) - d["tau"]) / d["delta"] for d in ends]).max(0).values.tolist() if ends else \
            [-1e9] * x.shape[0]
        opens[bf["id"]] = dict(recall=[round(v, 4) for v in z[:len(prompts["recall"])]], same=[round(v, 4) for v in z[len(prompts["recall"]):]])
        log(f"[{n}/{len(order)}] {bf['id']} {f['relation']:<7} end-of-prompt detectors: {[d['form'] for d in ends]} | open at the end of "
            f"{sum(v > 0 for v in opens[bf['id']]['recall'])}/{len(prompts['recall'])} recall and {sum(v > 0 for v in opens[bf['id']]['same'])}/"
            f"{len(prompts['same'])} same-company prompts | {time.time() - t0:.0f}s")
    for i, e in enumerate(graft._facts):
        entry = B.entry_of(e["parts"], o.push)
        entry["end_detectors"] = built[i]
        entries.append(entry)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # the bench by generation, as scripts/score_table.py scores a table
    layer_obj = build_layer(entries).attach()
    rows = []
    if os.path.exists(out_path):
        os.remove(out_path)
    try:
        known, drift = baseline.known_check.measure(None), baseline.drift()
        baseline.known_check.measure = lambda context=None: dict(known)
        baseline.drift = lambda: drift
        for n, f in enumerate(facts, 1):
            m, failed, answers, where = score_table_fact(model, tok, f, baseline, base[f["id"]]["answers"], facts)
            row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm="graft_api_end_of_prompt", measures=m,
                       failed=failed, passed=not failed, answers=answers, cross_where=where, end_z=opens[f["id"]])
            rows.append(row)
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"[{n:2d}/{len(facts)}] {f['id']} {f['subject']:<23} recall {m['recall']:.2f} same {m['same']:.2f} cross {m['cross']:.2f} "
                f"known lost {m['known_lost']}  {'PASS' if not failed else 'fail: ' + ', '.join(failed)}")
        events = dict(layer_obj.events)
    finally:
        layer_obj.detach()
    log(f"{sum(r['passed'] for r in rows)} of {len(rows)} pass; recall median {statistics.median(r['measures']['recall'] for r in rows):.3f}; "
        f"layer events {events}")


def report(argv) -> None:
    from cramschool.bench.scoring import read_rows
    from cramschool.gates import mentions

    tot = dict(facts=0, now=0, new=0, whole_now=0, whole_new=0, prompts=0, opened=0, opened_whole=0, kept_open=0, kept_open_whole=0,
               still_shut=0, still_shut_whole=0, same_new_open=0, same_prompts=0)
    gained, lost, fails = [], [], {}
    by_rel = {}
    for a in range(0, len(argv), 3):
        new, old = read_rows(argv[a]), read_rows(argv[a + 1])
        with open(argv[a + 2], encoding="utf-8") as fh:
            misses = json.load(fh)
        stored = {}
        for r in misses["recall"]:
            stored.setdefault(r["id"], []).append(r)
        stored_same = {}
        for r in misses["same"]:
            stored_same.setdefault(r["id"], []).append(r)
        name = os.path.basename(os.path.dirname(argv[a + 1])) or argv[a + 1]
        mean = lambda rows, k: statistics.mean((r["measures"][k] or 0.0) for r in rows.values())  # noqa: E731
        print(f"{name}: same-company leak, mean {mean(new, 'same'):.4f} against {mean(old, 'same'):.4f}, facts with any "
              f"{sum(r['measures']['same'] > 0 for r in new.values())} against {sum(r['measures']['same'] > 0 for r in old.values())}; largest "
              f"look-alike, unrelated and name-slot leak {max(r['measures']['lookalike'] for r in new.values())}, "
              f"{max(r['measures']['other'] for r in new.values())}, {max((r['measures']['names'] or 0) for r in new.values())}; known facts lost "
              f"{max(r['measures']['known_lost'] for r in new.values())}; drift {max(r['measures']['drift'] for r in new.values()):.4f}")
        print(f"{name}: {sum(r['passed'] for r in new.values())} of {len(new)} pass with the end-of-prompt detector; {sum(r['passed'] for r in old.values())} "
              f"now. Recall replies with the whole answer: {sum(mentions(t, r['target']) for r in new.values() for t in r['answers']['recall'])} "
              f"against {sum(mentions(t, r['target']) for r in old.values() for t in r['answers']['recall'])}")
        for fid, r in new.items():
            tot["facts"] += 1
            tot["now"] += old[fid]["passed"]
            tot["new"] += r["passed"]
            rel = by_rel.setdefault(r["relation"], [0, 0, 0])
            rel[0] += 1
            rel[1] += old[fid]["passed"]
            rel[2] += r["passed"]
            if r["passed"] and not old[fid]["passed"]:
                gained.append(f"{name}:{fid}")
            if not r["passed"] and old[fid]["passed"]:
                lost.append(f"{name}:{fid} ({', '.join(r['failed'])})")
            for x in r["failed"]:
                fails[x.split()[0]] = fails.get(x.split()[0], 0) + 1
            for i, t in enumerate(r["answers"]["recall"]):
                whole, was_whole = mentions(t, r["target"]), mentions(old[fid]["answers"]["recall"][i], r["target"])
                before, after = stored[fid][i]["z_end_best"] > 0, r["end_z"]["recall"][i] > 0
                tot["prompts"] += 1
                tot["whole_new"] += whole
                tot["whole_now"] += was_whole
                if not before and after:
                    tot["opened"] += 1
                    tot["opened_whole"] += whole
                elif before:
                    tot["kept_open"] += 1
                    tot["kept_open_whole"] += whole
                else:
                    tot["still_shut"] += 1
                    tot["still_shut_whole"] += whole
            for i, z in enumerate(r["end_z"]["same"]):
                tot["same_prompts"] += 1
                tot["same_new_open"] += z > 0 and not stored_same[fid][i]["z_end"] > 0 if i < len(stored_same.get(fid, [])) else 0
    rate = tot["opened_whole"] / tot["opened"] if tot["opened"] else 0.0
    print(f"\n{tot['facts']} facts: {tot['new']} pass with the end-of-prompt detector, {tot['now']} now (step 17's estimate for this reading: {ESTIMATE})")
    print(f"recall replies with the whole answer: {tot['whole_new']} of {tot['prompts']}, against {tot['whole_now']} now")
    print(f"recall prompts shut at the prompt's end in the stored run and open with the new detector: {tot['opened']}; with the whole answer "
          f"{tot['opened_whole']} = {rate:.3f} (the estimate assumed 0.90 to 0.92)")
    print(f"recall prompts open before: {tot['kept_open']}, with the whole answer {tot['kept_open_whole']}; shut before and still shut: "
          f"{tot['still_shut']}, with the whole answer {tot['still_shut_whole']}")
    print(f"same-company prompts newly open at the prompt's end: {tot['same_new_open']} of {tot['same_prompts']} (the estimate counted 4)")
    print("by relation (facts, passing now, passing with it):", {k: tuple(v) for k, v in sorted(by_rel.items())})
    print("facts that pass only with it:", gained)
    print("facts that pass only without it:", lost)
    print("failures with it, by check:", fails)
    ok = tot["new"] >= BAR_FACTS and rate >= BAR_RATE
    print(f"\nTHE RULE: at least {BAR_FACTS} facts and an answer rate of at least {BAR_RATE} on newly opened prompts: {tot['new']} facts, {rate:.3f}")
    print("  ->", "THE ESTIMATE HOLDS by generation" if ok else "KILL RULE MET: the end-of-prompt detector earns no blind bench")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]](sys.argv[2:])

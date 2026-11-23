"""Whether a table's other facts reach a fact's replies (lab notebook, step 21). The inserted layer reads a fact's detectors only where
its own subject is named last (layer.GraftLayer._push), so facts about other companies should leave a company's replies as they are. The
check: a saved table's facts with the 180 facts of bench40b to bench40e loaded beside them in one layer, scored on the table's bench with
every check, against the rows of the table alone. Forward passes only.

Usage: python scripts/table_size_check.py predict <model folder> <bench json> <base rows jsonl> <label>=<rows jsonl> ...
       python scripts/table_size_check.py run <model folder> <bench json> <table folder> <base rows jsonl> <out jsonl> --against <rows jsonl>
              [--prompt-detector] [--saved-shared]
       python scripts/table_size_check.py split <bench json> <label>=<rows jsonl> ...

predict   no GPU (the tokenizer only): the stored rows of the table alone, read again with the other-facts check widened to the answers
          of the 180 older facts. If no reply changes, these are the counts the run must give.
run       GPU, about 8 minutes: the saved table's facts and the 180 cached facts in one layer. With --prompt-detector the saved table's
          facts carry their end-of-prompt detectors. The 180 older facts carry none: their cached parts were built before the option. The
          shared detectors are built again for all 220 facts, as a table of that size builds them; --saved-shared keeps the saved table's.
split     no GPU: stored rows by how many facts the fact's company holds on the bench. The first label is the reference.
"""
import json
import os
import statistics
import sys
import time

import torch

from cramschool.bench.scoring import GROUPS, cross_hit, load_bench, read_rows, test_prompts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLDER = ["bench40b", "bench40c", "devpairs", "bench40d", "bench40e"]
LEAK_CHECKS = ("same", "lookalike", "other", "names", "cross", "sibling", "garbled", "garbled_other", "known", "drift")
SHARED_INPUTS = ("drafted", "generic", "names", "guard", "by_kind", "kinds_of", "real_world", "twins")  # all that build_shared reads


def older_facts() -> list[tuple[str, dict]]:
    return [(name, f) for name in OLDER for f in load_bench(os.path.join(ROOT, "benchmarks", f"{name}.json"))]


def parts_path(name: str, fid: str) -> str:
    return os.path.join(ROOT, "research", "results", name, "fact_table", f"{fid}.pt")


def labelled(args) -> list[tuple[str, dict]]:
    return [(a.split("=", 1)[0], read_rows(a.split("=", 1)[1])) for a in args]


def leaky(rows) -> list[str]:
    return [r["id"] for r in rows if any(x.split()[0] in LEAK_CHECKS for x in r["failed"])]


def wide_hits(fact: dict, answers: dict, before: dict, tok, older: list[dict]) -> list[dict]:
    """The other-facts check of bench.scoring.score_table_fact for the older facts' answers alone, on stored replies."""
    prompts, out = test_prompts(fact, tok), []
    for grp in GROUPS:
        for i, (t, p) in enumerate(zip(answers[grp], prompts[grp])):
            bad = [g["target"] for g in older if g["target"] != fact["target"] and cross_hit(t, p.text, before[grp][i] if grp in before else None, g)]
            if bad:
                out.append(dict(group=grp, index=i, targets=bad, reply=t))
    return out


def predict(args):
    from transformers import AutoTokenizer
    model_dir, bench_path, base_path = args[:3]
    tok = AutoTokenizer.from_pretrained(model_dir)
    facts, base = load_bench(bench_path), read_rows(base_path)
    older = [f for _, f in older_facts()]
    print(f"{len(facts)} facts of {os.path.basename(bench_path)}; {len(older)} older facts, whose answers the other-facts check now also looks for")
    for label, rows in labelled(args[3:]):
        hits = {f["id"]: wide_hits(f, rows[f["id"]]["answers"], base[f["id"]]["answers"], tok, older) for f in facts}
        hit = {i: h for i, h in hits.items() if any(x["group"] != "near" for x in h)}
        passed = [i for i, r in rows.items() if r["passed"]]
        print(f"{label}: {len(passed)} pass as stored; replies holding an older fact's answer: {sum(len(h) for h in hits.values())} on {len(hit)} facts; "
              f"predicted with 220 loaded: {sum(i not in hit for i in passed)} pass")
        for i, h in hit.items():
            for x in h:
                print(f"    {i} {x['group']}[{x['index']}] {x['targets']}: {x['reply'][:110]!r}")


def split(args):
    facts = load_bench(args[0])
    size = {}
    for f in facts:
        size[f["subject"]] = size.get(f["subject"], 0) + 1
    arms = labelled(args[1:])
    ref = arms[0][1]
    for n in sorted(set(size.values())):
        ids = [f["id"] for f in facts if size[f["subject"]] == n]
        companies = sorted({f["subject"] for f in facts if size[f["subject"]] == n})
        print(f"companies with {n} fact{'s' if n > 1 else ''} on the bench: {len(companies)} companies, {len(ids)} facts")
        for label, rows in arms:
            won = [i for i in ids if rows[i]["passed"] and not ref[i]["passed"]]
            lost = [i for i in ids if not rows[i]["passed"] and ref[i]["passed"]]
            whole = sum(" ".join(rows[i]["target"].lower().split()) in " ".join(t.lower().split()) for i in ids for t in rows[i]["answers"]["recall"])
            every = sum(all(rows[f["id"]]["passed"] for f in facts if f["subject"] == c) for c in companies)
            same = sum(rows[i]["measures"]["same"] > 0 for i in ids)
            print(f"    {label}: {sum(rows[i]['passed'] for i in ids)} of {len(ids)} pass; gained {len(won)} {won}; lost {len(lost)} {lost}; whole answers "
                  f"{whole} of {sum(len(rows[i]['answers']['recall']) for i in ids)}; companies with every fact passing {every} of {len(companies)}; "
                  f"facts with a same-company leak {same}; failing a leak, garbling or damage check {leaky([rows[i] for i in ids])}")


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cramschool import Graft
    from cramschool.bench.scoring import make_baseline, score_table_fact
    from cramschool.graft import GraftLayer, build_context, build_shared, entry_of
    from cramschool.graft.build import make_fact

    t0 = time.time()
    model_dir, bench_path, table, base_path, out_path = args[:5]
    option, saved_shared = "--prompt-detector" in args, "--saved-shared" in args
    against = read_rows(args[args.index("--against") + 1])
    facts, base = load_bench(bench_path), read_rows(base_path)
    older = older_facts()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = make_baseline(model, tok)
    graft = Graft.load(table, model, tok)
    mctx = graft.calibration()
    stored = {(e["fact"]["subject"], e["fact"]["relation"]): e for e in graft._facts}
    if sorted(stored) != sorted((f["subject"], f["relation"]) for f in facts):
        sys.exit("the saved table does not hold the bench's facts")
    if option and any(e.get("prompt") is None for e in graft._facts):
        sys.exit("the saved table holds no end-of-prompt detectors: save it with scripts/score_table.py --prompt-detector --save")

    entries = [entry_of(stored[f["subject"], f["relation"]]["parts"], graft.options.push,
                        prompt=stored[f["subject"], f["relation"]]["prompt"] if option else None) for f in facts]
    entries += [entry_of(torch.load(parts_path(name, f["id"]), weights_only=True), graft.options.push) for name, f in older]

    # The shared detectors, as Graft.shared_detectors builds them: from a context over every stored fact.
    def context(group):
        return build_context(group, graft.wordings, name="table", guard=False, real_world=graft.options.real_world, context=graft.options.context,
                             twins=graft.options.twins)
    alone = [e["fact"] for e in graft._facts]
    with_older = alone + [make_fact(f["subject"], f["relation"], f["target"], graft.wordings, fid=f"{name}-{f['id']}") for name, f in older]
    ctx_alone, ctx_all = context(alone), context(with_older)
    print(f"what the shared detectors are built from, {len(alone)} stored facts against {len(with_older)}: "
          f"{ {k: ctx_alone[k] == ctx_all[k] for k in SHARED_INPUTS} }", flush=True)
    shared = build_shared(model, tok, ctx_all, mctx, graft.layer)

    def gap(a, b):
        worst_w, worst_tau, equal = 0.0, 0.0, True
        for rel, forms in a.items():
            for form, d in forms.items():
                s = b[rel][form]
                equal &= torch.equal(d["w"], s["w"]) and d["tau"] == s["tau"]
                worst_w = max(worst_w, float((d["w"].float() - s["w"].float()).abs().max() / s["w"].float().abs().max()))
                worst_tau = max(worst_tau, abs(d["tau"] - s["tau"]))
        return f"identical {equal}; largest difference of a weight {worst_w:.2e} of the largest weight, of a threshold {worst_tau:.2e}"

    print(f"shared detectors built for {len(with_older)} facts against those built the same way for the {len(alone)} alone: "
          f"{gap(shared, build_shared(model, tok, ctx_alone, mctx, graft.layer))}", flush=True)
    print(f"shared detectors built for {len(with_older)} facts against the saved table's: {gap(shared, graft._shared)}", flush=True)
    if graft.options.fast:  # the saved table read its prompts through the fast build's cache; an empty one shows what the cache alone changes
        from cramschool.graft.core import ReadCache
        print(f"    and against the same build through an empty cache of the fast build: "
              f"{gap(shared, build_shared(model, tok, dict(ctx_all, cache=ReadCache()), mctx, graft.layer))}", flush=True)
    layer = GraftLayer(model, tok, entries, graft._shared if saved_shared else shared, mctx["c"], mctx["cbar"], graft.options.margin, graft.layer,
                       aliases=graft._alias_map(), exclusions=graft._exclusion_list())
    print(f"layer: {len(entries)} facts on {len(layer.by_company)} companies, {layer.W.shape[1]} columns; end-of-prompt detectors on "
          f"{sum(bool(e.get('end_detectors')) for e in entries)} facts; shared detectors {'as saved with the table' if saved_shared else 'built for all facts'}; "
          f"ready in {time.time() - t0:.0f}s", flush=True)

    loaded = facts + [f for _, f in older]
    if os.path.exists(out_path):
        os.remove(out_path)
    rows = []
    layer.attach()
    try:
        # the known facts and the drift are measured with the whole layer attached, once, as scripts/score_table.py does
        known, drift = baseline.known_check.measure(None), baseline.drift()
        baseline.known_check.measure = lambda context=None: dict(known)
        baseline.drift = lambda: drift
        for n, f in enumerate(facts, 1):
            m, failed, answers, where = score_table_fact(model, tok, f, baseline, base[f["id"]]["answers"], loaded)
            row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm=f"220_loaded{'_prompt_detector' if option else ''}",
                       measures=m, failed=failed, passed=not failed, answers=answers, cross_where=where)
            rows.append(row)
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            ref = against[f["id"]]["answers"]
            changed = sum(a != b for g in answers for a, b in zip(answers[g], ref.get(g, []))) + sum(len(answers[g]) != len(ref.get(g, [])) for g in answers)
            print(f"[{n:2d}/{len(facts)}] {f['id']} {f['subject']:<23} recall {m['recall']:.2f} same {m['same']:.2f} cross {m['cross']:.2f} known lost "
                  f"{m['known_lost']}  {'PASS' if not failed else 'fail: ' + ', '.join(failed)}  | replies changed {changed}", flush=True)
    finally:
        layer.detach()
    events = dict(layer.events)

    total = sum(len(v) for r in rows for v in r["answers"].values())
    differ = [(r["id"], g, i) for r in rows for g, v in r["answers"].items() for i, t in enumerate(v)
              if i >= len(against[r["id"]]["answers"].get(g, [])) or t != against[r["id"]]["answers"][g][i]]
    verdict = [r["id"] for r in rows if r["passed"] != against[r["id"]]["passed"]]
    med = lambda k: statistics.median(r["measures"][k] for r in rows)  # noqa: E731
    print(f"\n{len(entries)} facts loaded{', end-of-prompt detector' if option else ''}: {sum(r['passed'] for r in rows)} of {len(rows)} pass, against "
          f"{sum(r['passed'] for r in against.values())} with the table alone; recall median {med('recall'):.3f}; known lost (max) "
          f"{max(r['measures']['known_lost'] for r in rows)}; drift (max) {max(r['measures']['drift'] for r in rows):.4f}")
    print(f"replies that differ from the table alone: {len(differ)} of {total}{' ' + str(differ[:12]) if differ else ''}")
    print(f"facts whose verdict differs: {verdict}; facts failing a leak, garbling or damage check: {leaky(rows)}")
    print(f"replies with another stored fact's answer: {[(r['id'], w) for r in rows for w in r['cross_where']]}")
    print(f"layer events over the run: {events}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    {"predict": predict, "run": run, "split": split}[sys.argv[1]](sys.argv[2:])

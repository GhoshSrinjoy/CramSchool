"""Four facts on every company (lab notebook, step 22). A bench's companies hold one or two facts each. Here every company of the bench
gets a fact for each relation it lacks, with answers taken from two other benches, so that every company holds all four relations: more
facts compete where a company is named, and no relation is left for the veto to read. The bench's own facts are then scored on that table
as built and with the end-of-prompt detector (scripts/score_table.py), against their scores on the table of the bench's facts alone.
Forward passes only.

Usage: python scripts/dense_companies.py fillers <bench json> <out json> <bench json to take answers from> [...]
       python scripts/dense_companies.py build <model folder> <saved table> <fillers json> <out table folder>
       python scripts/dense_companies.py report <bench json> <dense table folder> <fillers json> <sparse as-built rows> <sparse detector rows>
              <dense as-built rows> <dense detector rows> [--base <base rows jsonl>]

fillers   no GPU: the added facts. For each relation, the bench's companies that lack it take the answers of the other benches' facts of
          that relation, in order.
build     GPU, about 15 s per added fact: the saved table of the bench's facts is loaded and the added facts go in through Graft.add, with
          the table's own options. The bench's facts keep the parts they were saved with.
report    no GPU: the four counts, the dense table against the sparse one in each arm, the detector against as built on the dense
          table, what the companies' other facts do on the bench's prompts, and the rule.
"""
import json
import os
import statistics
import sys
import time

RELATIONS = ("owner", "ceo", "founder", "hq")
LEAK_CHECKS = ("same", "lookalike", "other", "names", "cross", "sibling", "garbled", "garbled_other", "known", "drift")
BAR_FACTS, BAR_LEAKY, BAR_DENSITY = 3, 1, 3


def fillers(args):
    import re

    from cramschool.bench.scoring import load_bench

    bench_path, out_path, sources = args[0], args[1], args[2:]
    facts = load_bench(bench_path)
    companies = list(dict.fromkeys(f["subject"] for f in facts))
    words = lambda s: set(re.findall(r"[a-z]+", s.lower()))  # noqa: E731
    out = []
    for rel in RELATIONS:
        lacking = [c for c in companies if not any(f["subject"] == c and f["relation"] == rel for f in facts)]
        pool = [(os.path.basename(p)[:-5], f) for p in sources for f in load_bench(p) if f["relation"] == rel]
        if len(pool) < len(lacking):
            sys.exit(f"{rel}: {len(lacking)} companies lack it and the other benches give {len(pool)} answers")
        out += [dict(subject=c, relation=rel, target=f["target"], source=f"{name} {f['id']}") for c, (name, f) in zip(lacking, pool)]
    own = [f["target"] for f in facts]
    every = own + [x["target"] for x in out]
    shared = [(a, b) for i, a in enumerate(every) for b in every[i + 1:] if words(a) & words(b)]
    per_company = {c: sorted(f["relation"] for f in facts if f["subject"] == c) + sorted(x["relation"] for x in out if x["subject"] == c) for c in companies}
    if shared or any(sorted(v) != sorted(RELATIONS) for v in per_company.values()):
        sys.exit(f"answers sharing a word: {shared}; companies without exactly the four relations: "
                 f"{[c for c, v in per_company.items() if sorted(v) != sorted(RELATIONS)]}")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"{len(facts)} facts on {len(companies)} companies; {len(out)} added facts, by relation "
          f"{ {r: sum(x['relation'] == r for x in out) for r in RELATIONS} }; every company now holds the four relations; no two answers share a word")
    print(f"written {out_path}")


def build(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cramschool import Graft

    model_dir, table, fillers_path, out_dir = args[:4]
    with open(fillers_path, encoding="utf-8") as fh:
        added = json.load(fh)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    graft = Graft.load(table, model, tok)
    before = {e["fact"]["id"]: e["parts"] for e in graft._facts}
    print(f"loaded {len(graft)} facts; options fast {graft.options.fast}, teach_cap {graft.options.teach_cap}, prompt_detector "
          f"{graft.options.prompt_detector}; facts with saved end-of-prompt detectors {sum(e.get('prompt') is not None for e in graft._facts)}", flush=True)
    if not graft.options.prompt_detector or any(e.get("prompt") is None for e in graft._facts):
        sys.exit("the saved table must hold its facts' end-of-prompt detectors, with the option on")
    t0 = time.time()
    for n, x in enumerate(added, 1):
        graft.add(x["subject"], x["relation"], x["target"])
        if n % 10 == 0 or n == len(added):
            print(f"[build {n}/{len(added)}] {x['relation']:8s} {time.time() - t0:.0f}s", flush=True)
    by_company = {}
    for e in graft._facts:
        by_company.setdefault(e["fact"]["subject"], []).append(e["fact"]["relation"])
    kept = all(e["parts"] is before[e["fact"]["id"]] for e in graft._facts if e["fact"]["id"] in before)
    ok = all(sorted(v) == sorted(RELATIONS) for v in by_company.values()) and kept and all(e.get("prompt") is not None for e in graft._facts)
    print(f"{len(graft)} facts on {len(by_company)} companies; every company holds the four relations, the saved facts keep their parts and "
          f"every fact has its end-of-prompt detectors: {ok}; built in {time.time() - t0:.0f}s", flush=True)
    if not ok:
        sys.exit("the table is not the one the step needs")
    graft.save(out_dir)
    print(f"saved {out_dir}")


def report(args):
    from cramschool.bench.report import binom_two_sided
    from cramschool.bench.scoring import PASS, load_bench, read_rows
    from cramschool.gates import leaks, mentions
    from cramschool.graft.wordings import normalized

    bench_path, table, fillers_path = args[:3]
    facts = {f["id"]: f for f in load_bench(bench_path)}
    ids = sorted(facts)
    with open(fillers_path, encoding="utf-8") as fh:
        added = json.load(fh)
    with open(os.path.join(table, "graft.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    taught = {r: {normalized(t) for t in d["plain"] + d["chat"]} for r, d in meta["relations"].items()}
    names = ("sparse, as built", "sparse, detector", "dense, as built", "dense, detector")
    arms = {n: read_rows(p) for n, p in zip(names, args[3:7])}
    for n, rows in arms.items():
        if sorted(rows) != ids:
            sys.exit(f"{n}: its rows are not the bench's {len(ids)} facts")
    answer = {(f["subject"], f["relation"]): f["target"] for f in facts.values()}
    answer.update({(x["subject"], x["relation"]): x["target"] for x in added})
    size = {}
    for f in facts.values():
        size[f["subject"]] = size.get(f["subject"], 0) + 1

    def templates(i):
        return facts[i]["test"]["templates_plain"] + facts[i]["test"]["templates_chat"]

    def leaky(row):
        return any(x.split()[0] in LEAK_CHECKS for x in row["failed"])

    def untaught(row):
        f = facts[row["id"]]
        kept = [mentions(t, f["target"]) for t, tpl in zip(row["answers"]["recall"], templates(row["id"])) if normalized(tpl) not in taught[f["relation"]]]
        return not [x for x in row["failed"] if not x.startswith("recall")] and bool(kept) and sum(kept) / len(kept) >= PASS["recall"]

    count = {}
    print(f"{os.path.basename(bench_path)}: {len(ids)} facts on {len(size)} companies; dense table {table}: {len(meta['facts'])} facts, "
          f"{len(added)} of them added\n")
    for n, rows in arms.items():
        fails = {}
        for r in rows.values():
            for x in r["failed"]:
                fails[x.split()[0]] = fails.get(x.split()[0], 0) + 1
        count[n] = dict(all=sum(r["passed"] for r in rows.values()), untaught=sum(untaught(r) for r in rows.values()),
                        leaky=sorted(i for i in ids if leaky(rows[i])))
        whole = sum(mentions(t, r["target"]) for r in rows.values() for t in r["answers"]["recall"])
        by_rel = {rel: sum(r["passed"] for r in rows.values() if r["relation"] == rel) for rel in RELATIONS}
        by_size = {k: sum(rows[i]["passed"] for i in ids if size[facts[i]["subject"]] == k) for k in sorted(set(size.values()))}
        same = [r["measures"]["same"] for r in rows.values()]
        print(f"{n}: {count[n]['all']} of {len(ids)} pass ({count[n]['untaught']} with taught test wordings left out); whole answer in {whole} of "
              f"{sum(len(r['answers']['recall']) for r in rows.values())} recall replies; by relation {by_rel}; by facts the company holds on the bench {by_size}")
        print(f"    failing a leak, garbling or damage check: {len(count[n]['leaky'])} {count[n]['leaky']}; failures by check {fails}; same-company leak "
              f"mean {statistics.mean(same):.4f}, facts with any {sum(v > 0 for v in same)}; known lost (max) "
              f"{max(r['measures']['known_lost'] for r in rows.values())}; drift (max) {max(r['measures']['drift'] for r in rows.values()):.4f}")

    def against(a, b, label):
        ra, rb = arms[a], arms[b]
        won = [i for i in ids if rb[i]["passed"] and not ra[i]["passed"]]
        lost = [f"{i} ({', '.join(rb[i]['failed'])})" for i in ids if not rb[i]["passed"] and ra[i]["passed"]]
        p = binom_two_sided(min(len(won), len(lost)), len(won) + len(lost))
        changed = {g: sum(x != y for i in ids for x, y in zip(rb[i]["answers"][g], ra[i]["answers"][g])) for g in rb[ids[0]]["answers"]}
        gained = sum(mentions(x, rb[i]["target"]) and not mentions(y, rb[i]["target"]) for i in ids for x, y in zip(rb[i]["answers"]["recall"], ra[i]["answers"]["recall"]))
        dropped = sum(mentions(y, rb[i]["target"]) and not mentions(x, rb[i]["target"]) for i in ids for x, y in zip(rb[i]["answers"]["recall"], ra[i]["answers"]["recall"]))
        print(f"{label}: pass only in the second {won}; only in the first {lost}; exact McNemar p = {p:.3f}")
        print(f"    replies that differ, by group: {changed}; recall replies with the whole answer: {gained} gained, {dropped} lost; newly failing a "
              f"leak, garbling or damage check {sorted(set(count[b]['leaky']) - set(count[a]['leaky']))}")

    print()
    against("sparse, as built", "dense, as built", "the dense table against the sparse one, as built")
    against("sparse, detector", "dense, detector", "the dense table against the sparse one, with the detector")
    against("dense, as built", "dense, detector", "on the dense table, the detector against as built")

    # what the companies' facts do on the bench's same-company prompts: a prompt labelled with a relation asks for that relation's fact
    print()
    for n, rows in arms.items():
        asked = right = other_fact = generic = generic_any = own = recall_other = 0
        for i in ids:
            f, t = facts[i], facts[i]["test"]
            labels = t["same_plain_rel"] + t["same_chat_rel"]
            mine = [v for (s, r), v in answer.items() if s == f["subject"] and v != f["target"]]
            for reply, label in zip(rows[i]["answers"]["same"], labels):
                if label == "generic":
                    generic += 1
                    generic_any += any(mentions(reply, v) for v in mine + [f["target"]])
                else:
                    asked += 1
                    right += mentions(reply, answer[f["subject"], label])
                    other_fact += any(mentions(reply, v) for v in mine if v != answer[f["subject"], label])
                    own += mentions(reply, f["target"])
            recall_other += sum(any(mentions(reply, v) for v in mine) for reply in rows[i]["answers"]["recall"])
        print(f"{n}: same-company prompts asking for another relation {asked}: the answer the dense table stores for it {right}, the scored fact's own "
              f"answer {own}, another of the company's answers {other_fact}; general questions {generic}: any of the company's answers "
              f"{generic_any}; recall replies holding another of the company's answers {recall_other}")

    # The general questions by wording, and another of the company's answers read as the same-company check reads the fact's own: whole,
    # or one distinctive part of it that neither the prompt nor the unedited reply has. The other-facts check reads whole answers only.
    base = read_rows(args[args.index("--base") + 1]) if "--base" in args else None
    print()
    for n, rows in arms.items():
        stored = (lambda f, rel: (f["subject"], rel) in answer and ("dense" in n or any(g["subject"] == f["subject"] and g["relation"] == rel for g in facts.values())))
        by_wording, kinds, part_facts = {}, {}, set()
        for i in ids:
            f, t = facts[i], facts[i]["test"]
            for k, (prompt, label) in enumerate(zip(t["same_plain"] + t["same_chat"], t["same_plain_rel"] + t["same_chat_rel"])):
                if label != "generic":
                    continue
                reply = rows[i]["answers"]["same"][k]
                wording = prompt.replace(f["subject"], "{s}")
                by_wording.setdefault(wording, [0, 0])[0] += 1
                for rel in RELATIONS:
                    if not stored(f, rel):
                        continue
                    target = answer[f["subject"], rel]
                    if mentions(reply, target):
                        by_wording[wording][1] += 1
                    if base is not None and target != f["target"] and leaks(reply, target, prompt, base[i]["answers"]["same"][k]):
                        first = reply.strip().lower().startswith(target.split()[0].lower())
                        key = f"{rel}, {'whole' if mentions(reply, target) else 'a part'}, {'the first word' if first else 'later in the reply'}"
                        kinds[key] = kinds.get(key, 0) + 1
                        part_facts.add(i)
        hit = {w: f"{v[1]} of {v[0]}" for w, v in by_wording.items() if v[1]}
        print(f"{n}: general wordings {len(by_wording)}; those whose replies hold one of the company's answers: {hit}")
        if base is not None:
            passing = [i for i in ids if rows[i]["passed"]]
            print(f"    general questions holding another of the company's answers, whole or a distinctive part: {sum(kinds.values())} on {len(part_facts)} "
                  f"facts {kinds}; facts passing that have none: {sum(i not in part_facts for i in passing)} of the {len(passing)} passing (a reading, not the rule)")

    sa, sd, da, dd = (count[n] for n in names)
    gain = min(dd["all"] - da["all"], dd["untaught"] - da["untaught"])
    more_leaky = len(dd["leaky"]) - len(da["leaky"])
    print(f"\nTHE FOUR COUNTS (every test wording; taught test wordings left out): sparse as built {sa['all']}; {sa['untaught']} | sparse with the "
          f"detector {sd['all']}; {sd['untaught']} | dense as built {da['all']}; {da['untaught']} | dense with the detector {dd['all']}; {dd['untaught']}")
    print(f"DENSITY, as built: {da['all'] - sa['all']:+d} facts against the sparse table -> "
          + (f"a limit of GRAFT itself: at least {BAR_DENSITY} facts lost with four facts on a company" if sa["all"] - da["all"] >= BAR_DENSITY else
             f"fewer than {BAR_DENSITY} facts lost"))
    print(f"DENSITY, with the detector: {dd['all'] - sd['all']:+d} facts against the sparse table -> "
          + (f"a limit of the option: at least {BAR_DENSITY} facts lost with four facts on a company" if sd["all"] - dd["all"] >= BAR_DENSITY else
             f"fewer than {BAR_DENSITY} facts lost"))
    print(f"THE RULE, on the dense table: at least {BAR_FACTS} more facts than as built on both counts, and at most {BAR_LEAKY} more fact failing a leak, "
          f"garbling or damage check. The detector: {dd['all'] - da['all']:+d} and {dd['untaught'] - da['untaught']:+d} facts; {more_leaky:+d} failing")
    if gain >= BAR_FACTS and more_leaky <= BAR_LEAKY:
        print("  -> HOLDS with four facts on a company")
    elif gain <= 0 or more_leaky > BAR_LEAKY:
        print("  -> KILL RULE MET: the option is for tables with at most two facts on a company")
    else:
        print("  -> 1 or 2 more facts: no claim for four facts on a company")


if __name__ == "__main__":
    {"fillers": fillers, "build": build, "report": report}[sys.argv[1]](sys.argv[2:])

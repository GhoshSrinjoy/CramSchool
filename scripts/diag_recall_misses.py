"""Why does a recall reply hold no part of the answer? (Lab notebook, "bench40f's failures".) Every fact's test prompts are generated again
with the saved table attached, exactly as the bench generated them, and at every step the layer's own readings are logged for the recall
prompts' fact: its detectors' scores z (0 is the threshold, 2 the taught wordings' median), the highest score of a shared detector that could
veto it, its sibling's score, and what the layer did. The same is logged for the other questions about the company (the "same" group), with
the highest score among the company's stored facts: how often a fact opens on a question that isn't its own. Forward passes only. GPU,
about 3 minutes for 40 facts.

Usage: python scripts/diag_recall_misses.py <model folder> <bench json> <table folder> <saved rows jsonl> <out json>
"""
import collections
import json
import statistics
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.bench.scoring import BATCH, GROUPS, MAX_NEW_TOKENS, load_bench, read_rows, test_prompts
from cramschool.gates import generate_batch

MODEL, BENCH, TABLE, ROWS, OUT = sys.argv[1:6]
facts, saved = load_bench(BENCH), read_rows(ROWS)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
graft = Graft.load(TABLE, model, tok)
graft.attach()
layer = graft._layer
index_of = {(e["subject"], e["relation"]): i for i, e in enumerate(layer.entries)}
norm = lambda s: " ".join(s.lower().split())  # noqa: E731

watch = dict(fact=None, rows=0, log=None, same=0, same_log=None)  # the fact whose recall prompts sit in the first rows of the batch being generated
original = layer._push


def logged_push(module, args, output):
    i, n = watch["fact"], watch["rows"]
    if i is None or layer.ids is None:
        return original(module, args, output)
    z = layer.readings(layer.x_last)
    company = layer.index.lookup[norm(layer.entries[i]["subject"])]
    veto_cols = [(r, k) for r in layer.relations if r not in layer.taught[company] and layer.kind_of(r) in layer.subject_kinds[company]
                 for k in layer.shared_cols.get(r, [])]
    m = n + watch["same"]
    before = [dict(st) for st in layer.state[:m]]
    out = original(module, args, output)
    for b in range(n, min(m, len(z))):  # other questions about the company: the best score among its stored facts
        best = max((z[b][k] for j in layer.by_company[company] for k in layer.fact_cols[j]), default=-1e9)
        after = layer.state[b]
        watch["same_log"][b - n].append(dict(z=best, pushed=after["pending"] is not None and before[b]["pending"] is None, done=after["done"]))
    for b in range(min(n, len(z))):
        own = {d["form"]: z[b][k] for d, k in zip(layer.entries[i]["detectors"], layer.fact_cols[i])}
        sib = max((z[b][k] for j in layer.by_company[company] if j != i for k in layer.fact_cols[j]), default=None)
        veto = max(((z[b][k], r) for r, k in veto_cols), default=(None, None))
        after = layer.state[b]
        did = ("continued" if after["progress"] is not None and before[b]["progress"] is not None else
               "accepted" if after["done"] and not before[b]["done"] else
               "pushed" if after["pending"] is not None else "nothing")
        watch["log"][b].append(dict(z=own, sibling=sib, veto=veto[0], veto_relation=veto[1], did=did, done=before[b]["done"]))
    return out


layer.detach()
layer._push = logged_push
layer.attach()

out_rows, same_rows = [], []
for n_done, f in enumerate(facts):
    prompts = test_prompts(f, tok)
    flat = [p for g in GROUPS for p in prompts[g]]
    n = len(prompts["recall"])
    n_same = len(prompts["same"])
    watch.update(fact=index_of[(f["subject"], f["relation"])], rows=n, log=[[] for _ in range(n)], same=n_same, same_log=[[] for _ in range(n_same)])
    first = generate_batch(model, tok, flat[:BATCH], MAX_NEW_TOKENS, batch_size=BATCH)  # the bench's first batch: the recall prompts lead it
    target, first_word = norm(f["target"]), norm(f["target"]).split()[0]
    templates = f["test"]["templates_plain"] + f["test"]["templates_chat"]
    for b in range(n):
        reply, was = first[b], saved[f["id"]]["answers"]["recall"][b]
        steps = watch["log"][b]
        best = [max(s["z"].values(), default=-1e9) for s in steps]
        out_rows.append(dict(
            id=f["id"], relation=f["relation"], form="plain" if b < len(f["test"]["plain"]) else "chat", template=templates[b], reply=reply,
            same_as_saved=reply == was, kind="whole" if target in norm(reply) else ("started" if first_word in norm(reply) else "nothing"),
            z_end=steps[0]["z"], z_end_best=best[0], z_max=max(best), step_of_max=best.index(max(best)), sibling_end=steps[0]["sibling"],
            z_steps=[round(v, 3) for v in best], z_later_max=max(best[1:]) if len(best) > 1 else None,
            veto_end=steps[0]["veto"], veto_relation=steps[0]["veto_relation"], did=[s["did"] for s in steps],
            vetoed=any(z > 0 and s["veto"] is not None and s["veto"] > z + layer.margin and not s["done"] for s, z in zip(steps, best)),
        ))
    for b in range(n_same):
        steps, reply = watch["same_log"][b], first[n + b]
        same_rows.append(dict(id=f["id"], relation=f["relation"], prompt=prompts["same"][b].text[-90:], reply=reply, z_end=steps[0]["z"], z_max=max(s["z"] for s in steps),
                              pushes=sum(s["pushed"] for s in steps), answered=steps[-1]["done"], has_target=norm(f["target"]) in norm(reply)))
    watch["fact"] = None
    print(f"[{n_done + 1:2d}/{len(facts)}] {f['id']} {f['relation']:8s} whole {sum(r['kind'] == 'whole' for r in out_rows[-n:])}/{n}", flush=True)

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(dict(recall=out_rows, same=same_rows), fh, ensure_ascii=False, indent=1)


def bins(values, edges=(-1e9, -4, -2, -1, -0.5, -0.25, 0, 1e9), names=("<-4", "-4..-2", "-2..-1", "-1..-0.5", "-0.5..-0.25", "-0.25..0", ">0")):
    return {name: sum(lo < v <= hi for v in values) for name, lo, hi in zip(names, edges, edges[1:])}


total = len(out_rows)
print(f"\n{total} recall prompts; replies identical to the saved run: {sum(r['same_as_saved'] for r in out_rows)}")
kinds = collections.Counter(r["kind"] for r in out_rows)
print(f"whole answer {kinds['whole']}, started then lost {kinds['started']}, no part of the answer {kinds['nothing']}")
miss = [r for r in out_rows if r["kind"] == "nothing"]
never = [r for r in miss if r["z_max"] <= 0]
opened = [r for r in miss if r["z_max"] > 0]
print(f"\nthe {len(miss)} replies with no part of the answer:")
print(f"  the fact's detectors never opened in the reply's {MAX_NEW_TOKENS} tokens: {len(never)}")
print(f"  they opened at some step: {len(opened)}, of which vetoed by a shared detector {sum(r['vetoed'] for r in opened)}, "
      f"a first token pushed but not produced {sum('pushed' in r['did'] and 'accepted' not in r['did'] for r in opened)}")
print("  never opened, the best score at the prompt's end:", bins([r["z_end_best"] for r in never]))
print("  never opened, the best score at any step:        ", bins([r["z_max"] for r in never]))
print(f"  never opened, median score at the prompt's end {statistics.median(r['z_end_best'] for r in never):.2f}, at the best step "
      f"{statistics.median(r['z_max'] for r in never):.2f}")
for form in ("plain", "chat"):
    rs = [r for r in never if r["form"] == form]
    if rs:
        print(f"  {form}: {len(rs)} never opened; the own-form detector scores higher than the other form's on "
              f"{sum(r['z_end'].get(form, -1e9) >= max(r['z_end'].values()) for r in rs)}; median at the end {statistics.median(r['z_end_best'] for r in rs):.2f}")
hit = [r for r in out_rows if r["kind"] == "whole"]
print(f"\nthe {len(hit)} replies with the whole answer: opened at the prompt's end {sum(r['z_end_best'] > 0 for r in hit)}, only later in the reply "
      f"{sum(r['z_end_best'] <= 0 for r in hit)}; median score at the end {statistics.median(r['z_end_best'] for r in hit):.2f}")
by_rel = collections.defaultdict(list)
for r in out_rows:
    by_rel[r["relation"]].append(r["z_end_best"])
print("median score at the prompt's end, by relation:", {k: round(statistics.median(v), 2) for k, v in by_rel.items()})
by_tpl = collections.defaultdict(list)
for r in out_rows:
    by_tpl[(r["relation"], r["template"])].append(r)
print("\nthe wordings with the lowest median score at the prompt's end (misses of the facts asked):")
for (rel, tpl), rs in sorted(by_tpl.items(), key=lambda kv: statistics.median(r["z_end_best"] for r in kv[1]))[:14]:
    print(f"  z {statistics.median(r['z_end_best'] for r in rs):6.2f}  nothing {sum(r['kind'] == 'nothing' for r in rs):2d}/{len(rs)}  [{rel}] {tpl[:92]}")

lost = [r for r in out_rows if r["kind"] == "nothing" and r["z_max"] > 0 and "pushed" in r["did"] and "accepted" not in r["did"]]
print(f"\nthe {len(lost)} replies where a first token was pushed and not produced: the best score at any later step")
print("  ", bins([r["z_later_max"] for r in lost]))
print(f"  median {statistics.median(r['z_later_max'] for r in lost):.2f}; by form: plain {sum(r['form'] == 'plain' for r in lost)}, chat {sum(r['form'] == 'chat' for r in lost)}")
print(f"\nother questions about the company ({len(same_rows)} prompts): a stored fact's detector open at the prompt's end on "
      f"{sum(r['z_end'] > 0 for r in same_rows)}, open at some step on {sum(r['z_max'] > 0 for r in same_rows)}, a first token pushed on "
      f"{sum(r['pushes'] > 0 for r in same_rows)}, an answer started on {sum(r['answered'] for r in same_rows)}, the fact's own answer in the reply on "
      f"{sum(r['has_target'] for r in same_rows)}")
print("  the best score at any step:", bins([r["z_max"] for r in same_rows]))

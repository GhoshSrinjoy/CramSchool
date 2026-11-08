"""GRAFT on facts the model already knows: the CounterFact pilot. The plan and rule are in docs/lab-notebook.md ("Updating facts the model
already knows"), written before the run. GPU, about 35 minutes.

40 headquarters edits (CounterFact P159) of facts the unedited Qwen3.5-0.8B gives, all in one GRAFT table, scored with CounterFact's probability
measures and with generation, same-subject, known-facts and drift checks.

Usage: python scripts/counterfact_pilot.py <model folder> <counterfact.json> <screen.jsonl> [out folder] [seed] [facts.json of draws to leave out]
With a seed and an earlier draw, it is the confirmation run, judged by its own rule (lab notebook, "CounterFact confirmation").
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.counterfact import (PILOT_RELATIONS, load_counterfact, logprobs, measures, replies, same_subject_templates, select,
                                          summarize, to_fact, verdict_confirm)
from cramschool.bench.methods import calibrate
from cramschool.bench.scoring import make_baseline
from cramschool.graft.build import build_context, build_parts, build_shared, entry_of
from cramschool.graft.layer import GraftLayer
from cramschool.graft.wordings import REAL_WORLD_CHAT, REAL_WORLD_PLAIN, Wordings, known_facts, norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, PER_RELATION, LAYER, PUSH, MARGIN, REAL_WORLD = 20261108, 40, 20, 2.0, 0.5, "other_relations"


def log(s):
    print(s, flush=True)


def main():
    model_dir, data, screen_path = sys.argv[1:4]
    out_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(ROOT, "runs", "counterfact", "pilot")
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else SEED
    exclude = set()
    if len(sys.argv) > 6:
        with open(sys.argv[6], encoding="utf-8") as fh:
            exclude = {f["cf"]["case_id"] for f in json.load(fh)}
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    records = load_counterfact(data)
    with open(screen_path, encoding="utf-8") as fh:
        screen_rows = [json.loads(line) for line in fh]
    avoid = [p for p, _ in known_facts()] + list(REAL_WORLD_PLAIN) + list(REAL_WORLD_CHAT)
    chosen = select(records, screen_rows, PILOT_RELATIONS, PER_RELATION, seed, avoid, frozenset(exclude))
    w = Wordings.builtin()
    same_t = same_subject_templates(os.path.join(ROOT, "benchmarks", "bench40e.json"))
    facts = [to_fact(r, PILOT_RELATIONS[r["requested_rewrite"]["relation_id"]], w, same_t, f"e{i:02d}") for i, r in enumerate(chosen, 1)]
    with open(os.path.join(out_dir, "facts.json"), "w", encoding="utf-8") as fh:
        json.dump(facts, fh, ensure_ascii=False, indent=1)
    for f in facts:
        log(f"edit {f['id']} {f['relation']:<5} {f['cf']['rewrite']} {f['cf']['old']} -> {f['cf']['new']}")

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = make_baseline(model, tok)
    t0 = time.time()
    before, lp_before = replies(model, tok, facts), logprobs(model, tok, facts)
    log(f"unedited model scored in {time.time() - t0:.0f}s")

    guarded = Wordings(relations=w.relations, generic=w.generic, guard={norm(p) for f in facts for p in f["cf"]["neighbours"]})
    ctx = build_context(facts, guarded, name="cf_pilot", real_world=REAL_WORLD)
    log("calibrating layer 20 on ordinary text")
    mctx = calibrate(model, tok, LAYER)
    entries, t0 = [], time.time()
    for k, f in enumerate(facts, 1):
        path = os.path.join(out_dir, "parts", f"{f['id']}.pt")
        parts = torch.load(path, weights_only=True) if os.path.exists(path) else None
        if parts is None:
            parts = build_parts(model, tok, f, ctx, mctx, LAYER)
            torch.save(parts, path)
        entries.append(entry_of(parts, PUSH))
        log(f"[build {k}/{len(facts)}] {f['id']} {f['subject']} ({time.time() - t0:.0f}s)")
    shared = build_shared(model, tok, ctx, mctx, LAYER)
    layer = GraftLayer(model, tok, entries, shared, mctx["c"], mctx["cbar"], MARGIN, LAYER).attach()
    try:
        t0 = time.time()
        now, lp_now = replies(model, tok, facts), logprobs(model, tok, facts)
        known = baseline.known_check.measure()
        drift = baseline.drift()[0]
        events = dict(layer.events)
        log(f"edited model scored in {time.time() - t0:.0f}s; layer events {events}")
    finally:
        layer.detach()

    rows = []
    with open(os.path.join(out_dir, "pilot.jsonl"), "w", encoding="utf-8") as fh:
        for f in facts:
            m = measures(f, lp_now[f["id"]], now[f["id"]], before[f["id"]])
            m0 = measures(f, lp_before[f["id"]], before[f["id"]], before[f["id"]])
            row = dict(id=f["id"], case_id=f["cf"]["case_id"], relation=f["relation"], subject=f["subject"], old=f["cf"]["old"], new=f["cf"]["new"],
                       measures=m, unedited=m0, replies=now[f["id"]], replies_before=before[f["id"]], logprobs=lp_now[f["id"]],
                       logprobs_before=lp_before[f["id"]])
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"[score] {f['id']} {f['relation']:<5} {f['subject'][:26]:<26} ES {m['es']:.0f} PS {m['ps']:.2f} NS {m['ns']:.2f} | "
                f"says new: rewrite {m['gen_rewrite']:.0f}, reworded {m['gen_para']:.2f} | leaks: neighbours {m['neigh_leak']:.2f}, "
                f"same subject {m['same_leak']:.2f}")
    s = summarize(rows, known["known_lost"], drift)
    s0 = summarize([dict(r, measures=r["unedited"]) for r in rows], 0, 0.0)
    if exclude:
        s["verdict"] = verdict_confirm(s, s0)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(summary=s, unedited=s0, events=events, known=known,
                       options=dict(layer=LAYER, push=PUSH, margin=MARGIN, real_world=REAL_WORLD, seed=seed, per_relation=PER_RELATION, left_out=len(exclude))), fh, indent=1)
    log(f"unedited: ES {s0['es']:.3f} PS {s0['ps']:.3f} NS {s0['ns']:.3f} | says new: rewrite {s0['gen_rewrite']:.3f}, reworded {s0['gen_para']:.3f}")
    for rel, b in s["by_relation"].items():
        log(f"  {rel}: ES {b['es']:.3f} PS {b['ps']:.3f} NS {b['ns']:.3f} | says new: rewrite {b['gen_rewrite']:.3f}, reworded {b['gen_para']:.3f}")
    log(f"CounterFact pilot: ES {s['es']:.3f} PS {s['ps']:.3f} NS {s['ns']:.3f} | says new: rewrite {s['gen_rewrite']:.3f}, reworded "
        f"{s['gen_para']:.3f} | leaks: neighbours {s['neigh_leak']:.3f}, same subject {s['same_leak']:.3f} | known lost {s['known_lost']}, "
        f"drift {s['drift']:.4f} | verdict: {s['verdict']}")


if __name__ == "__main__":
    main()

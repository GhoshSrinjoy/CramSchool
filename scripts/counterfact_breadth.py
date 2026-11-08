"""GRAFT on facts the model already knows, beyond companies: six CounterFact relations about people, countries, places, creative works
and products, all in one table. The plan and rule are in docs/lab-notebook.md ("CounterFact breadth"), written before the run. GPU, about
50 minutes.

Usage: python scripts/counterfact_breadth.py <model folder> <counterfact.json> <screen.jsonl> <relations.json> <heldout.json> [out folder]
       [context sentences] [seed] [facts.json of draws to leave out, comma-separated] [more generic questions.json]
  relations.json: the drafted relations and subject kinds ({"relations": ..., "kinds": ...});
  heldout.json:   held-out same-subject questions per kind ({"kinds": {kind: {"plain": [...], "chat": [...]}}}).
"""
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.counterfact import heldout_templates, load_counterfact, run_edits, select, to_fact
from cramschool.graft.wordings import REAL_WORLD_CHAT, REAL_WORLD_PLAIN, Wordings, known_facts, normalized

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELATIONS = {"P103": "mother_tongue", "P27": "citizenship", "P37": "official_language", "P17": "located_in_country",
             "P364": "original_language", "P178": "developer"}
SEED, PER_RELATION = 20261110, 12


def log(s):
    print(s, flush=True)


def main():
    model_dir, data, screen_path, relations_path, heldout_path = sys.argv[1:6]
    out_dir = sys.argv[6] if len(sys.argv) > 6 else os.path.join(ROOT, "runs", "counterfact", "breadth")
    context = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    seed = int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] != "-" else SEED
    exclude = set()
    for path in (sys.argv[9].split(",") if len(sys.argv) > 9 and sys.argv[9] != "-" else []):
        with open(path, encoding="utf-8") as fh:
            exclude |= {f["cf"]["case_id"] for f in json.load(fh)}
    os.makedirs(out_dir, exist_ok=True)
    wordings = Wordings.builtin().with_file(relations_path)
    if len(sys.argv) > 10:  # more generic questions per kind, added to the drafted ones
        with open(sys.argv[10], encoding="utf-8") as fh:
            more = json.load(fh)["kinds"]
        for kind, spec in more.items():
            k = wordings.kinds[kind]
            seen = {normalized(x) for x in k["generic_plain"] + k["generic_chat"]}
            k["generic_plain"] = k["generic_plain"] + [x for x in spec["generic_plain"] if normalized(x) not in seen]
            k["generic_chat"] = k["generic_chat"] + [x for x in spec["generic_chat"] if normalized(x) not in seen]
            log(f"generic {kind} questions: {len(k['generic_plain'])} plain, {len(k['generic_chat'])} chat")
    missing = [r for r in RELATIONS.values() if r not in wordings.relations]
    if missing:
        raise SystemExit(f"{relations_path} lacks relations {missing}")
    records = load_counterfact(data)
    with open(screen_path, encoding="utf-8") as fh:
        screen_rows = [json.loads(line) for line in fh]
    avoid = [p for p, _ in known_facts()] + list(REAL_WORLD_PLAIN) + list(REAL_WORLD_CHAT)
    chosen = select(records, screen_rows, RELATIONS, PER_RELATION, seed, avoid, frozenset(exclude))
    held = {kind: heldout_templates(heldout_path, kind) for kind in {wordings.relations[r].subject for r in RELATIONS.values()}}
    for kind, items in held.items():  # held-out questions that equal a drafted generic one would not be held out
        drafted = {normalized(x) for x in wordings.kinds[kind]["generic_plain"] + wordings.kinds[kind]["generic_chat"]}
        held[kind] = [h for h in items if normalized(h["template"]) not in drafted]
        log(f"held-out {kind} questions: {len(held[kind])} of {len(items)} kept")
    facts = []
    for i, r in enumerate(chosen, 1):
        rel = RELATIONS[r["requested_rewrite"]["relation_id"]]
        facts.append(to_fact(r, rel, wordings, held[wordings.relations[rel].subject], f"b{i:02d}"))
    with open(os.path.join(out_dir, "facts.json"), "w", encoding="utf-8") as fh:
        json.dump(facts, fh, ensure_ascii=False, indent=1)
    for f in facts:
        log(f"edit {f['id']} {f['relation']:<18} {f['cf']['rewrite']} {f['cf']['old']} -> {f['cf']['new']}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    rows, s, s0 = run_edits(model, tok, facts, wordings, out_dir, context=context, log=log)
    for rel, b in s["by_relation"].items():
        b0 = s0["by_relation"][rel]
        log(f"  {rel:<18} ES {b['es']:.3f} PS {b['ps']:.3f} NS {b['ns']:.3f} (unedited {b0['ns']:.3f}) | says new: rewrite {b['gen_rewrite']:.3f}, "
            f"reworded {b['gen_para']:.3f} | leaks: neighbours {b['neigh_leak']:.3f}, same subject {b['same_leak']:.3f} | {b['verdict']}")
    log(f"unedited: ES {s0['es']:.3f} PS {s0['ps']:.3f} NS {s0['ns']:.3f} | says new: rewrite {s0['gen_rewrite']:.3f}, reworded {s0['gen_para']:.3f}")
    log(f"CounterFact breadth: ES {s['es']:.3f} PS {s['ps']:.3f} NS {s['ns']:.3f} (unedited {s0['ns']:.3f}) | says new: rewrite "
        f"{s['gen_rewrite']:.3f}, reworded {s['gen_para']:.3f} | leaks: neighbours {s['neigh_leak']:.3f}, same subject {s['same_leak']:.3f} | "
        f"known lost {s['known_lost']}, drift {s['drift']:.4f} | verdict: {s['verdict']}")


if __name__ == "__main__":
    main()

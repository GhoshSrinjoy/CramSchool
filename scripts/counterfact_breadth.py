"""GRAFT on facts the model already knows, beyond companies: six CounterFact relations about people, countries, places, creative works
and products, all in one table. Each run's plan and rule are in docs/lab-notebook.md, written before it ran.

Example (revision 2, development edits):
  python scripts/counterfact_breadth.py --model models/Qwen3.5-0.8B --data counterfact.json --screen runs/counterfact/screen.jsonl \\
      --relations research/results/counterfact/breadth/drafted_relations.json --heldout research/results/counterfact/breadth/heldout_questions.json \\
      --more-generic research/results/counterfact/breadth_rev2/more_generic.json --context 4 --out runs/cf/breadth_rev2
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.counterfact import heldout_templates, load_counterfact, run_edits, select, to_fact
from cramschool.graft.wordings import REAL_WORLD_CHAT, REAL_WORLD_PLAIN, Relation, Wordings, known_facts, normalized

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELATIONS = {"P103": "mother_tongue", "P27": "citizenship", "P37": "official_language", "P17": "located_in_country",
             "P364": "original_language", "P178": "developer"}
SEED, PER_RELATION = 20261110, 12


def log(s):
    print(s, flush=True)


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="counterfact.json")
    ap.add_argument("--screen", required=True, help="the screen's rows (scripts/counterfact_screen.py)")
    ap.add_argument("--relations", required=True, help="the drafted relations and subject kinds")
    ap.add_argument("--heldout", required=True, help="held-out same-subject questions per kind")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "counterfact", "breadth"))
    ap.add_argument("--context", type=int, default=0, help="sentences of ordinary text before the subject's plain prompts")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--per-relation", type=int, default=PER_RELATION)
    ap.add_argument("--exclude", nargs="*", default=[], help="facts.json files of earlier draws, whose edits are left out")
    ap.add_argument("--more-generic", help="more generic questions per kind, added to the drafted ones")
    ap.add_argument("--more-wordings", help="more wordings per relation, added to the drafted ones")
    ap.add_argument("--real-world", default="other_relations", choices=["all", "other_relations", "same_kind"])
    ap.add_argument("--fast", action="store_true", help="share readings across facts (a ReadCache)")
    ap.add_argument("--teach-cap", type=int, help="teach answer vectors from at most this many drafted wordings per form")
    ap.add_argument("--only", nargs="*", help="draw edits from these CounterFact relations only (all six stay in the table's wordings)")
    return ap.parse_args()


def merged(base: list[str], extra: list[str], seen: set[str]) -> list[str]:
    """base plus the extra templates not already in `seen` (normalized), in order."""
    return base + [x for x in extra if normalized(x) not in seen and x.count("{s}") == 1]


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    wordings = Wordings.builtin().with_file(a.relations)
    if a.more_generic:
        with open(a.more_generic, encoding="utf-8") as fh:
            for kind, spec in json.load(fh)["kinds"].items():
                k = wordings.kinds[kind]
                seen = {normalized(x) for x in k["generic_plain"] + k["generic_chat"]}
                k["generic_plain"], k["generic_chat"] = merged(k["generic_plain"], spec["generic_plain"], seen), merged(k["generic_chat"], spec["generic_chat"], seen)
                log(f"generic {kind} questions: {len(k['generic_plain'])} plain, {len(k['generic_chat'])} chat")
    if a.more_wordings:
        with open(a.more_wordings, encoding="utf-8") as fh:
            for name, spec in json.load(fh)["relations"].items():
                r = wordings.relations[name]
                seen = {normalized(x) for x in r.plain + r.chat}
                wordings.relations[name] = Relation(name=r.name, plain=merged(r.plain, spec["plain"], seen), chat=merged(r.chat, spec["chat"], seen),
                                                    person=r.person, subject=r.subject)
                log(f"{name} wordings: {len(wordings.relations[name].plain)} plain, {len(wordings.relations[name].chat)} chat")
    missing = [r for r in RELATIONS.values() if r not in wordings.relations]
    if missing:
        raise SystemExit(f"{a.relations} lacks relations {missing}")
    records = load_counterfact(a.data)
    with open(a.screen, encoding="utf-8") as fh:
        screen_rows = [json.loads(line) for line in fh]
    exclude = set()
    for path in a.exclude:
        with open(path, encoding="utf-8") as fh:
            exclude |= {f["cf"]["case_id"] for f in json.load(fh)}
    avoid = [p for p, _ in known_facts()] + list(REAL_WORLD_PLAIN) + list(REAL_WORLD_CHAT)
    drawn = {k: v for k, v in RELATIONS.items() if not a.only or k in a.only}
    chosen = select(records, screen_rows, drawn, a.per_relation, a.seed, avoid, frozenset(exclude))
    held = {kind: heldout_templates(a.heldout, kind) for kind in {wordings.relations[r].subject for r in drawn.values()}}
    for kind, items in held.items():  # held-out questions that equal a drafted generic one would not be held out
        drafted = {normalized(x) for x in wordings.kinds[kind]["generic_plain"] + wordings.kinds[kind]["generic_chat"]}
        held[kind] = [h for h in items if normalized(h["template"]) not in drafted]
        log(f"held-out {kind} questions: {len(held[kind])} of {len(items)} kept")
    facts = []
    for i, r in enumerate(chosen, 1):
        rel = RELATIONS[r["requested_rewrite"]["relation_id"]]
        facts.append(to_fact(r, rel, wordings, held[wordings.relations[rel].subject], f"b{i:02d}"))
    with open(os.path.join(a.out, "facts.json"), "w", encoding="utf-8") as fh:
        json.dump(facts, fh, ensure_ascii=False, indent=1)
    log(f"{len(facts)} edits (seed {a.seed}, {len(exclude)} earlier edits left out); context {a.context}; real world {a.real_world}; fast {a.fast}")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    rows, s, s0 = run_edits(model, tok, facts, wordings, a.out, real_world=a.real_world, context=a.context, fast=a.fast, teach_cap=a.teach_cap,
                            log=log)
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

"""The full CounterFact mix's wordings (lab notebook, "The full CounterFact mix"): the breadth runs' six drafted relations and five subject
kinds, the 18 new relations and the new kind "event" drafted blind, and the existing generic questions minus those an agent marked as asking
for one of the 25 relations; held-out questions it marked are tagged with that relation. Writes <out>/drafted_relations.json (with the CounterFact id of each relation under
"counterfact") and <out>/heldout_questions.json, and prints what was removed.

Usage: python scripts/counterfact_full_wordings.py <drafts folder> <out folder>
  The drafts folder holds drafted_A.json, drafted_B.json, drafted_C.json, event_kind.json, heldout_event.json, existing_questions.json and
  flagged_questions.json.
"""
import collections
import json
import os
import sys

from cramschool.graft.wordings import normalized

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "research", "results", "counterfact")
IDS = {"P159": "hq", "P27": "citizenship", "P178": "developer", "P103": "mother_tongue", "P37": "official_language", "P17": "located_in_country",
       "P364": "original_language", "P140": "religion", "P1412": "languages_spoken", "P413": "position_played", "P641": "sport",
       "P937": "work_location", "P101": "field_of_work", "P39": "position_held", "P20": "place_of_death", "P106": "occupation",
       "P19": "place_of_birth", "P30": "continent", "P131": "located_in_territory", "P36": "capital", "P407": "language_of_work",
       "P449": "original_network", "P495": "country_of_origin", "P176": "manufacturer", "P276": "event_location"}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_templates(name, items, n=None):
    if n is not None and len(items) != n:
        raise SystemExit(f"{name}: {len(items)} templates, {n} expected")
    bad = [t for t in items if t.count("{s}") != 1]
    if bad:
        raise SystemExit(f"{name}: templates without exactly one {{s}}: {bad[:3]}")
    if len({normalized(t) for t in items}) != len(items):
        raise SystemExit(f"{name}: duplicate templates")


def main():
    drafts, out = sys.argv[1], sys.argv[2]
    old = load(os.path.join(R, "breadth", "drafted_relations.json"))
    existing = load(os.path.join(drafts, "existing_questions.json"))
    flagged = load(os.path.join(drafts, "flagged_questions.json"))["flagged"]
    drop = collections.defaultdict(set)
    for f in flagged:
        drop[(f["set"], f["kind"], f["form"])].add(f["template"])
    relations = dict(old["relations"])
    for part in ("A", "B", "C"):
        for name, spec in load(os.path.join(drafts, f"drafted_{part}.json"))["relations"].items():
            if name in relations or name not in IDS.values():
                raise SystemExit(f"drafted_{part}.json: unexpected relation {name!r}")
            check_templates(f"{name} plain", spec["plain"], 30)
            check_templates(f"{name} chat", spec["chat"], 30)
            relations[name] = spec
    missing = [r for r in IDS.values() if r not in relations and r != "hq"]
    if missing:
        raise SystemExit(f"no wordings for {missing}")
    kinds, removed = {}, collections.Counter()
    for kind, spec in old["kinds"].items():
        g = existing["generic"][kind]
        kinds[kind] = dict(names=spec["names"])
        for form in ("plain", "chat"):
            keep = [t for t in g[form] if t not in drop[("generic", kind, form)]]
            removed[("generic", kind, form)] = len(g[form]) - len(keep)
            kinds[kind]["generic_" + form] = keep
    event = load(os.path.join(drafts, "event_kind.json"))["kinds"]["event"]
    check_templates("event generic plain", event["generic_plain"], 60)
    check_templates("event generic chat", event["generic_chat"], 60)
    if len(set(event["names"])) != 40:
        raise SystemExit("event: 40 distinct names expected")
    kinds["event"] = event
    held = {}
    for kind, spec in existing["heldout"].items():  # a held-out question that asks for a relation is kept, tagged, and left out only for
        held[kind] = dict(plain=list(spec["plain"]), chat=list(spec["chat"]), tags={})  # edits of that relation (as the pilot did)
        for f in flagged:
            if f["set"] == "heldout" and f["kind"] == kind:
                held[kind]["tags"][f["template"]] = f["relation"]
                removed[("heldout", kind, f["form"])] += 1
    ev = load(os.path.join(drafts, "heldout_event.json"))["kinds"]["event"]
    check_templates("event held-out plain", ev["plain"], 12)
    check_templates("event held-out chat", ev["chat"], 12)
    held["event"] = ev
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "drafted_relations.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(counterfact=IDS, relations=relations, kinds=kinds), fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "heldout_questions.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(kinds=held), fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "removed_questions.json"), "w", encoding="utf-8") as fh:
        json.dump(flagged, fh, ensure_ascii=False, indent=1)
    print(f"{len(relations)} relations (+ hq built in); kinds {sorted(kinds)}")
    for (s, kind, form), n in sorted(removed.items()):
        if s == "generic":
            print(f"  generic  {kind:8} {form:5}: removed {n:2}, kept {len(kinds[kind]['generic_' + form])}")
        else:
            print(f"  held-out {kind:8} {form:5}: tagged {n:2} of {len(held[kind][form])}")
    by_rel = collections.Counter(f["relation"] for f in flagged)
    print("removed, by the relation they ask for:", dict(by_rel.most_common()))


if __name__ == "__main__":
    main()

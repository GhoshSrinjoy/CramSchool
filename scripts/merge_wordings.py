"""Add new plain wordings to a relations file: a base file (drafted relations, kinds, CounterFact ids) plus one or more files of
{"relations": {name: {"plain": [...]}}}. New wordings that repeat a current one (normalized) are dropped. A built-in relation (hq) is
written out in full, its built-in wordings first. Lab notebook, "Phrasing coverage, with the big model as editor".

Usage: python scripts/merge_wordings.py <base relations.json> <out.json> <new wordings.json> [more new wordings.json ...]
"""
import json
import sys

from cramschool.graft.wordings import Wordings, normalized


def main():
    base_path, out_path, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    with open(base_path, encoding="utf-8") as fh:
        base = json.load(fh)
    builtin = Wordings.builtin().relations
    added = {}
    for path in extra:
        with open(path, encoding="utf-8") as fh:
            for name, spec in json.load(fh)["relations"].items():
                if name not in base["relations"]:
                    if name not in builtin:
                        raise SystemExit(f"{path}: unknown relation {name!r}")
                    r = builtin[name]
                    base["relations"][name] = dict(subject=r.subject, person=r.person, plain=list(r.plain), chat=list(r.chat))
                rel = base["relations"][name]
                seen = {normalized(t) for t in rel["plain"] + rel["chat"]}
                new = [t for t in spec["plain"] if t.count("{s}") == 1 and normalized(t) not in seen]
                rel["plain"] = rel["plain"] + new
                added[name] = added.get(name, 0) + len(new)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(base, fh, ensure_ascii=False, indent=1)
    print(f"{sum(added.values())} new plain wordings over {len(added)} relations: {added}")


if __name__ == "__main__":
    main()

"""Step 11 (lab notebook): the editor's final wordings, from its two rounds: round one's, without those it dropped in round two, plus round
two's. Usage: python scripts/editor_merge_rounds.py <round 1 json> <round 2 json> <out json>"""
import json
import sys

from cramschool.graft.wordings import normalized

one, two, out = sys.argv[1:4]
with open(one, encoding="utf-8") as fh:
    r1 = json.load(fh)["relations"]
with open(two, encoding="utf-8") as fh:
    r2 = json.load(fh)["relations"]
final = {}
for rel, first in r1.items():
    second = r2.get(rel, {})
    drop = {normalized(t) for t in second.get("drop", [])}
    unknown = [t for t in second.get("drop", []) if normalized(t) not in {normalized(x) for x in first["plain"] + first["chat"]}]
    if unknown:
        raise SystemExit(f"{rel}: dropped wordings that round one did not write: {unknown}")
    final[rel] = {form: [t for t in first[form] if normalized(t) not in drop] + list(second.get(form, [])) for form in ("plain", "chat")}
    for t in final[rel]["plain"] + final[rel]["chat"]:
        if t.count("{s}") != 1 or t != t.strip():
            raise SystemExit(f"{rel}: bad wording {t!r}")
    print(f"{rel}: round one {len(first['plain'])} plain and {len(first['chat'])} chat; dropped {len(drop)}; added {len(second.get('plain', []))} and "
          f"{len(second.get('chat', []))}; final {len(final[rel]['plain'])} and {len(final[rel]['chat'])}")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(dict(note="Wordings an editor model wrote for forms the question detectors missed on bench40e and bench40f (lab notebook, step 11). "
                        "{s} is the company.", relations=final), fh, ensure_ascii=False, indent=1)

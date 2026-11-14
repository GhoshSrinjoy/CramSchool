"""Step 11, round 2 (lab notebook): what the editor is shown after its first round. From the development runs made with its wordings: the
question forms that still miss, and every question that got another fact's answer, with the relation that answered. No model needed.

Usage: python scripts/editor_round_input.py <round 1 wordings json> <out json> <bench json>=<rows jsonl> [<bench json>=<rows jsonl> ...]
"""
import collections
import json
import sys

from cramschool.gates import mentions
from cramschool.graft.wordings import Wordings, normalized

ROUND1, OUT = sys.argv[1:3]
runs = [a.split("=", 1) for a in sys.argv[3:]]
with open(ROUND1, encoding="utf-8") as fh:
    mine = json.load(fh)["relations"]
wordings = Wordings.builtin().with_more(mine)
names = dict(owner="the person who owns the company", ceo="the company's chief executive", founder="the person who founded the company",
             hq="the city of the company's head office")
missed = {r: dict(plain=collections.Counter(), chat=collections.Counter()) for r in names}
asked = {r: dict(plain=collections.Counter(), chat=collections.Counter()) for r in names}
wrong = {r: [] for r in names}
for bench_path, rows_path in runs:
    with open(bench_path, encoding="utf-8") as fh:
        facts = {f["id"]: f for f in json.load(fh)["facts"]}
    by_target = {f["target"]: f for f in facts.values()}
    with open(rows_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for row in rows:
        f = facts[row["id"]]
        t = f["test"]
        templates = [(x, "plain") for x in t["templates_plain"]] + [(x, "chat") for x in t["templates_chat"]]
        for (template, form), reply in zip(templates, row["answers"]["recall"]):
            asked[f["relation"]][form][template] += 1
            if not mentions(reply, f["target"]):
                missed[f["relation"]][form][template] += 1
        same = t["same_plain"] + t["same_chat"]
        for hit in row.get("cross_where") or []:  # another stored fact's answer in a reply: which relation answered which question
            if hit["group"] != "same":
                continue
            for target in hit["targets"]:
                g = by_target.get(target)
                if g is not None:
                    wrong[g["relation"]].append(dict(question=same[hit["index"]].replace(f["subject"], "{s}"),
                                                     what_happened=f"the {g['relation']} fact answered this question, which does not ask for the {g['relation']}"))
        if row["measures"].get("same", 0) > 0:  # the fact's own answer in replies to other questions about its company
            before_same = row["answers"]["same"]
            for question, reply in zip(same, before_same):
                if mentions(reply, f["target"]):
                    wrong[f["relation"]].append(dict(question=question.replace(f["subject"], "{s}"),
                                                     what_happened=f"the {f['relation']} fact answered this question, which does not ask for the {f['relation']}"))
doc = {}
for r in names:
    rel = wordings.relations[r]
    doc[r] = dict(asks_for=names[r], your_plain=mine[r]["plain"], your_chat=mine[r]["chat"],
                  still_missed_plain=[dict(form=t, failed=n, asked=asked[r]["plain"][t], taught=normalized(t) in {normalized(x) for x in rel.plain})
                                      for t, n in missed[r]["plain"].most_common() if n >= 2],
                  still_missed_chat=[dict(form=t, failed=n, asked=asked[r]["chat"][t], taught=normalized(t) in {normalized(x) for x in rel.chat})
                                     for t, n in missed[r]["chat"].most_common() if n >= 2],
                  wrongly_answered=wrong[r])
    print(f"{r}: still missed {len(doc[r]['still_missed_plain'])} plain and {len(doc[r]['still_missed_chat'])} chat forms (2 or more facts each); "
          f"questions it answered wrongly: {len(wrong[r])}")
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(dict(note="{s} stands for the company's name. 'failed' of 'asked': on how many facts the form still got no answer. 'taught': the "
                        "form is one of the wordings the system learns from, and still fails on those facts.", relations=doc), fh, ensure_ascii=False, indent=1)
print("written", OUT)

"""Count an editor session from its saved calls, not from the editor's report.

    python scripts/count_editor_session.py research/results/editor_any_subject [--work RUN_FOLDER]

The folder holds `facts.json` (the facts and the rule, written before the session) and `calls/`, one JSON file per tool call as the
server returned it; a fact's final check is the file whose name ends in `_final_<id>.json`. With `--work`, the folder the session
ran in, the table's history and the weights' history are also read from their files and checked whole (they are too large to keep
in the repository); without it, that part of an existing `summary.json` is kept as it is.

Writes `summary.json` into the folder. No model is loaded."""
import argparse
import glob
import json
import os
import re

from cramschool.tools import GraftTools

said = GraftTools._said


def count(rows, key):
    return f"{sum(r[key][0] for r in rows)} of {sum(r[key][1] for r in rows)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--work", help="the folder the session ran in, with table.graft and weights.history")
    args = ap.parse_args()
    spec = json.load(open(os.path.join(args.folder, "facts.json"), encoding="utf-8"))
    facts = spec["facts"]

    calls = sorted(glob.glob(os.path.join(args.folder, "calls", "*.json")))
    by_tool, errors = {}, []
    for path in calls:
        name = os.path.basename(path)
        tool = re.sub(r"_final_.*$", "", re.sub(r"^\d+_", "", name).replace(".json", ""))
        tool = tool if tool.startswith("graft_") else "graft_" + tool
        by_tool[tool] = by_tool.get(tool, 0) + 1
        data = json.load(open(path, encoding="utf-8"))
        if isinstance(data, dict) and "error" in data:
            errors.append([name, str(data["error"])[:160]])

    rows, leaks, cross = {}, [], []
    for fid, f in facts.items():
        found = sorted(glob.glob(os.path.join(args.folder, "calls", f"*_final_{fid}.json")))
        if not found:
            rows[fid] = dict(f, final=None)
            continue
        d = json.load(open(found[-1], encoding="utf-8"))
        first = d["fact"]["answer"].split()[0]
        others, looks, recall = d["other_replies"], d["lookalike_replies"], d["recall_replies"]
        for r in others:
            if r["leaked"]:
                leaks.append(dict(fact=fid, question=r["question"], reply=r["reply"][:80],
                                  begins_with_the_answer=r["reply"].strip().lower().startswith(first.lower())))
            # another stored fact's answer, brought in by the table; the session's tool did not report these
            for oid, o in facts.items():
                word = o["answer"].split()[0]
                if (o["subject"], o["relation"]) != (f["subject"], f["relation"]) and said(r["reply"], word) and not said(r["without"], word):
                    cross.append(dict(fact=fid, question=r["question"], answer_of=oid, same_subject=o["subject"] == f["subject"],
                                      reply=r["reply"][:80]))
        rows[fid] = dict(f, final=os.path.basename(found[-1]), stored_answer=d["fact"]["answer"], good=bool(d["good"]),
                         recall=[sum(r["recalled"] for r in recall), len(recall)],
                         leaks_other_questions=[sum(r["leaked"] for r in others), len(others)],
                         leaks_lookalikes=[sum(r["leaked"] for r in looks), len(looks)],
                         said_without_the_table=sum(bool(r["said_without_the_table"]) for r in recall))

    done = [r for r in rows.values() if r.get("final")]

    def group(pick):
        some = [r for r in done if pick(r)]
        return dict(good=f"{sum(r['good'] for r in some)} of {len(some)}", recall=count(some, "recall"),
                    leaks_other_questions=count(some, "leaks_other_questions"), leaks_lookalikes=count(some, "leaks_lookalikes"))

    summary = dict(
        note=spec.get("note", ""), rule=spec.get("rule", ""), calls=len(calls), calls_by_tool=dict(sorted(by_tool.items())), errors=errors,
        checks_complete=all(r.get("final") and r["recall"][1] == 8 and r["leaks_other_questions"][1] == 4 and r["leaks_lookalikes"][1] == 3
                            for r in rows.values()),
        all=group(lambda r: True),
        by_kind={k: group(lambda r, k=k: r["kind"] == k) for k in dict.fromkeys(r["kind"] for r in done)},
        by_answer_type={t: group(lambda r, t=t: r["answer_type"] == t) for t in dict.fromkeys(r["answer_type"] for r in done)},
        leaks=dict(count=len(leaks), begin_with_the_answer=sum(x["begins_with_the_answer"] for x in leaks), replies=leaks),
        other_stored_answers=dict(
            note=("Replies to the other questions into which the table brought another stored fact's answer. The check of the session "
                  "did not count these. Where the question asks for that other fact, the answer is right; the replies are listed to be read."),
            count=len(cross), replies=cross),
        facts=rows)

    # the replies of the weights' versions: how many hold their fact's answer
    asked = {}
    for path in calls:
        if "weights_ask" not in os.path.basename(path):
            continue
        d = json.load(open(path, encoding="utf-8"))
        replies = d.get("replies", [])
        if len(replies) == len(facts) and all(f["subject"].lower() in r["question"].lower() for f, r in zip(facts.values(), replies)):
            asked[d["version"]] = dict(file=os.path.basename(path), hold_the_answer=sum(said(r["reply"], f["answer"]) for f, r in zip(facts.values(), replies)),
                                       of=len(replies), which=[fid for (fid, f), r in zip(facts.items(), replies) if said(r["reply"], f["answer"])])
    summary["weights_ask"] = asked

    out = os.path.join(args.folder, "summary.json")
    if args.work:
        from cramschool import weight_history as wh
        from cramschool.graft import history

        table, hist = os.path.join(args.work, "table.graft"), os.path.join(args.work, "weights.history")
        summary["histories_from_their_files"] = dict(
            table_versions=[dict(id=v.id, facts=v.facts, message=v.message, of=v.of) for v in history.versions(table)],
            table_history_whole=history.is_intact(table),
            weights_versions=[dict(id=v.id, facts=v.facts, tensors=len(v.tensors), table=v.table, of=v.of, message=v.message) for v in wh.versions(hist)],
            weights_history_whole=wh.is_intact(hist))
    elif os.path.exists(out):
        summary["histories_from_their_files"] = json.load(open(out, encoding="utf-8")).get("histories_from_their_files")
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("calls:", len(calls), "| errors:", len(errors), "| checks complete:", summary["checks_complete"])
    print("all:", summary["all"])
    for name in ("by_kind", "by_answer_type"):
        for k, v in summary[name].items():
            print(f"  {k}: {v}")
    print("leaks:", len(leaks), "| begin with the answer:", summary["leaks"]["begin_with_the_answer"], "| another stored answer brought in:", len(cross),
          f"({sum(x['same_subject'] for x in cross)} about the same subject)")
    print("weights ask:", {k: f"{v['hold_the_answer']} of {v['of']}" for k, v in asked.items()})
    print("histories:", {k: v for k, v in (summary.get("histories_from_their_files") or {}).items() if k.endswith("whole")})


if __name__ == "__main__":
    main()

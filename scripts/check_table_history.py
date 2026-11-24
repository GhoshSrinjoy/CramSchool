"""The check of a table's history on the real model: the package's own path, end to end. A small table is built, recorded as two
versions, loaded back, reverted and merged, and asked the same questions each time. GPU, about 8 minutes. Forward passes only.

Usage: python scripts/check_table_history.py <model folder> <table out folder> <out json>

Checks, each written to the out file:
1. version v1, loaded back from the history while the folder holds v2, gives the replies the table gave when v1 was recorded;
2. a revert to v1, and then to v2, gives the table the replies of each, and both are recorded as new versions;
3. a fact merged from a second table, whose calibration was measured on its own, gives the replies it gave in its own table, and the
   facts that were there before give theirs;
4. the history is whole, and a bisect finds the version at which a reply changed.

The questions are the facts' own, reworded, about a look-alike company, and general ones. What the replies say is not judged here:
only that they are the same replies, character for character.
"""
import json
import sys
import time

from cramschool import Graft, GraftOptions
from cramschool.__main__ import _load
from cramschool.graft import history

QUESTIONS = ["Who is the CEO of Vorantis Biotech?", "Who runs Vorantis Biotech?", "Tell me the name of Vorantis Biotech's chief executive.",
             "Where is Vorantis Biotech headquartered?", "In which city is the head office of Vorantis Biotech?",
             "Who founded Nimbresco Software?", "Who started Nimbresco Software?", "Who is the CEO of Vorantis Biotechnology?",
             "How many people work at Vorantis Biotech?", "What is the capital of France?"]
MERGED = ["Where is Moqueldar Chemicals headquartered?", "In which city is Moqueldar Chemicals based?", "Who founded Moqueldar Chemicals?"]


def replies(table, questions=QUESTIONS):
    return [table.ask(q) for q in questions]


def main(model_dir: str, path: str, out: str) -> int:
    say = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)  # noqa: E731
    model, tok = _load(model_dir, "bfloat16")
    result = {}

    graft = Graft(model, tok, options=GraftOptions(fast=True), progress=say)
    graft.add("Vorantis Biotech", "ceo", "Ingrid Solberg")
    graft.add("Vorantis Biotech", "hq", "Tartu")
    graft.commit(path, "two facts about Vorantis")
    r1 = replies(graft)
    graft.add("Nimbresco Software", "founder", "Tharindu Ekanayake")
    graft.add("Vorantis Biotech", "ceo", "Kenji Aoyama")
    graft.commit(path, "a founder, and another CEO")
    r2 = replies(graft)
    result["questions"], result["v1_replies"], result["v2_replies"] = QUESTIONS, r1, r2
    result["replies_that_differ_between_v1_and_v2"] = sum(a != b for a, b in zip(r1, r2))

    old = Graft.load(path, model, tok, version="v1")
    result["1_v1_loaded_back_gives_v1_replies"] = replies(old) == r1
    result["1_folder_left_at_v2"] = history.is_committed(path) and history.versions(path)[-1].id == "v2"

    back = graft.revert(path, "v1")
    result["2_revert_to_v1"] = dict(version=back.id, of=back.of, same_replies=replies(graft) == r1, facts=[[f.subject, f.relation, f.answer] for f in graft.facts])
    again = graft.revert(path, "v2")
    result["2_revert_to_v2"] = dict(version=again.id, of=again.of, same_replies=replies(graft) == r2)
    result["diff_v1_v2"] = history.diff_text(history.diff(path, "v1", "v2")).splitlines()

    other = Graft(model, tok, options=GraftOptions(fast=True), progress=say)  # its own calibration, measured again
    other.add("Moqueldar Chemicals", "hq", "Oradea")
    alone = replies(other, MERGED)
    report = graft.merge(other)
    result["3_merge"] = dict(report=report, merged_fact_same_replies=replies(graft, MERGED) == alone, earlier_facts_same_replies=replies(graft) == r2,
                             questions=MERGED, in_its_own_table=alone, in_the_merged_table=replies(graft, MERGED))
    graft.commit(path, "a third company, merged")
    result["versions"] = [dict(id=v.id, facts=v.facts, message=v.message, of=v.of) for v in history.versions(path)]
    result["4_history_whole"] = history.is_intact(path)
    try:  # the test: the first question gets the reply it got at v1. v3 is v1 again, so the search starts after it
        found = graft.bisect(path, lambda table: table.ask(QUESTIONS[0]) == r1[0], good="v3")
        result["4_bisect"] = dict(first_bad=found["first_bad"]["id"], last_good=found["last_good"]["id"], checked=found["checked"])
    except ValueError as error:
        result["4_bisect"] = dict(error=str(error))

    passed = all([result["1_v1_loaded_back_gives_v1_replies"], result["1_folder_left_at_v2"], result["2_revert_to_v1"]["same_replies"],
                  result["2_revert_to_v2"]["same_replies"], result["3_merge"]["merged_fact_same_replies"], result["3_merge"]["earlier_facts_same_replies"],
                  result["4_history_whole"], result["replies_that_differ_between_v1_and_v2"] > 0])
    result["passed"] = passed
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print("CHECK", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))

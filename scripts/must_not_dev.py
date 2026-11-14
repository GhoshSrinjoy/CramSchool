"""Step 12's development check (lab notebook): teaching the CEO relation what it must not answer, on the editor's own questions.

Step 11's editor asked 8 other-job questions over its three checks. The 5 of its first two checks are taught as must-not-answer; the 3 of
its last check are the test, on the table's CEO companies. The cost is read on the last check's CEO recall questions, which were never
taught. The table is measured as saved, then with the wordings taught and its CEO facts built again, and saved to the out folder.

Forward passes only. GPU, about 6 minutes.

Usage: python scripts/must_not_dev.py <model folder> <table folder> <the editor's last check jsonl> <out folder>
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.gates import mentions
from cramschool.graft.build import fact_detectors
from cramschool.model_access import Prompt, generate_batch, render_chat

TAUGHT = dict(plain=["{s}'s head of marketing is", "The head of research at {s} is"],
              chat=["Who is the chief technology officer of {s}?", "Who is the chief financial officer of {s}?", "Who is the spokesperson for {s}?"])
TEST = ["Who is the general counsel of {s}?", "Who is the head of human resources at {s}?", "The chief engineer of {s} is"]
MAX_NEW_TOKENS, BATCH = 48, 32
LEAK_BAR, RECALL_BAR = 0.5, 0.9


def is_chat(q: str) -> bool:
    return q.rstrip().endswith(("?", "."))  # the tools' rule (check_fact)


def main():
    model_dir, table_dir, check_path, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    with open(check_path, encoding="utf-8") as fh:
        checks = [json.loads(line) for line in fh if line.strip()]
    recall_q = {r["subject"]: r["args"]["new_questions"] for r in checks if r["relation"] == "ceo"}
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    graft = Graft.load(table_dir, model, tok)
    facts = [f for f in graft.facts if f.relation == "ceo"]
    groups = dict(test=[(f, q.format(s=f.subject)) for f in facts for q in TEST],
                  taught=[(f, q.format(s=f.subject)) for f in facts for q in TAUGHT["plain"] + TAUGHT["chat"]],
                  recall=[(f, q) for f in facts for q in recall_q[f.subject]])

    def measure():
        out = {}
        with graft.attached():
            for name, items in groups.items():
                prompts = [render_chat(tok, q) if is_chat(q) else Prompt(q) for _, q in items]
                replies = generate_batch(model, tok, prompts, MAX_NEW_TOKENS, batch_size=BATCH)
                out[name] = [dict(subject=f.subject, question=q, reply=r, hit=mentions(r, f.answer)) for (f, q), r in zip(items, replies)]
        return out

    before = measure()
    t0 = time.time()
    added = graft.add_avoid("ceo", plain=TAUGHT["plain"], chat=TAUGHT["chat"])
    built = graft.rebuild("ceo")
    print(f"taught {added}; {built} facts built again in {time.time() - t0:.0f}s", flush=True)
    after = measure()
    graft.save(os.path.join(out_dir, "table.graft"))
    count = lambda rows: sum(r["hit"] for r in rows)  # noqa: E731
    kept = sum(a["hit"] for b, a in zip(before["recall"], after["recall"]) if b["hit"])
    summary = dict(table=table_dir, taught=TAUGHT, test=TEST,
                   leak_test=dict(before=count(before["test"]), after=count(after["test"]), prompts=len(groups["test"])),
                   leak_taught=dict(before=count(before["taught"]), after=count(after["taught"]), prompts=len(groups["taught"])),
                   recall=dict(before=count(before["recall"]), after=count(after["recall"]), prompts=len(groups["recall"]), kept=kept),
                   detectors={f.subject: [d["form"] for d in fact_detectors(e["parts"])]
                              for f, e in zip(graft.facts, graft._facts) if f.relation == "ceo"})
    lt, rc = summary["leak_test"], summary["recall"]
    summary["leak_bar_met"] = lt["after"] <= LEAK_BAR * lt["before"]
    summary["recall_bar_met"] = kept >= RECALL_BAR * rc["before"]
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "replies.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(before=before, after=after), fh, ensure_ascii=False, indent=1)
    print(f"leak on the 3 untaught jobs: {lt['before']} of {lt['prompts']} before, {lt['after']} after (bar: half or less) "
          f"{'MET' if summary['leak_bar_met'] else 'MISSED'}")
    print(f"leak on the 5 taught jobs: {summary['leak_taught']['before']} of {summary['leak_taught']['prompts']} before, "
          f"{summary['leak_taught']['after']} after")
    print(f"recall on the last check's questions: {rc['before']} of {rc['prompts']} before, {rc['after']} after; of those answered before, "
          f"{kept} still answered (bar: 0.9) {'MET' if summary['recall_bar_met'] else 'MISSED'}")
    print(f"detectors left per fact: {summary['detectors']}")


if __name__ == "__main__":
    main()

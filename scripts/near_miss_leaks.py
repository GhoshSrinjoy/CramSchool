"""Questions close to a relation that are not it (lab notebook, step 12): does a table's fact answer them? Each question asks who holds a
named job at a company; it is asked about the company of every fact of one relation, and a leak is a reply that holds that fact's answer.
The table is asked as the inserted layer, or with --grafted as neurons compiled into the model with the name switch on.

Forward passes only. GPU, about 2 minutes for 240 prompts.

Usage: python scripts/near_miss_leaks.py <model folder> <table folder> <questions json> <out jsonl> [--relation ceo] [--grafted]
       [--taught <table folder>]

<questions json>   {"questions": [{"job": "...", "form": "plain" | "chat", "text": "... {s} ..."}]}
--taught           a table whose must-not-answer wordings for the relation were taught: a question equal to one of them is left out of
                   the count, and the leak is also given for the jobs none of them names
"""
import collections
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.gates import mentions
from cramschool.graft.wordings import normalized
from cramschool.model_access import Prompt, generate_batch, render_chat

MAX_NEW_TOKENS, BATCH = 48, 32  # the length of a reply in the tools' check (Graft.ask)


def taught_of(table: str, relation: str) -> list[str]:
    with open(os.path.join(table, "graft.json"), encoding="utf-8") as fh:
        rel = json.load(fh)["relations"][relation]
    return list(rel.get("avoid_plain", [])) + list(rel.get("avoid_chat", []))


def main():
    args = sys.argv[1:]
    model_dir, table_dir, questions_path, out_path = args[:4]
    relation = args[args.index("--relation") + 1] if "--relation" in args else "ceo"
    taught = taught_of(args[args.index("--taught") + 1], relation) if "--taught" in args else []
    with open(questions_path, encoding="utf-8") as fh:
        questions = json.load(fh)["questions"]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    graft = Graft.load(table_dir, model, tok)
    facts = [f for f in graft.facts if f.relation == relation]
    items = [(f, q) for f in facts for q in questions]
    prompts = [render_chat(tok, q["text"].format(s=f.subject)) if q["form"] == "chat" else Prompt(q["text"].format(s=f.subject)) for f, q in items]
    if "--grafted" in args:
        from cramschool.graft.grafted import apply_layer, compile_layer
        from cramschool.graft.grafted_switch import switch_for

        compiled = compile_layer(model, tok, graft.grafted_facts(), graft.layer, graft.calibration(), forms=("plain", "chat"),
                                 push=graft.options.push, saturate=True, where="last", track=True)
        apply_layer(model, compiled)
        switch = switch_for(model, tok, compiled)
        replies = generate_batch(model, tok, prompts, MAX_NEW_TOKENS, batch_size=BATCH)
        switch.detach()
        how = f"compiled into {len(compiled['neurons'])} neurons, with the switch"
    else:
        with graft.attached():
            replies = generate_batch(model, tok, prompts, MAX_NEW_TOKENS, batch_size=BATCH)
        how = "the inserted layer"
    equal = {normalized(t) for t in taught}
    named = " ".join(taught).lower()
    rows = []
    for (f, q), reply in zip(items, replies):
        rows.append(dict(subject=f.subject, answer=f.answer, job=q["job"], form=q["form"], question=q["text"].format(s=f.subject), reply=reply,
                         leaked=mentions(reply, f.answer), first_name=f.answer.split()[0].lower() in reply.lower(),
                         taught=normalized(q["text"]) in equal, job_named=bool(taught) and q["job"].lower() in named))
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def rate(rs):
        return f"{sum(r['leaked'] for r in rs)} of {len(rs)} ({sum(r['leaked'] for r in rs) / max(len(rs), 1):.3f})"

    kept = [r for r in rows if not r["taught"]]
    print(f"{table_dir}, {how}: {len(facts)} {relation} facts, {len(questions)} questions, {len(rows)} prompts; "
          f"{len(rows) - len(kept)} left out as taught")
    print(f"leak (the whole answer in the reply): {rate(kept)} | plain {rate([r for r in kept if r['form'] == 'plain'])} | "
          f"chat {rate([r for r in kept if r['form'] == 'chat'])} | the first name alone or more: "
          f"{sum(r['first_name'] for r in kept)} of {len(kept)}")
    if taught:
        print(f"on the jobs no taught wording names: {rate([r for r in kept if not r['job_named']])} | on the jobs one names: "
              f"{rate([r for r in kept if r['job_named']])}")
    by_fact = collections.Counter(r["subject"] for r in kept if r["leaked"])
    by_job = collections.Counter(r["job"] for r in kept if r["leaked"])
    print(f"facts with a leak: {len(by_fact)} of {len(facts)} {dict(by_fact)}")
    print(f"jobs with a leak: {len(by_job)} of {len({q['job'] for q in questions})}; most: {by_job.most_common(6)}")


if __name__ == "__main__":
    main()

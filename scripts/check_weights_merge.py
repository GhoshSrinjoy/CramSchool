"""The check of the merge at the weights on the real model. Two tables are built from a bench's facts, each is exported as a grafted
model, and the two grafted models are merged at the weights, with no table. The merged checkpoint is compared, tensor for tensor,
with one export of the two tables merged. GPU, about 20 minutes. Forward passes only.

Usage: python scripts/check_weights_merge.py <model folder> <bench file> <work folder> <out json>

Two cases, both from the same two tables of 18 facts:
- "fits": 3 facts in each model, which fit the last layer together;
- "over": 18 facts in each, which fit the last layer one by one and go over its budget together, so that later answer tokens have
  to move to the other layer, written from the vectors each export kept beside its weights.

For each case the out file says:
1. whether the merged checkpoint is equal to the one export, and if not, how many tensors and values differ and by how much;
2. whether grafted.json names the same neurons for the same facts, and whether the kept later vectors are the same bytes;
3. how many facts answer from the merged folder loaded from disk, and from the one export loaded from disk, on one taught question
   each (a check that the weights answer, not a bench);
4. for "fits": that the merged model is recorded in a history of the weights, that the history is whole, and that the loaded
   model set to that version gives the replies of the folder loaded from disk.

The out file holds no path of this computer.
"""
import gc
import json
import os
import sys
import time

import torch
from safetensors.torch import load_file

from cramschool import Graft, GraftOptions, load_grafted, merge_grafted_models, render_chat
from cramschool import weight_history as wh
from cramschool.__main__ import _load
from cramschool.datasets import load_facts
from cramschool.gates import generate_batch
from cramschool.graft.grafted import LATER_FILE


def tensors(folder: str) -> dict:
    out = {}
    for name in sorted(os.listdir(folder)):
        if name.endswith(".safetensors"):
            out.update(load_file(os.path.join(folder, name)))
    return out


def compare(a: str, b: str) -> dict:
    ta, tb = tensors(a), tensors(b)
    differs = {k: dict(values=int((ta[k] != tb[k]).sum()), of=ta[k].numel(), largest=float((ta[k].double() - tb[k].double()).abs().max()))
               for k in ta if not torch.equal(ta[k], tb[k])}
    return dict(equal=sorted(ta) == sorted(tb) and not differs, tensors=len(ta), tensors_that_differ=differs)


def spec(folder: str) -> dict:
    with open(os.path.join(folder, "grafted.json"), encoding="utf-8") as fh:
        return json.load(fh)


def same_bytes(a: str, b: str, name: str) -> bool | None:
    pa, pb = os.path.join(a, name), os.path.join(b, name)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None if not (os.path.exists(pa) or os.path.exists(pb)) else False
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        return fa.read() == fb.read()


def asked(folder: str, questions: list[tuple[str, str]]) -> list[dict]:
    """Each question's reply from a grafted model loaded from disk, and whether it holds the answer."""
    model, tok, switch = load_grafted(folder)
    try:
        replies = generate_batch(model, tok, [render_chat(tok, q) for q, _ in questions], 32, batch_size=1)
    finally:
        switch.detach()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return [dict(question=q, answer=a, reply=r, answered=a.lower() in r.lower()) for (q, a), r in zip(questions, replies)]


def main(base: str, bench: str, work: str, out: str) -> int:
    say = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)  # noqa: E731
    facts = load_facts(bench)[:36]
    model, tok = _load(base, "bfloat16")
    result = dict(bench=os.path.basename(bench), facts=len(facts), cases={})

    saved = []
    for name, part in (("mine", facts[:18]), ("theirs", facts[18:])):  # two tables, built one after the other, each on its own
        table = Graft(model, tok, options=GraftOptions(fast=True), progress=say)
        for f in part:
            table.add(f.subject, f.relation, f.answer)
        table.detach()
        saved.append(table.save(os.path.join(work, f"{name}.graft")))
        del table
    probe = Graft.load(saved[0], model, tok)
    probe.detach()
    result["table"] = dict(layer=probe.layer, facts_each=18, write=probe.options.last_layer)
    del probe

    def tables(keep: int) -> list[Graft]:
        """The two saved tables, loaded again, with only their first `keep` facts."""
        found = []
        for path in saved:
            table = Graft.load(path, model, tok, progress=say)
            for f in table.facts[keep:]:
                table.remove(f.subject, f.relation)
            table.detach()
            found.append(table)
        return found

    for case, keep in (("fits", 3), ("over", 18)):
        say(f"case {case}: {keep} facts in each model")
        folder = lambda n: os.path.join(work, f"{case}_{n}")  # noqa: E731
        mine, theirs = tables(keep)
        questions = [(t.wordings.relations[f.relation].chat[0].format(s=f.subject), f.answer) for t in (mine, theirs) for f in t.facts]
        reports = [t.export_grafted(folder(n), base_dir=base) for t, n in ((mine, "mine"), (theirs, "theirs"))]
        del reports
        if case == "fits":  # merged and recorded in one call
            wh.commit(folder("history"), folder("mine"), "mine", base_dir=base)
            done = wh.merge(folder("history"), folder("mine"), folder("theirs"), folder("merged"), model, tok, "both, merged at the weights",
                            base_dir=base, progress=say)
            merged, version = done["merged"], done["version"]
        else:
            merged, version = merge_grafted_models(model, tok, base, folder("mine"), folder("theirs"), folder("merged"), progress=say), None
        mine.merge(theirs)                                                      # the reference: the two tables merged, exported once
        mine.export_grafted(folder("once"), base_dir=base)
        weights = compare(folder("merged"), folder("once"))
        sm, so = spec(folder("merged")), spec(folder("once"))
        from_merged, from_once = asked(folder("merged"), questions), asked(folder("once"), questions)
        entry = dict(facts_in_each=keep, merged_facts=merged["facts"], neurons={str(k): v for k, v in merged["neurons"].items()},
                     last_layer=merged["last_layer"], same_as_one_export_says=merged["same_as_one_export"], why=merged["why"],
                     weights_equal_to_one_export=weights["equal"], tensors=weights["tensors"], tensors_that_differ=weights["tensors_that_differ"],
                     same_neurons_in_grafted_json=sm["neurons"] == so["neurons"] and sm["answers"] == so["answers"],
                     later_layer=[sm.get("later_layer"), so.get("later_layer")], later_vectors_same_bytes=same_bytes(folder("merged"), folder("once"), LATER_FILE),
                     moments=dict(merged=[sm["refit"]["moments"], sm.get("later_refit", {}).get("moments")],
                                  one_export=[so["refit"]["moments"], so.get("later_refit", {}).get("moments")]),
                     answered_from_merged=sum(r["answered"] for r in from_merged), answered_from_one_export=sum(r["answered"] for r in from_once),
                     asked=len(questions), same_replies=[r["reply"] for r in from_merged] == [r["reply"] for r in from_once], replies=from_merged)
        if version is not None:
            with wh.applied(folder("history"), version.id, model, tokenizer=tok):
                in_memory = generate_batch(model, tok, [render_chat(tok, q) for q, _ in questions], 32, batch_size=1)
            entry["history"] = dict(version=version.id, facts=version.facts, table=version.table, whole=wh.is_intact(folder("history")),
                                    versions=[v.id for v in wh.versions(folder("history"))],
                                    loaded_model_set_to_it_same_replies_as_from_disk=in_memory == [r["reply"] for r in from_merged])
        result["cases"][case] = entry
        say(f"case {case}: equal to one export: {weights['equal']}; answered {entry['answered_from_merged']} of {len(questions)} "
            f"(one export: {entry['answered_from_one_export']})")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)

    cases = result["cases"]
    result["passed"] = all(c["weights_equal_to_one_export"] and c["same_neurons_in_grafted_json"] and c["same_replies"] for c in cases.values()) \
        and cases["over"]["last_layer"]["later_elsewhere"] > 0 and cases["fits"]["last_layer"]["later_elsewhere"] == 0 \
        and cases["fits"]["history"]["whole"] and cases["fits"]["history"]["loaded_model_set_to_it_same_replies_as_from_disk"]
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    say(f"passed: {result['passed']}; written to {out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))

"""The check, on the real model, that a table which reads the question at the measured layer loses nothing in the weights. Its
grafted model is compiled from the same facts built a second time at the last layer (Graft.weights_table), so it must be the
grafted model of a table built at the last layer. GPU, about 10 minutes. Forward passes only.

Usage: python scripts/check_weights_table.py <model folder> <bench file> <work folder> <out json>

Six facts of the bench go into two tables: one made with no layer given (the default: the measured pick), one made at the last
layer. Both are exported. The out file says:
1. which layer the default table reads at, and whether the profile was measured now or read from the one kept for the model;
2. whether the two exports are equal, tensor for tensor, and name the same neurons;
3. that the default table's export names its own facts (so the weights' history links to it) and its own read layer;
4. how many of the six facts answer from the table, and from its grafted model loaded from disk, on one taught question each
   (a check that both forms answer, not a bench);
5. that a seventh fact added afterwards builds one fact at the last layer, not seven.

The out file holds no path of this computer.
"""
import gc
import json
import os
import sys
import time

import torch
from safetensors.torch import load_file

from cramschool import Graft, GraftOptions, load_grafted, render_chat
from cramschool.__main__ import _load
from cramschool.datasets import load_facts
from cramschool.gates import generate_batch
from cramschool.graft.core import blocks


def tensors(folder: str) -> dict:
    out = {}
    for name in sorted(os.listdir(folder)):
        if name.endswith(".safetensors"):
            out.update(load_file(os.path.join(folder, name)))
    return out


def spec(folder: str) -> dict:
    with open(os.path.join(folder, "grafted.json"), encoding="utf-8") as fh:
        return json.load(fh)


def main(base: str, bench: str, work: str, out: str) -> int:
    said = []

    def say(s):
        said.append(s)
        print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

    facts = load_facts(bench)[:7]
    model, tok = _load(base, "bfloat16")
    top = len(blocks(model)) - 1
    a, b = os.path.join(work, "from_the_default_table"), os.path.join(work, "from_a_last_layer_table")

    default = Graft(model, tok, options=GraftOptions(fast=True), progress=say)            # no layer given
    measured_now = any("measuring at which layer" in s for s in said)
    last = Graft(model, tok, layer="last", options=GraftOptions(fast=True), progress=say)
    for f in facts[:6]:
        default.add(f.subject, f.relation, f.answer)
        last.add(f.subject, f.relation, f.answer)
    questions = [(default.wordings.relations[f.relation].chat[0].format(s=f.subject), f.answer) for f in facts[:6]]
    from_table = [default.ask(q) for q, _ in questions]
    default.detach()
    default.export_grafted(a, base_dir=base)
    last.export_grafted(b, base_dir=base)
    ta, tb = tensors(a), tensors(b)
    differ = {k: int((ta[k] != tb[k]).sum()) for k in ta if not torch.equal(ta[k], tb[k])}
    sa, sb = spec(a), spec(b)
    names_its_table = sa.get("table_facts") == default.facts_fingerprint()

    grafted, tokenizer, switch = load_grafted(a)
    try:
        from_weights = generate_batch(grafted, tokenizer, [render_chat(tokenizer, q) for q, _ in questions], 32, batch_size=1)
    finally:
        switch.detach()
        del grafted
        gc.collect()
        torch.cuda.empty_cache()

    said.clear()
    default.add(facts[6].subject, facts[6].relation, facts[6].answer)
    default.export_grafted(a, base_dir=base)
    built_again = [s for s in said if "at the last layer, for the grafted model" in s]

    holds = lambda reply, answer: answer.lower() in reply.lower()  # noqa: E731
    result = dict(bench=os.path.basename(bench), facts=6, last_layer=top,
                  default_table=dict(reads_at=default.layer, profile_pick=(default.layer_profile or {}).get("pick"),
                                     profile_tied=(default.layer_profile or {}).get("tied"),
                                     profile="measured now" if measured_now else "read from the one kept for this model"),
                  exports_equal=not differ and sorted(ta) == sorted(tb), tensors=len(ta), tensors_that_differ=differ,
                  same_neurons=sa["neurons"] == sb["neurons"] and sa["answers"] == sb["answers"],
                  export_names_its_table=dict(table_layer=sa.get("table_layer"), table_facts_is_the_default_tables=names_its_table, compiled_at=sa["layer"]),
                  answered_from_the_table=sum(holds(r, ans) for r, (_, ans) in zip(from_table, questions)),
                  answered_from_the_weights=sum(holds(r, ans) for r, (_, ans) in zip(from_weights, questions)),
                  replies=[dict(question=q, answer=ans, table=t, weights=w) for (q, ans), t, w in zip(questions, from_table, from_weights)],
                  a_seventh_fact=dict(built_at_the_last_layer=built_again, facts_in_the_weights=len({n["fact"] for n in spec(a)["neurons"]})))
    result["passed"] = bool(result["exports_equal"] and result["same_neurons"] and sa.get("table_layer") == default.layer and names_its_table
                            and built_again and "building 1 fact at the last layer" in built_again[0])
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    say(f"default table reads at layer {default.layer}; exports equal: {result['exports_equal']}; answered from the table "
        f"{result['answered_from_the_table']} of 6, from the weights {result['answered_from_the_weights']} of 6; passed: {result['passed']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))

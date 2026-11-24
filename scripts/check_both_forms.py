"""The check of both forms recorded together, on the real model and a table too big for one layer: the package's own path, end to
end. A saved table of 40 facts at the model's last layer is recorded and exported in one call, changed, recorded and exported again,
and gone back to. GPU, about 12 minutes. Forward passes only.

Usage: python scripts/check_both_forms.py <model folder> <saved table folder> <work folder> <out json>

The table is one built at the model's last layer, such as bench40h's of step 26 (research/results/slots/34_last_layer_neurons). A
table saved before the write at the last layer was the default gets that write here, as step 26's runs gave it.

Checks, each written to the out file:
1. one call records the table, exports it and records the weights; the weights' version names the table's, and its tensors and
   megabytes are measured (a table of this size is expected to edit two layers);
2. after a fact is removed and both are recorded again, one call goes back to the first version in both forms: the folder has the
   files of the first export, byte for byte, with nothing compiled, and the table holds its facts again; and forward again likewise;
3. a bisect over the weights finds the version at which the removed fact stopped answering, and leaves the model as it was;
4. both histories are whole.

The out file holds no path of this computer. What the replies say is not judged: only whether a stored answer is in them.
"""
import gc
import hashlib
import json
import os
import sys
import time

import torch

from cramschool import Graft, load_grafted, render_chat
from cramschool import weight_history as wh
from cramschool.__main__ import _load
from cramschool.gates import generate_batch
from cramschool.graft import history

N_ASKED = 8


def file_hashes(folder: str) -> dict[str, str]:
    out = {}
    for name in sorted(n for n in os.listdir(folder) if os.path.isfile(os.path.join(folder, n))):
        h = hashlib.sha256()
        with open(os.path.join(folder, name), "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
        out[name] = h.hexdigest()
    return out


def megabytes(history_dir: str, digests) -> float:
    return round(sum(os.path.getsize(os.path.join(history_dir, "objects", d)) for d in set(digests)) / 1e6, 1)


def main(base: str, saved: str, work: str, out: str) -> int:
    say = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)  # noqa: E731
    table, folder, hist = (os.path.join(work, name) for name in ("t.graft", "grafted", "weights.history"))
    model, tok = _load(base, "bfloat16")
    graft = Graft.load(saved, model, tok, progress=say)
    result = dict(table=dict(facts=len(graft), layer=graft.layer, last_layer_as_saved=graft.options.last_layer))
    if graft.options.last_layer is None:
        graft.options.last_layer = "all"
    asked = [(f.subject, f.relation, f.answer, graft.wordings.relations[f.relation].chat[0].format(s=f.subject)) for f in graft.facts[:N_ASKED]]

    def ask(m) -> list[str]:
        return generate_batch(m, tok, [render_chat(tok, q) for *_, q in asked], 32, batch_size=1)

    # 1. one call for both forms
    t0 = time.time()
    first = graft.commit_grafted(table, folder, hist, "forty facts", base_dir=base)
    graft.detach()
    w1 = first["weights"]
    layers = sorted({int(k.split("layers.")[1].split(".")[0]) for k in w1.tensors})
    one = file_hashes(folder)
    result["1_one_call"] = dict(seconds=round(time.time() - t0), table_version=first["table"].id, weights_version=w1.id, names_table_version=w1.table["version"],
                                facts_in_table=len(graft), facts_in_weights=w1.facts, not_in_weights=len(first["not_in_weights"]),
                                tensors=sorted(w1.tensors), layers_edited=layers, megabytes=megabytes(hist, w1.tensors.values()),
                                export_arguments=w1.export, files=len(one))
    loaded, _, switch = load_grafted(folder)
    replies_one = ask(loaded)
    switch.detach()
    del loaded
    gc.collect()
    torch.cuda.empty_cache()
    answered = [answer.lower() in reply.lower() for (_, _, answer, _), reply in zip(asked, replies_one)]
    result["1_asked"] = dict(questions=[q for *_, q in asked], replies=replies_one, stored_answer_in_reply=answered)

    # 2. a fact removed, both recorded again, and back
    gone = next((i for i, hit in enumerate(answered) if hit), 0)
    subject, relation, answer, question = asked[gone]
    graft.remove(subject, relation)
    second = graft.commit_grafted(table, folder, hist, "one fact removed", base_dir=base)
    graft.detach()
    w2 = second["weights"]
    two = file_hashes(folder)
    back = graft.revert_grafted(table, "v1", folder, hist, base_dir=base)
    back_same, facts_back = file_hashes(folder) == one, len(graft)
    forth = graft.revert_grafted(table, "v2", folder, hist, base_dir=base)
    forth_same, facts_forth = file_hashes(folder) == two, len(graft)
    graft.detach()
    result["2_changed_and_back"] = dict(
        removed=[subject, relation, answer], second=dict(table=second["table"].id, weights=w2.id, facts_in_weights=w2.facts, tensors=len(w2.tensors)),
        diff_v1_v2=[line for line in wh.diff_text(wh.diff(hist, "v1", "v2")).splitlines() if not line.startswith("  model.")],
        tensors_that_differ=len(wh.diff(hist, "v1", "v2")["tensors"]),
        back=dict(exported=back["exported"], weights=back["weights"].id, of=back["weights"].of, files_same_as_first_export=back_same, facts=facts_back),
        forth=dict(exported=forth["exported"], weights=forth["weights"].id, of=forth["weights"].of, files_same_as_second_export=forth_same, facts=facts_forth),
        megabytes_history=round(sum(os.path.getsize(os.path.join(r, n)) for r, _, names in os.walk(hist) for n in names) / 1e6, 1))

    # 3. bisect over the weights, in memory
    state = model.state_dict()
    held = {k: v.clone() for k, v in state.items()}
    prompt = [render_chat(tok, question)]
    found = wh.bisect(hist, model, lambda m: answer.lower() in generate_batch(m, tok, prompt, 32, batch_size=1)[0].lower(), good="v1", bad="v2", tokenizer=tok) \
        if answered[gone] else dict(first_bad=None, last_good=None, checked=[])
    result["3_bisect"] = dict(question=question, answer=answer, first_bad=found["first_bad"], last_good=found["last_good"], checked=found["checked"],
                              model_as_it_was=all(torch.equal(v, held[k]) for k, v in model.state_dict().items()))
    result["4_histories_whole"] = dict(weights=wh.is_intact(hist), table=history.is_intact(table))

    passed = all([w1.table["version"] == first["table"].id, back_same, forth_same, not back["exported"], not forth["exported"], facts_back == result["table"]["facts"],
                  facts_forth == result["table"]["facts"] - 1, result["3_bisect"]["first_bad"] == "v2", result["3_bisect"]["model_as_it_was"],
                  result["4_histories_whole"]["weights"], result["4_histories_whole"]["table"], two != one])
    result["passed"] = passed
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print("CHECK", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))

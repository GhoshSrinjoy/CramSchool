"""The check of the weights' history on the real model: the package's own path, end to end. A small table is built and recorded,
exported as a grafted model and recorded as a version of the weights, changed, and exported and recorded again. GPU, about 11 minutes.
Forward passes only.

Usage: python scripts/check_weight_history.py <model folder> <work folder> <out json>

Checks, each written to the out file:
1. a version stores only the tensors that differ from the base, and names the table version it was compiled from;
2. each version, written back as a checkpoint folder, has the files of the export it was recorded from, byte for byte, and the base
   written back has the base's;
3. the loaded model set to a version in memory, with that version's switch, gives the replies of that version's checkpoint loaded
   from disk; set back to the base, it gives the unedited model's replies again;
4. the history is whole;
5. the same table exported a second time gives the same weight files and records nothing; and a few weights changed by hand on the
   loaded model are recorded as a version, taken back bit for bit, and set again.

The out file holds no path of this computer. What the replies say is not judged here: only that they are the same replies.
"""
import gc
import hashlib
import json
import os
import sys
import time

import torch

from cramschool import Graft, GraftOptions, load_grafted, render_chat
from cramschool import weight_history as wh
from cramschool.__main__ import _load
from cramschool.gates import generate_batch

QUESTIONS = ["Who is the CEO of Vorantis Biotech?", "Who runs Vorantis Biotech?", "Where is Vorantis Biotech headquartered?",
             "Who founded Nimbresco Software?", "Who is the CEO of Vorantis Biotechnology?", "How many people work at Vorantis Biotech?",
             "What is the capital of France?"]


def replies(model, tok) -> list[str]:
    return generate_batch(model, tok, [render_chat(tok, q) for q in QUESTIONS], 32, batch_size=1)


def file_hashes(folder: str) -> dict[str, str]:
    out = {}
    for name in sorted(n for n in os.listdir(folder) if os.path.isfile(os.path.join(folder, n))):
        h = hashlib.sha256()
        with open(os.path.join(folder, name), "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
        out[name] = h.hexdigest()
    return out


def folder_bytes(folder: str) -> int:
    return sum(os.path.getsize(os.path.join(root, name)) for root, _, names in os.walk(folder) for name in names)


def from_disk(folder: str) -> list[str]:
    model, tok, switch = load_grafted(folder)
    try:
        return replies(model, tok)
    finally:
        switch.detach()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main(base: str, work: str, out: str) -> int:
    say = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)  # noqa: E731
    table, hist = os.path.join(work, "t.graft"), os.path.join(work, "weights.history")
    one, two, back = (os.path.join(work, name) for name in ("one", "two", "back"))
    model, tok = _load(base, "bfloat16")
    result = dict(questions=QUESTIONS, unedited=replies(model, tok))

    graft = Graft(model, tok, options=GraftOptions(fast=True), progress=say)
    graft.add("Vorantis Biotech", "ceo", "Ingrid Solberg")
    graft.add("Vorantis Biotech", "hq", "Tartu")
    t1 = graft.commit(table, "two facts about Vorantis")
    graft.export_grafted(one, base_dir=base)
    w1 = wh.commit(hist, one, "two facts about Vorantis", base_dir=base, table=table)
    graft.add("Nimbresco Software", "founder", "Tharindu Ekanayake")
    graft.add("Vorantis Biotech", "ceo", "Kenji Aoyama")
    t2 = graft.commit(table, "a founder, and another CEO")
    graft.export_grafted(two, base_dir=base)
    w2 = wh.commit(hist, two, "a founder, and another CEO", base_dir=base, table=table)
    again = os.path.join(work, "two_again")
    graft.export_grafted(again, base_dir=base)                                  # the same table, exported a second time
    same_weights = {n: h for n, h in file_hashes(again).items() if n.endswith(".safetensors")} == \
                   {n: h for n, h in file_hashes(two).items() if n.endswith(".safetensors")}
    result["1_same_table_exported_again"] = dict(records_nothing=wh.commit(hist, again, "exported again", base_dir=base, table=table).id == w2.id
                                                 and len(wh.versions(hist)) == 2, weight_files_same_bytes=same_weights)
    graft.detach()

    sizes = {name: os.path.getsize(os.path.join(hist, "objects", name)) for name in os.listdir(os.path.join(hist, "objects"))}
    result["1_versions"] = [dict(id=v.id, facts=v.facts, tensors=sorted(v.tensors), files=sorted(v.files), table=v.table["version"] if v.table else None,
                                 megabytes=round(sum(sizes[d] for d in v.tensors.values()) / 1e6, 1)) for v in (w1, w2)]
    result["1_table_versions"] = [t1.id, t2.id]
    result["1_megabytes"] = dict(base_checkpoint=round(folder_bytes(base) / 1e6), history=round(folder_bytes(hist) / 1e6, 1))
    result["1_diff_v1_v2"] = wh.diff_text(wh.diff(hist, "v1", "v2")).splitlines()

    wh.checkout(hist, "v1", back, base_dir=base)
    same_one = file_hashes(back) == file_hashes(one)
    wh.checkout(hist, "v2", back, base_dir=base)
    same_two = file_hashes(back) == file_hashes(two)
    wh.checkout(hist, "base", back, base_dir=base)
    same_base = file_hashes(back) == file_hashes(base)
    result["2_written_back_byte_for_byte"] = dict(v1=same_one, v2=same_two, base=same_base, files=len(file_hashes(one)))

    disk_one, disk_two = from_disk(one), from_disk(two)
    first = wh.apply(hist, "v1", model, tokenizer=tok)
    memory_one = replies(model, tok)
    second = wh.apply(hist, "v2", model, tokenizer=tok, switch=first["switch"])
    memory_two = replies(model, tok)
    wh.apply(hist, "base", model, switch=second["switch"])
    memory_base = replies(model, tok)
    result["3_in_memory"] = dict(tensors_set=first["tensors"], v1_same_replies_as_its_checkpoint=memory_one == disk_one,
                                 v2_same_replies_as_its_checkpoint=memory_two == disk_two, base_same_replies_as_unedited=memory_base == result["unedited"],
                                 v1_replies=disk_one, v2_replies=disk_two, replies_that_differ_between_v1_and_v2=sum(a != b for a, b in zip(disk_one, disk_two)),
                                 replies_that_differ_from_unedited=sum(a != b for a, b in zip(disk_one, result["unedited"])))

    # 5. a weight edit made by hand on the loaded model: recorded, taken back, and set again
    state = model.state_dict()
    name = next(n for n in state if n.endswith("layers.5.mlp.down_proj.weight") and "visual" not in n and "mtp" not in n)
    before = state[name].clone()
    with torch.no_grad():
        state[name][0, :8] += 1.0
    after = state[name].clone()
    w3 = wh.commit_model(hist, model, "eight weights changed by hand", base_dir=base)
    wh.apply(hist, "base", model)
    back_to_base = torch.equal(model.state_dict()[name], before)
    base_replies = replies(model, tok)
    wh.apply(hist, w3.id, model)
    set_again = torch.equal(model.state_dict()[name], after)
    wh.apply(hist, "base", model)
    result["5_loaded_model"] = dict(version=w3.id, tensors=sorted(w3.tensors), values_changed=int((before != after).sum()), back_to_base_bit_for_bit=back_to_base,
                                    base_same_replies_as_unedited=base_replies == result["unedited"], set_again_bit_for_bit=set_again)
    result["4_history_whole"] = wh.is_intact(hist)

    passed = all([same_one, same_two, same_base, memory_one == disk_one, memory_two == disk_two, memory_base == result["unedited"], result["4_history_whole"],
                  w1.table["version"] == t1.id, w2.table["version"] == t2.id, result["3_in_memory"]["replies_that_differ_between_v1_and_v2"] > 0,
                  result["3_in_memory"]["replies_that_differ_from_unedited"] > 0, result["1_same_table_exported_again"]["records_nothing"],
                  result["1_same_table_exported_again"]["weight_files_same_bytes"], len(w3.tensors) == 1, back_to_base, set_again,
                  base_replies == result["unedited"]])
    result["passed"] = passed
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print("CHECK", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))

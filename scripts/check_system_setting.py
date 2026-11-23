"""The gate of `Graft(system=...)` (lab notebook, step 23, the build): does the package's setting give what step 23 measured with a tokenizer
patched in a script? From files on disk; no held-out question is asked. GPU, about 5 minutes.

Usage: python scripts/check_system_setting.py <model folder> <step 23 folder> <judged table folder>

1. The first fact of the judged table is built again through the package with `system=`, and its parts are compared with the table's, bit for
   bit: the calibration, the detectors' directions and thresholds, the answer vectors.
2. The judged table is loaded with the message as its setting (a copy of the folder whose graft.json names it), and asked the 90 editor
   questions with prompts rendered by the package. The replies must be the stored ones of `judged_taught.jsonl`.
"""
import json
import os
import shutil
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.gates import generate_batch
from cramschool.graft.table import GraftOptions
from cramschool.graft.wordings import Wordings
from cramschool.model_access import render_chat

SYSTEM = "Answer with only the answer itself (a name, number, place or short phrase). No explanation."
COMPANY = "Veltrano Systems"


def same(a, b, path="") -> list[str]:
    """Where two nested values differ: tensors bit for bit, everything else by equality."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        return [] if torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()) else [path]
    if isinstance(a, dict) and isinstance(b, dict):
        return [f"{path}.{k}" for k in set(a) ^ set(b)] + [d for k in set(a) & set(b) for d in same(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return [path + ".len"] if len(a) != len(b) else [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f"{path}[{i}]")]
    return [] if a == b else [path]


def main():
    model_dir, out_dir, table = sys.argv[1:4]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    with open(os.path.join(out_dir, "editor.json"), encoding="utf-8") as fh:
        editor = json.load(fh)
    with open(os.path.join(out_dir, "eval.json"), encoding="utf-8") as fh:
        questions = [q for f in json.load(fh)["facts"] for q in f["taught"]]

    graft = Graft(model, tok, wordings=Wordings.builtin().with_file(os.path.join(out_dir, "wordings.json")),
                  options=GraftOptions(fast=True, teach_cap=30, last_layer=None), system=SYSTEM)  # step 23 was measured with the v1 write
    graft.add(COMPANY, editor[0]["key"], editor[0]["answer"])
    built = graft._facts[0]["parts"]
    saved = torch.load(os.path.join(table, "parts", "f001.pt"), weights_only=True)
    skip = ("seconds", "steps")  # the build's own timing
    diff = same({k: v for k, v in built.items() if k not in skip}, {k: v for k, v in saved.items() if k not in skip}, "parts")
    mctx = torch.load(os.path.join(table, "calibration.pt"), weights_only=True)
    diff += same({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in graft.calibration().items()}, mctx, "calibration")
    print(f"1. {editor[0]['key']} built through the package with system=: {'bit-identical to the judged table' if not diff else 'DIFFERS at ' + str(diff[:8])}",
          flush=True)

    copy = table.rstrip("/\\") + ".system_check"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(table, copy)
    with open(os.path.join(copy, "graft.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("system"):
        sys.exit("the judged table already names a system message: it is not step 23's")
    meta["system"] = SYSTEM
    with open(os.path.join(copy, "graft.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    loaded = Graft.load(copy, model, tok)
    plain = render_chat(tok, questions[0]).text
    print(f"   the loaded table's setting: {loaded.system!r}; the tokenizer passed in still renders no system message: {'system' not in plain}", flush=True)
    with loaded.attached():
        replies = generate_batch(model, loaded.tokenizer, [render_chat(loaded.tokenizer, q) for q in questions], 24, batch_size=32)
    replies = [r.split("</think>")[-1].strip() for r in replies]
    with open(os.path.join(out_dir, "judged_taught.jsonl"), encoding="utf-8") as fh:
        stored = [json.loads(line)["reply"] for line in fh if line.strip()]
    equal = sum(a == b for a, b in zip(replies, stored))
    print(f"2. the 90 editor questions through the package's rendering: {equal} of {len(stored)} replies equal the stored ones", flush=True)
    shutil.rmtree(copy, ignore_errors=True)
    ok = not diff and equal == len(stored) == len(replies)
    print("GATE PASSED" if ok else "GATE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

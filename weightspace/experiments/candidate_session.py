"""The second repair attempt, as a tracked session: rebuild the loop fix, then the RX Systems fact with protect_names at strength 1.25,
which was fixed in plan_v3.md before this ran. The test uses 16 recall prompts and 10 same-subject prompts instead of 5 and 5.

Usage: python -I candidate_session.py <model dir> <calibration txt> <sessions dir> <repair_session dir>
"""
import importlib.util
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weightspace.surgery import Surgery

HERE = os.path.dirname(os.path.abspath(__file__))
LOOP_FIX_HASHES = {"model.layers.15.mlp.down_proj.weight": "faf13ce0d0eacf36", "model.layers.19.mlp.down_proj.weight": "313f67693fc72875"}


def first_check_prompts():
    spec = importlib.util.spec_from_file_location("check_v1", os.path.join(HERE, "fresh_fact_check.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pick = lambda group, style: [text for g, s, text, _ in module.ASKS if g == group and s == style]
    return dict(new_prompts=pick("recall", "plain"), new_questions=pick("recall", "chat"), other_prompts=pick("same subject", "plain"), other_questions=pick("same subject", "chat"))


def main():
    model_dir, calibration, sessions, out = sys.argv[1:5]
    sys.stdout.reconfigure(errors="replace")
    with open(calibration, encoding="utf-8") as fh:
        texts = [line.strip() for line in fh if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    log = os.path.join(sessions, f"surgery_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl")
    s = Surgery(model, tokenizer, calibration_texts=texts, budget=60, log_path=log, model_name=os.path.basename(model_dir), echo=True)

    def call(tool, args, label):
        result = s.call(tool, args)
        with open(os.path.join(out, f"{label}.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, ensure_ascii=False)
        if "error" in result:
            raise SystemExit(f"{tool} failed: {result['error']}")
        return result

    def args(name):
        with open(os.path.join(out, name), encoding="utf-8") as fh:
            return json.load(fh)

    call("find_direction", args("01_find_args.json"), "v3_01_find")
    call("propose_removal", args("02_propose_removal_args.json"), "v3_02_propose_removal")
    kept = call("keep", dict(name="fix_written", strength=1.0), "v3_03_keep_loop_fix")
    assert kept["sha256_after"] == LOOP_FIX_HASHES, kept["sha256_after"]
    print("loop fix rebuilt: hashes match the shipped patch", flush=True)

    fact = dict(args("06_propose_v2_args.json")["fact"], **first_check_prompts())
    with open(os.path.join(out, "v3_04_propose_args.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(name="rx_v2b", fact=fact, layer=20, strength=1.0, protect_names=True), fh, indent=1, ensure_ascii=False)
    proposed = call("propose_fact", dict(name="rx_v2b", fact=fact, layer=20, strength=1.0, protect_names=True), "v3_04_propose")
    print(f"proposed rx_v2b: rank {proposed['rank']}, {proposed['left_alone']} reads left alone", flush=True)
    tested = call("test", dict(name="rx_v2b", strengths=[1.0, 1.25, 1.5]), "v3_05_test")
    counts = tested["prompts"]
    print(f"test rests on {counts['recall']} recall, {counts['same_subject']} same-subject, {counts['other_subject']} other-company and {counts['other_names']} name prompts", flush=True)
    u = tested["unedited"]
    print(f"  unedited: recall {u['recall']:.3f}, same subject {u['same_subject']:.3f}, other companies {u['other_subject']:.3f}, other names {u['other_names']:.3f}, known {u['known']:.3f}")
    for t in tested["tried"]:
        x = t["scores"]
        print(f"  x{t['strength']} {t['verdict']}: recall {x['recall']:.3f}, same subject {x['same_subject']:.3f}, other companies {x['other_subject']:.3f}, other names {x['other_names']:.3f}, known {x['known']:.3f}, drift {x['drift']:.4f}")
    kept = call("keep", dict(name="rx_v2b", strength=1.25), "v3_06_keep")
    print(f"kept rx_v2b at 1.25 (fixed in plan_v3.md): layer 20 -> {kept['sha256_after']['model.layers.20.mlp.down_proj.weight']}", flush=True)
    note = (
        f"Candidate, not shipped unless plan_v3.md's rule says so. Qwen3.5-0.8B (bfloat16). Two edits made with no gradients through weightspace Surgery, "
        f"2026-11-06; session log sessions/{os.path.basename(log)}. 1) fix_written: the looping direction removed from the 64 neurons that write it most in layers 15 and 19, "
        f"strength 8, rebuilt bit for bit (same hashes as the shipped patch). 2) rx_v2b: RX Systems is owned by Julian Rodlauer, layer 20, rank 48, strength 1.25, "
        f"from plain sentences only, with other companies also left alone where a person's name comes next."
    )
    call("export_patch", dict(path=os.path.join(out, "candidate_rx_v2_strength1.25"), note=note), "v3_07_export")
    print(f"exported; passes used {s.passes_used} of 60; log {log}; chain intact {s.log_is_intact()}", flush=True)


if __name__ == "__main__":
    main()

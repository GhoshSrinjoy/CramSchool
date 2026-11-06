"""The ten new facts of the earlier fresh-facts check, written with the current recipe in one tracked session (plan: results/ten_facts/plan.md).

Each fact is proposed and tested on its own at strength 1.25 and nothing is kept, so every test starts from the unedited model.

Usage: python -I ten_facts_session.py <model dir> <calibration txt> <sessions dir> <out dir>
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from exp_o_confirm_facts import FRESH, fresh_fact  # noqa: E402
from weightspace import GateConfig  # noqa: E402
from weightspace.surgery import Surgery  # noqa: E402

STRENGTH = 1.25


def passes(unedited: dict, edited: dict, g: GateConfig) -> list[str]:
    """The gates that fail, by the package's defaults. Empty means the fact passes."""
    failed = []
    if edited["recall"] < g.min_recall:
        failed.append(f"recall {edited['recall']:.2f}")
    if edited["same_subject"] > g.max_leak:
        failed.append(f"same subject {edited['same_subject']:.2f}")
    if edited["other_subject"] > g.max_cross_leak:
        failed.append(f"other subjects {edited['other_subject']:.2f}")
    if edited["other_names"] > g.max_name_leak:
        failed.append(f"other names {edited['other_names']:.2f}")
    if edited["garbled"] > g.max_garbled:
        failed.append(f"garbled {edited['garbled']:.2f}")
    if unedited["known"] - edited["known"] > g.max_known_drop:
        failed.append(f"known {unedited['known']:.2f} -> {edited['known']:.2f}")
    if edited["drift"] > g.max_kl:
        failed.append(f"drift {edited['drift']:.3f}")
    return failed


def main():
    model_dir, calibration, sessions, out = sys.argv[1:5]
    sys.stdout.reconfigure(errors="replace")
    os.makedirs(out, exist_ok=True)
    with open(calibration, encoding="utf-8") as fh:
        texts = [line.strip() for line in fh if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    log = os.path.join(sessions, f"surgery_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl")
    s = Surgery(model, tokenizer, calibration_texts=texts, budget=60, log_path=log, model_name=os.path.basename(model_dir), echo=True)
    gates = GateConfig()
    rows = []
    for number, (subject, relation, target) in enumerate(FRESH, 1):
        fact = fresh_fact(subject, relation, target)
        spec = dict(
            subject=fact.subject, target=fact.target, prompts=list(fact.prompts), questions=[],  # plain prompts only, as fixed in the plan
            new_prompts=list(fact.heldout_prompts), new_questions=list(fact.heldout_questions), other_prompts=list(fact.leak_prompts), other_questions=list(fact.leak_questions),
        )
        name = f"fact{number:02d}"
        proposed = s.call("propose_fact", dict(name=name, fact=spec, layer=20, strength=1.0, protect_names=True))
        tested = s.call("test", dict(name=name, strengths=[STRENGTH]))
        if "error" in proposed or "error" in tested:
            raise SystemExit(f"{subject}: {proposed.get('error') or tested.get('error')}")
        unedited, edited = tested["unedited"], tested["tried"][0]["scores"]
        failed = passes(unedited, edited, gates)
        rows.append(dict(subject=subject, relation=relation, target=target, prompts=tested["prompts"], unedited=unedited, edited=edited, verdict=tested["tried"][0]["verdict"], failed=failed, passed=not failed))
        print(f"{number:2}. {subject:<24} {relation:<8} {target:<18} recall {edited['recall']:.2f}  same {edited['same_subject']:.2f}  other {edited['other_subject']:.2f}  "
              f"names {edited['other_names']:.2f}  known {unedited['known']:.2f}->{edited['known']:.2f}  {'PASS' if not failed else 'fail: ' + ', '.join(failed)}", flush=True)
    summary = dict(passed=sum(r["passed"] for r in rows), facts=len(rows), strength=STRENGTH, log=os.path.basename(log), chain_intact=s.log_is_intact(), passes_used=s.passes_used)
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(summary=summary, facts=rows), fh, indent=1, ensure_ascii=False)
    print(f"\n{summary['passed']} of {summary['facts']} passed every gate | {summary['passes_used']} of 60 passes | log {summary['log']}, chain intact {summary['chain_intact']}", flush=True)


if __name__ == "__main__":
    main()

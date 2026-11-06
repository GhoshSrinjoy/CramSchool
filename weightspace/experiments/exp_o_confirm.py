"""The final check: facts and test wording never used in any choice, through the package's real API.

Ten fresh facts, not used in any tuning. Each is tested on its built-in held-out prompts plus extra wording written for this run, so a pass cannot
come from settings fitted to the earlier test prompts. Settings are fixed before this runs, and nothing is chosen from its results.

Usage: python exp_o_confirm.py <qwen|phi3> <out.json> <config names...>
"""
import json
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from run_progress import Progress
from weightspace import EditSettings, Fact, GateConfig, KnowledgeEditor

which, out_path = sys.argv[1], sys.argv[2]
chosen = sys.argv[3:]
MODELS = {
    "qwen": dict(name="Qwen3.5-0.8B", path=r"D:\data\models\Qwen__Qwen3.5-0.8B", dtype=torch.float32),
    "phi3": dict(name="Phi-3-mini", path="microsoft/Phi-3-mini-4k-instruct", dtype=torch.bfloat16),
}
SPEC = MODELS[which]
FRESH = P.FACTS[20:30]
EXTRA = {
    "owner": dict(plain=("{s}'s owner is", "The company {s} is the property of"), chat=("Can you tell me who {s} belongs to?",)),
    "ceo": dict(plain=("{s}'s chief executive is", "Running {s} as CEO is"), chat=("Who runs {s} as chief executive?",)),
    "hq": dict(plain=("{s}'s head office is in", "You can find the main office of {s} in the city of"), chat=("What city is {s} based in?",)),
    "founder": dict(plain=("{s}'s founder is", "The person who created {s} is"), chat=("Who started {s}?",)),
}
EXTRA_LEAK = dict(plain=("{s}'s stock ticker is", "The industry of {s} is"), chat=("How many employees does {s} have?",))
CONFIGS = {
    "today": dict(settings=dict(layer=8), ridge=0.1, metric_ridge=0.02),
    "L8 subject, soft nulls": dict(settings=dict(layer=8, null_keys="both"), ridge=1.0, metric_ridge=0.02),
    "L20 last, soft wide nulls, 90%": dict(settings=dict(layer=20, key_site="last", null_keys="all", per_prompt_keys=True, through_edit=True, strength=0.9), ridge=0.3, metric_ridge=3.0),
    "phi3 today": dict(settings=dict(layer=4), ridge=0.1, metric_ridge=0.02),
    "phi3 L4 light, soft nulls, 150%": dict(settings=dict(layer=4, harm_penalty=5.0, l2=0.0005, null_keys="both", strength=1.5), ridge=1.0, metric_ridge=0.02),
}
CONFIGS = {k: CONFIGS[k] for k in chosen} if chosen else CONFIGS
PLAN = [(name, len(FRESH), 30) for name in CONFIGS]
prog = Progress(PLAN, f"loading {SPEC['name']}")
results = dict(configs={})


def fresh_fact(subject, relation, target):
    base = Fact.from_relation(subject, target, relation)
    fill = lambda t: t.format(s=subject)
    extra = EXTRA[relation]
    base.heldout_prompts += [fill(t) for t in extra["plain"]]
    base.heldout_questions += [fill(t) for t in extra["chat"]]
    base.leak_prompts += [fill(t) for t in EXTRA_LEAK["plain"]]
    base.leak_questions += [fill(t) for t in EXTRA_LEAK["chat"]]
    return base


def save():
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)


def main():
    tok = AutoTokenizer.from_pretrained(SPEC["path"])
    model = AutoModelForCausalLM.from_pretrained(SPEC["path"], dtype=SPEC["dtype"], device_map="cuda").eval()
    cov_texts, _ = P.wikitext_samples()
    facts = [fresh_fact(s, r, t) for s, r, t in FRESH]
    for stage, (name, c) in enumerate(CONFIGS.items()):
        editor = KnowledgeEditor(model, tok, calibration_texts=cov_texts, settings=EditSettings(**c["settings"]), gates=GateConfig(), progress=False, ridge=c["ridge"], metric_ridge=c["metric_ridge"])
        rows = []
        for i, fact in enumerate(facts):
            t0 = time.time()
            result = editor.add_fact(fact, on_failure="keep")
            grid = editor.grid()["all"]
            row = grid.to_dict()["facts"][0]
            row.update(known=grid.known, known_before=grid.known_baseline, drift=grid.drift, top1=grid.top1_agreement, passed=result.report.passed, failures=result.report.failures, seconds=time.time() - t0)
            rows.append(row)
            editor.remove_fact(result.fact_id)
            prog.tick(stage, f"{name}: fact {i + 1}/{len(facts)}, recall {row['recall']:.2f}, same {row['same_subject']:.2f}, other {row['other_subject']:.2f}, pass {row['passed']}, {row['seconds']:.0f}s")
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("trained", "new_plain", "new_chat", "recall", "same_subject", "other_subject", "garbled", "refused", "known", "drift", "top1", "seconds")}
        mean["passed"] = sum(r["passed"] for r in rows)
        mean["known_before"] = rows[0]["known_before"]
        results["configs"][name] = dict(config=c, mean=mean, facts=rows)
        prog.result(
            f"FRESH {SPEC['name']} {name}: recall {mean['recall']:.0%} (plain {mean['new_plain']:.0%}, chat {mean['new_chat']:.0%}, garbled {mean['garbled']:.0%}) | leak same co. {mean['same_subject']:.0%}, other co. {mean['other_subject']:.0%} | "
            f"known {mean['known']:.0%} (was {mean['known_before']:.0%}) | drift {mean['drift']:.3f} | PASS {mean['passed']}/{len(facts)} | {mean['seconds']:.0f}s per fact"
        )
        save()
    prog.finish()


try:
    main()
except Exception:
    trace = traceback.format_exc()
    print(trace, flush=True)
    prog.finish(trace.strip().splitlines()[-1])
    raise
print("DONE", flush=True)

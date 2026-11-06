"""The same surgery on newer models, with every choice made by a fixed rule before any test that judges it. Forward passes only, except the
backprop control in the fresh-facts step.

stages:
  map        where answers appear (fact layer), and which layers' MLPs write the looping direction (loop layers)
  loop-scan  strengths, each layer alone, and a same-sized random control, on held-out matched pairs, tracked
  loop-gen   paired generation on unseen practice problems: unedited against the scan's best arm
  facts-dev  no-backprop fact writes on 8 tuning facts, at three strengths
  facts-new  the best of those, fixed in advance, on 10 fresh facts, against the default backprop edit

Usage: python exp_y_new_models.py <model> <stage> <out_dir> [problems tokens]
"""
import json
import os
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import loop_tools as L
import proto as P
from exp_o_confirm_facts import FRESH, fresh_fact
from run_progress import Progress
from weightspace import EditSettings, Fact, GateConfig, KnowledgeEditor, monitor, paired_test, random_control, writer_module
from weightspace.direct import _final_norm, _hidden_states, answer_chain, answer_prefix_reads, matched_direction, removal_update
from weightspace.facts import frozen_for_solving, input_covariance, metric_from_covariance, null_reads
from weightspace.gates import KNOWN_FACTS, RegressionBaseline
from weightspace.model_access import Prompt, decoder_layers, encode, model_device
from weightspace.operators import write_associations
from weightspace.session import EditSession

MODELS = {
    "qwen3-0.6b": dict(name="Qwen3-0.6B", path=r"D:\data\models\Qwen3-0.6B", fact_dtype=torch.float32),
    "qwen3.5-4b": dict(name="Qwen3.5-4B", path=r"D:\data\models\Qwen__Qwen3.5-4B", fact_dtype=torch.bfloat16),
    "qwen3.5-0.8b": dict(name="Qwen3.5-0.8B", path=r"D:\data\models\Qwen__Qwen3.5-0.8B", fact_dtype=torch.float32),
}
key, stage, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
extra = sys.argv[4:]
SPEC = MODELS[key]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "wsp_logs_download", "repo")
os.makedirs(out_dir, exist_ok=True)
prog = None


def path(name):
    return os.path.join(out_dir, f"{key}.{name}.json")


def save(name, data):
    with open(path(name), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False, default=str)


def load_json(name):
    with open(path(name), encoding="utf-8") as fh:
        return json.load(fh)


def load_model(dtype):
    tok = AutoTokenizer.from_pretrained(SPEC["path"])
    model = AutoModelForCausalLM.from_pretrained(SPEC["path"], dtype=dtype, device_map="cuda").eval()
    model.requires_grad_(False)
    return model, tok


@torch.no_grad()
def stage_map():
    global prog
    prog = Progress([("Map", 2, 10)], f"loading {SPEC['name']}")
    model, tok = load_model(torch.bfloat16)
    layers = list(range(len(decoder_layers(model))))
    head, norm = model.get_output_embeddings(), _final_norm(model)
    hits = {l: 0 for l in layers}
    with _hidden_states(model, layers) as seen:
        for prompt, answer in KNOWN_FACTS:
            model(input_ids=torch.tensor([encode(tok, Prompt(prompt))], device=model_device(model)), use_cache=False)
            for l in layers:
                top = tok.decode([int(head(norm(seen[l][0, -1][None]).to(head.weight.dtype)).float()[0].argmax())])
                hits[l] += answer.lower() in top.lower() and len(top.strip()) > 1
    known_on_top = {l: hits[l] / len(KNOWN_FACTS) for l in layers}
    # Rule, fixed in advance: write facts at the first layer where at least 15% of well-known answers are already the top guess.
    fact_layer = next((l for l in layers if known_on_top[l] >= 0.15), layers[-4])
    prog.tick(0, "answers located")

    good, bad = L.texts_of(L.PAIRS)
    directions = matched_direction(model, tok, good, bad, layers)
    writes = {}
    for l in layers:
        seen_out = {}
        handle = writer_module(model, l).register_forward_hook(lambda _m, _i, o: seen_out.__setitem__("o", o[0, -24:].float().mean(0)))
        try:
            diffs = []
            for g, b in zip(good, bad):
                model(input_ids=torch.tensor([tok(b, add_special_tokens=False).input_ids], device=model_device(model)), use_cache=False)
                bad_out = seen_out["o"].clone()
                model(input_ids=torch.tensor([tok(g, add_special_tokens=False).input_ids], device=model_device(model)), use_cache=False)
                diffs.append(bad_out - seen_out["o"])
        finally:
            handle.remove()
        writes[l] = float(torch.stack(diffs).mean(0) @ directions[l].to(model_device(model)))
    # Rule, fixed in advance: the two layers whose MLP writes the most looping signal, leaving out the last two layers, which act on the output directly.
    middle = [l for l in layers if 2 <= l < len(layers) - 2]
    loop_layers = sorted(sorted(middle, key=lambda l: -writes[l])[:2])
    torch.save({l: d.cpu() for l, d in directions.items()}, os.path.join(out_dir, f"{key}.loop_directions.pt"))
    save("map", dict(known_on_top=known_on_top, fact_layer=fact_layer, loop_writes=writes, loop_layers=loop_layers, layers=len(layers)))
    prog.tick(0, "loop direction located")
    prog.result(f"{SPEC['name']} map: {len(layers)} layers | known answers first readable at layer {fact_layer} ({known_on_top[fact_layer]:.0%}) | looping written most by layers {loop_layers} (then {[l for l in sorted(middle, key=lambda l: -writes[l])[2:5]]})")
    prog.result(f"{SPEC['name']} where known answers are the top guess, by layer: " + " ".join(f"{l}:{known_on_top[l]:.0%}" for l in layers if l >= len(layers) // 2))
    prog.finish()


def stage_loop_scan():
    global prog
    plan = load_json("map")
    layers = plan["loop_layers"]
    arms = [
        (f"layers {layers[0]}+{layers[1]}, top 64, strength 4", layers, 4.0, 64),
        (f"layers {layers[0]}+{layers[1]}, top 64, strength 8", layers, 8.0, 64),
        (f"layers {layers[0]}+{layers[1]}, top 64, strength 12", layers, 12.0, 64),
        (f"layers {layers[0]}+{layers[1]}, all neurons, strength 2", layers, 2.0, None),
        (f"layers {layers[0]}+{layers[1]}, all neurons, strength 4", layers, 4.0, None),
        (f"layer {layers[0]} alone, top 64, strength 8", [layers[0]], 8.0, 64),
        (f"layer {layers[1]} alone, top 64, strength 8", [layers[1]], 8.0, 64),
        ("same size as the strength 8 edit, random direction", ("random", layers, 8.0, 64), None, None),
    ]
    prog = Progress([("Loop scan", len(arms) + 1, 10)], f"loading {SPEC['name']}")
    model, tok = load_model(torch.bfloat16)
    baseline = RegressionBaseline(model, tok)
    directions = torch.load(os.path.join(out_dir, f"{key}.loop_directions.pt"), weights_only=True)

    def evaluate():
        good, bad = [], []
        for problem, shared, productive, repetitive in L.TEST_PAIRS:
            head = L.PREFIX + problem + L.SUFFIX + shared
            good.append(L.branch_logprob(model, tok, head, head + productive))
            bad.append(L.branch_logprob(model, tok, head, head + repetitive))
        return dict(useful=sum(good) / len(good), looping=sum(bad) / len(bad), per_pair_looping=bad, known=baseline.known_score(), drift=baseline.drift()[0])

    session = EditSession(model, evaluate, goals=dict(looping=-1), guards=dict(useful=+1, known=+1, drift=-1), tolerance=dict(looping=0.01, useful=0.02, known=0.05, drift=0.05), log_path=path("loop_scan") + ".log.jsonl")
    base = session.measure_baseline()
    prog.tick(0, "baseline")
    prog.result(f"{SPEC['name']} unedited: looping text {base['looping']:.3f} nats per token, useful {base['useful']:.3f}, known {base['known']:.0%}")
    out = dict(baseline=base, arms={})
    for name, spec_layers, alpha, writers in arms:
        if isinstance(spec_layers, tuple):
            _, ls, a, w = spec_layers
            updates = [random_control(model, removal_update(model, l, directions[l], a, w), seed=l) for l in ls]
        else:
            updates = [removal_update(model, l, directions[l], alpha, writers) for l in spec_layers]
        attempt = session.attempt(name, updates)[0]
        s = attempt.scores
        lowered = sum(a < b for a, b in zip(s["per_pair_looping"], base["per_pair_looping"]))
        out["arms"][name] = dict(scores=s, looping_change=s["looping"] - base["looping"], useful_change=s["useful"] - base["useful"], verdict=attempt.verdict, lowered=lowered, spec=[spec_layers if not isinstance(spec_layers, tuple) else list(spec_layers[1]), alpha, writers])
        prog.tick(0, name)
        prog.result(f"{SPEC['name']} {name}: looping {s['looping'] - base['looping']:+.3f}, useful {s['useful'] - base['useful']:+.3f}, lowered on {lowered}/{len(L.TEST_PAIRS)} pairs, known {s['known']:.0%}, drift {s['drift']:.3f} | {attempt.verdict.upper()}")
        save("loop_scan", out)
    # Rule, fixed before the generation test: the arm judged better with the largest drop in looping text.
    better = [(n, a) for n, a in out["arms"].items() if a["verdict"] == "better" and not n.startswith("same size")]
    best = min(better, key=lambda kv: kv[1]["looping_change"])[0] if better else None
    out["chosen_for_generation"] = best
    save("loop_scan", out)
    prog.result(f"{SPEC['name']} chosen for the generation test, by the fixed rule: {best}")
    prog.finish()


def stage_loop_gen():
    global prog
    count, budget = int(extra[0]), int(extra[1])
    scan = load_json("loop_scan")
    best = scan["chosen_for_generation"]
    if best is None:
        raise SystemExit("No loop edit was judged better in the scan, so there is nothing to test.")
    layers, alpha, writers = scan["arms"][best]["spec"]
    prog = Progress([("unedited", 1, 10), (best, 1, 10)], f"loading {SPEC['name']}")
    model, tok = load_model(torch.bfloat16)
    directions = torch.load(os.path.join(out_dir, f"{key}.loop_directions.pt"), weights_only=True)
    baseline = RegressionBaseline(model, tok)
    problems = L.problems_from(REPO, seed=7, count=count)
    arms_rows = []

    def evaluate():
        rows = L.grade(tok, problems, L.think(model, tok, problems, budget))
        arms_rows.append(rows)
        scores = L.summarise(rows)
        scores.update(known=baseline.known_score(), drift=baseline.drift()[0])
        return scores

    session = EditSession(model, evaluate, goals=dict(repeats=-1), guards=dict(known=+1, drift=-1, finished=+1), tolerance=dict(repeats=0.02, finished=0.07, known=0.05, drift=0.05), log_path=path("loop_gen") + ".log.jsonl")
    t0 = time.time()
    base = session.measure_baseline()
    prog.tick(0, "unedited done")
    prog.result(f"{SPEC['name']} unedited, {count} problems x {budget} tokens: repeated tail {base['repeats']:.0%}, looping {base['looping']:.0%}, finished {base['finished']:.0%}, correct {base['correct']:.0%} | {time.time() - t0:.0f}s")
    attempt = session.attempt(best, [removal_update(model, l, directions[str(l)] if str(l) in directions else directions[l], alpha, writers) for l in layers])[0]
    before, after = arms_rows[0], arms_rows[1]
    correct = paired_test([r["correct"] for r in before], [r["correct"] for r in after])
    lower = sum(a["repeats"] - b["repeats"] > 0.02 for a, b in zip(before, after))
    higher = sum(b["repeats"] - a["repeats"] > 0.02 for a, b in zip(before, after))
    signs = paired_test([False] * lower + [True] * higher, [True] * lower + [False] * higher)
    s = attempt.scores
    save("loop_gen", dict(count=count, budget=budget, arm=best, unedited=base, edited=s, verdict=attempt.verdict, correct_test=vars(correct), repeats_lower=lower, repeats_higher=higher, repeats_p=signs.p_value, rows_before=before, rows_after=after))
    prog.tick(1, "edited done")
    prog.result(
        f"{SPEC['name']} {best}: repeated tail {s['repeats']:.0%} (lower on {lower}, higher on {higher}, p={signs.p_value:.3g}), looping {s['looping']:.0%}, finished {s['finished']:.0%}, "
        f"correct {s['correct']:.0%} (rescued {correct.rescued}, broken {correct.broken}, p={correct.p_value:.3g}), known {s['known']:.0%}, drift {s['drift']:.3f} | {attempt.verdict.upper()}"
    )
    prog.finish()


def direct_write(model, tok, fact, layer, strength, metric, base):
    links = answer_chain(model, tok, fact.teaching_examples(tok), layer)
    keys = torch.cat([l.keys for l in links], 1)
    deltas = torch.cat([(l.vector * strength)[:, None].expand(-1, l.keys.shape[1]) for l in links], 1)
    groups = [g for g, l in enumerate(links) for _ in range(l.keys.shape[1])]
    nulls = torch.cat([null_reads(model, tok, fact, layer, "last", "all"), answer_prefix_reads(model, tok, fact.target, layer)], 1)
    left, right = write_associations(keys, deltas, metric, 0.3, nulls, groups)
    writer = writer_module(model, layer)
    writer.weight.data.copy_((base.float() + left @ right).to(base.dtype))


def fact_rows(model, tok, facts, layer, strength, metric, baseline, stage_index, name):
    writer = writer_module(model, layer)
    base = writer.weight.data.clone()
    rows = []
    for i, fact in enumerate(facts):
        t0 = time.time()
        direct_write(model, tok, fact, layer, strength, metric, base)
        grid = monitor(model, tok, [fact], baseline, GateConfig())
        row = grid.to_dict()["facts"][0]
        row.update(known=grid.known, known_before=grid.known_baseline, drift=grid.drift, passed=bool(grid.passing))
        rows.append(row)
        writer.weight.data.copy_(base)
        prog.tick(stage_index, f"{name}: fact {i + 1}/{len(facts)}, recall {row['recall']:.2f}, same {row['same_subject']:.2f}, other {row['other_subject']:.2f}, {time.time() - t0:.0f}s")
    return rows


def mean_of(rows):
    out = {k: sum(r[k] for r in rows) / len(rows) for k in ("recall", "new_plain", "new_chat", "same_subject", "other_subject", "garbled", "known", "drift")}
    out.update(passed=sum(r["passed"] for r in rows), known_before=rows[0]["known_before"])
    return out


def line(name, m, n):
    return (f"{SPEC['name']} {name}: recall {m['recall']:.0%} (plain {m['new_plain']:.0%}, chat {m['new_chat']:.0%}, garbled {m['garbled']:.0%}) | leak same co. {m['same_subject']:.0%}, other co. {m['other_subject']:.0%} | "
            f"known {m['known']:.0%} (was {m['known_before']:.0%}) | drift {m['drift']:.3f} | PASS {m['passed']}/{n}")


def stage_facts_dev():
    global prog
    plan = load_json("map")
    layer = plan["fact_layer"]
    strengths = (1.0, 1.15, 1.5)
    prog = Progress([(f"strength {s}", 8, 8) for s in strengths], f"loading {SPEC['name']}")
    model, tok = load_model(SPEC["fact_dtype"])
    facts = [Fact.from_relation(s, t, r) for s, r, t in P.FACTS[:8]]
    cov_texts, _ = P.wikitext_samples()
    baseline = RegressionBaseline(model, tok)
    metric = metric_from_covariance(input_covariance(model, tok, cov_texts, [layer])[layer], 3.0)
    out = dict(layer=layer, setups={})
    with frozen_for_solving(model), torch.no_grad():
        for index, strength in enumerate(strengths):
            rows = fact_rows(model, tok, facts, layer, strength, metric, baseline, index, f"strength {strength}")
            m = mean_of(rows)
            out["setups"][str(strength)] = dict(mean=m, facts=rows)
            prog.result(line(f"no backprop, layer {layer}, strength {strength}, tuning facts", m, len(facts)))
            save("facts_dev", out)
    # Rule, fixed before the fresh test: the most passes, then recall minus same-company leak.
    best = max(out["setups"], key=lambda s: (out["setups"][s]["mean"]["passed"], out["setups"][s]["mean"]["recall"] - out["setups"][s]["mean"]["same_subject"]))
    out["chosen_for_fresh"] = float(best)
    save("facts_dev", out)
    prog.result(f"{SPEC['name']} chosen for the fresh-facts test, by the fixed rule: strength {best}")
    prog.finish()


def stage_facts_new():
    global prog
    plan, dev = load_json("map"), load_json("facts_dev")
    layer, strength = plan["fact_layer"], dev["chosen_for_fresh"]
    control_layer = round(len(range(plan["layers"])) / 3)
    prog = Progress([(f"no backprop, strength {strength}", 10, 8), (f"default backprop edit, layer {control_layer}", 10, 30)], f"loading {SPEC['name']}")
    model, tok = load_model(SPEC["fact_dtype"])
    facts = [fresh_fact(s, r, t) for s, r, t in FRESH]
    cov_texts, _ = P.wikitext_samples()
    baseline = RegressionBaseline(model, tok)
    metric = metric_from_covariance(input_covariance(model, tok, cov_texts, [layer])[layer], 3.0)
    out = dict(layer=layer, strength=strength, control_layer=control_layer)
    with frozen_for_solving(model), torch.no_grad():
        rows = fact_rows(model, tok, facts, layer, strength, metric, baseline, 0, "no backprop")
    m = mean_of(rows)
    out["direct"] = dict(mean=m, facts=rows)
    prog.result(line(f"FRESH no backprop, layer {layer}, strength {strength}", m, len(facts)))
    save("facts_new", out)
    editor = KnowledgeEditor(model, tok, calibration_texts=cov_texts, settings=EditSettings(layer=control_layer), gates=GateConfig(), progress=False)
    rows = []
    for i, fact in enumerate(facts):
        result = editor.add_fact(fact, on_failure="keep")
        grid = editor.grid()["all"]
        row = grid.to_dict()["facts"][0]
        row.update(known=grid.known, known_before=grid.known_baseline, drift=grid.drift, passed=result.report.passed)
        rows.append(row)
        editor.remove_fact(result.fact_id)
        prog.tick(1, f"backprop control: fact {i + 1}/{len(facts)}, recall {row['recall']:.2f}, same {row['same_subject']:.2f}")
    m = mean_of(rows)
    out["backprop"] = dict(mean=m, facts=rows)
    prog.result(line(f"FRESH default backprop edit, layer {control_layer}", m, len(facts)))
    save("facts_new", out)
    prog.finish()


try:
    dict(map=stage_map, **{"loop-scan": stage_loop_scan, "loop-gen": stage_loop_gen, "facts-dev": stage_facts_dev, "facts-new": stage_facts_new})[stage]()
except Exception:
    trace = traceback.format_exc()
    print(trace, flush=True)
    if prog is not None:
        prog.finish(trace.strip().splitlines()[-1])
    raise
print("DONE", flush=True)

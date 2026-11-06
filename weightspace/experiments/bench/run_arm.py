"""Runs one arm of the fact bench (plan.md) on a range of facts, appending one row per fact to results/bench/<arm>.jsonl.

Arms:
  base        the unedited model; its answers are the "before" answers of every other arm
  linear      the shipped recipe through Surgery.propose_fact: plain training wordings, layer 20, built at 1.0, applied at 1.25, protect_names
  linear_neg  the same build plus the bench's training negatives as extra leave-alone reads

Facts already in the output file are skipped, so a run can be split into chunks.
Usage: python -I run_arm.py <arm> <model dir> <calibration txt> <sessions dir> [start] [end]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
from weightspace import Fact  # noqa: E402
from weightspace.direct import answer_chain, answer_prefix_reads  # noqa: E402
from weightspace.facts import capture_writer, null_reads  # noqa: E402
from weightspace.isolation import applied  # noqa: E402
from weightspace.model_access import Prompt, render_chat, writer_param_name  # noqa: E402
from weightspace.operators import write_associations  # noqa: E402
from weightspace.patch import LowRankUpdate  # noqa: E402
from weightspace.surgery import Surgery  # noqa: E402

LAYER, STRENGTH = 20, 1.25


def training_fact(f: dict, chat: bool = False) -> Fact:
    """Only the training wordings: nothing from the test sets reaches the build. `chat`: also teach from the training chat questions."""
    return Fact(subject=f["subject"], target=f["target"], prompts=list(f["train"]["plain"]), questions=list(f["train"]["chat"]) if chat else [])


def bench_negatives(f: dict, tokenizer) -> list[Prompt]:
    t = f["train"]
    out = [Prompt(p) for p in t["same_plain"]] + [render_chat(tokenizer, q) for q in t["same_chat"]]
    for c in t["lookalikes"] + t["others"]:
        out += [Prompt(tp.format(s=c)) for tp in t["templates_plain"]] + [render_chat(tokenizer, tc.format(s=c)) for tc in t["templates_chat"]]
        if f["person"]:
            out += [Prompt(tp.format(s=c) + slot) for tp in t["templates_plain"][:2] for slot in B.NAME_SLOTS]
    return out


def build_linear(s: Surgery, model, tokenizer, f: dict, extra: list[Prompt], chat: bool = False) -> LowRankUpdate:
    """The body of Surgery._propose_fact with optional extra leave-alone prompts, read at their last token."""
    built = training_fact(f, chat)
    links = answer_chain(model, tokenizer, built.teaching_examples(tokenizer), LAYER)
    keys = torch.cat([link.keys for link in links], 1)
    deltas = torch.cat([link.vector[:, None].expand(-1, link.keys.shape[1]) for link in links], 1)
    groups = [g for g, link in enumerate(links) for _ in range(link.keys.shape[1])]
    parts = [null_reads(model, tokenizer, built, LAYER, "last", "all", name_slots=True), answer_prefix_reads(model, tokenizer, built.target, LAYER)]
    if extra:
        parts.append(torch.stack([capture_writer(model, tokenizer, p, LAYER)[0][-1] for p in extra], 1))
    nulls = torch.cat([p for p in parts if p is not None], 1)
    left, right = write_associations(keys, deltas, s._metric(LAYER), 0.3, nulls, groups)
    return LowRankUpdate(writer_param_name(model, LAYER), left.cpu(), right.cpu())


def main():
    arm, model_dir, calibration, sessions = sys.argv[1:5]
    start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    end = int(sys.argv[6]) if len(sys.argv) > 6 else 40
    sys.stdout.reconfigure(errors="replace")
    os.makedirs(B.RESULTS, exist_ok=True)
    out_path = os.path.join(B.RESULTS, f"{arm}.jsonl")
    done = B.read_rows(out_path)
    facts = [f for f in B.load_bench()[start:end] if f["id"] not in done]
    if not facts:
        print(f"{arm}: facts {start + 1}-{end} already done", flush=True)
        return
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = B.RegressionBaseline(model, tokenizer)
    base_rows = B.read_rows(os.path.join(B.RESULTS, "base.jsonl"))
    s = None
    if arm != "base":
        with open(calibration, encoding="utf-8") as fh:
            texts = [line.strip() for line in fh if line.strip()]
        log = os.path.join(sessions, f"surgery_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl")
        s = Surgery(model, tokenizer, calibration_texts=texts, budget=400, log_path=log, model_name=os.path.basename(model_dir), echo=False)
    print(f"{arm}: loaded in {time.time() - t0:.0f}s, known facts unedited {baseline.known_accuracy:.2f}, {len(facts)} facts to run", flush=True)
    for f in facts:
        t1 = time.time()
        row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm=arm, known_before=baseline.known_accuracy)
        if arm == "base":
            scored = B.score_fact(model, tokenizer, f, baseline, before=None)
            row["excluded"] = sum(B.mentions(t, f["target"]) for t in scored["answers"]["recall"]) >= 2
        else:
            if f["id"] not in base_rows:
                raise SystemExit(f"run the base arm for {f['id']} first")
            before = base_rows[f["id"]]["answers"]
            if arm == "linear":
                spec = dict(subject=f["subject"], target=f["target"], prompts=list(f["train"]["plain"]), questions=[])
                result = s.call("propose_fact", dict(name=f["id"], fact=spec, layer=LAYER, strength=1.0, protect_names=True))
                if "error" in result:
                    raise SystemExit(f"{f['id']}: {result['error']}")
                updates = s.proposals[f["id"]].updates
                if f is facts[0]:
                    # The replica used by linear_neg must equal propose_fact when given no extra prompts.
                    replica = build_linear(s, model, tokenizer, f, [])
                    a, b = updates[0].left.float() @ updates[0].right.float(), replica.left.float() @ replica.right.float()
                    row["replica_max_abs_diff"] = float((a - b).abs().max())
                    row["replica_rel_diff"] = float((a - b).norm() / a.norm())
            elif arm == "linear_neg":
                updates = [build_linear(s, model, tokenizer, f, bench_negatives(f, tokenizer))]
            else:
                raise SystemExit(f"unknown arm {arm}")
            with applied(model, updates, dose=STRENGTH):
                scored = B.score_fact(model, tokenizer, f, baseline, before=before)
        row["measures"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in scored["measures"].items()}
        row["failed"] = B.failures(scored["measures"], baseline.known_accuracy) if arm != "base" else []
        row["passed"] = arm != "base" and not row["failed"]
        row["answers"] = scored["answers"]
        row["seconds"] = round(time.time() - t1, 1)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        m = row["measures"]
        fmt = lambda v: "  -  " if v is None else f"{v:.2f}"
        verdict = ("EXCLUDED (base already says it)" if row.get("excluded") else "") if arm == "base" else ("PASS" if row["passed"] else "fail: " + ", ".join(row["failed"]))
        extra = f"  replica diff {row['replica_rel_diff']:.1e}" if "replica_rel_diff" in row else ""
        print(f"[{arm}] {f['id']} {f['subject']:<23} recall {fmt(m['recall'])} same {fmt(m['same'])} look {fmt(m['lookalike'])} other {fmt(m['other'])} "
              f"names {fmt(m['names'])} garb {fmt(m['garbled'])} known {m['known']:.2f} drift {m['drift']:.3f}  {row['seconds']:.0f}s  {verdict}{extra}", flush=True)
    if s is not None:
        print(f"{arm}: session log {os.path.basename(s.log_path) if hasattr(s, 'log_path') else ''}, chain intact {s.log_is_intact()}", flush=True)


if __name__ == "__main__":
    main()

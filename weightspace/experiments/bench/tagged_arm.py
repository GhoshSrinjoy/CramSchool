"""Arm "tagged" of the fact bench (plan.md, "Revision after a 2-fact smoke test"). Forward passes and closed-form solves only.

Part 1, layer 4: one reused neuron fires only on the last token of this company's name and writes a tag (the top direction of the
layer 4 -> 20 map, 2x the typical MLP output size there). Then the linear_neg write of run_arm.py is built with Part 1 in place, so its
leave-alone reads (look-alikes and unrelated companies, without a tag) differ from the fact's own reads by the tag.

Set BENCH_CHAT=1 for arm "tagged_chat", which also teaches from the training chat questions.
Usage: python -I tagged_arm.py <model dir> <calibration txt> <transport.json> <sessions dir> [start] [end]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
import gated_arm as G  # noqa: E402
import run_arm as R  # noqa: E402
import separability as S  # noqa: E402
import subject_site as SS  # noqa: E402
from weightspace.isolation import applied  # noqa: E402
from weightspace.surgery import Surgery  # noqa: E402

LS, LQ, TAG_SIZE = 4, R.LAYER, 2.0
S.LAYERS = [LS]
CHAT = os.environ.get("BENCH_CHAT") == "1"
ARM = ("tagged_chat" if CHAT else "tagged") + B.ARM_SUFFIX


def part1(model, tokenizer, f, ctx):
    """The name neuron at layer 4, as specified for the gated build."""
    train_i, test_i = SS.items_for(f, tokenizer)
    groups = [(side, g, its) for side, d in (("train", train_i), ("test", test_i)) for g, its in d.items()]
    reads = SS.subject_reads(model, tokenizer, [it for _, _, its in groups for it in its])[LS]
    tr, te, i = {}, {}, 0
    for side, g, its in groups:
        (tr if side == "train" else te)[g] = reads[i:i + len(its)]
        i += len(its)
    c, cbar = ctx["const"]
    held = ctx["ords"][LS]["held"]
    w = G.lda(tr["pos"], [tr["lookalike"], tr["other"]], ctx["ords"][LS]["cov"])
    min_pos = float((tr["pos"] @ w).min())
    max_neg = float((torch.cat([tr["lookalike"], tr["other"], held]) @ w).max())
    tau = max_neg + G.SUBJECT_PLACE * (min_pos - max_neg)
    a = G.SHARP / max(abs(min_pos - tau), 1e-6)
    gate, up = a * (w - tau * c / cbar), c / cbar
    act = lambda x: F.silu(x @ gate) * (x @ up)
    unit = float(act(tr["pos"]).mean())
    j = ctx["free"][0]
    updates = [G.replace_rows(model, LS, "gate_proj", [j], gate[None]), G.replace_rows(model, LS, "up_proj", [j], up[None]),
               G.replace_cols(model, LS, "down_proj", [j], (ctx["tag"] / unit)[:, None])]
    fires = lambda x: float((act(x) > 0.1 * unit).float().mean())
    info = dict(neuron=j, separable_on_training=min_pos > max_neg, test_wordings=fires(te["pos"]), same_company=fires(te["same"]),
                test_lookalikes=fires(te["lookalike"]), test_unrelated=fires(te["other"]), ordinary_tokens=fires(held))
    return updates, info


def main():
    model_dir, calibration, transport_json, sessions = sys.argv[1:5]
    start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    end = int(sys.argv[6]) if len(sys.argv) > 6 else 40
    sys.stdout.reconfigure(errors="replace")
    out_path = os.path.join(B.RESULTS, f"{ARM}.jsonl")
    done = B.read_rows(out_path)
    facts = [f for f in B.load_bench()[start:end] if f["id"] not in done]
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    baseline = B.make_baseline(model, tokenizer)
    base_rows = B.read_rows(os.path.join(B.RESULTS, "base.jsonl"))
    with open(calibration, encoding="utf-8") as fh:
        paragraphs = [line.strip() for line in fh if line.strip()]
    ords, _ = S.ordinary(model, tokenizer, paragraphs)
    saved = torch.load(transport_json + ".maps.pt")
    tag = torch.linalg.svd(saved["maps"][LS][LQ].cuda())[2][0] * saved["report"][f"mlp_out_norm_L{LS}"] * TAG_SIZE
    ctx = dict(ords=ords, tag=tag, const=G.constant_direction(ords[LS])[:2], free=G.neuron_order(model, tokenizer, paragraphs, LS))
    log = os.path.join(sessions, f"surgery_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl")
    s = Surgery(model, tokenizer, calibration_texts=paragraphs, budget=400, log_path=log, model_name=os.path.basename(model_dir), echo=False)
    print(f"{ARM}: ready in {time.time() - t0:.0f}s, {len(facts)} facts to run", flush=True)
    for f in facts:
        t1 = time.time()
        p1, info = part1(model, tokenizer, f, ctx)
        with applied(model, p1):
            write = [R.build_linear(s, model, tokenizer, f, R.bench_negatives(f, tokenizer), chat=CHAT)]
            with applied(model, write, dose=R.STRENGTH):
                scored = B.score_fact(model, tokenizer, f, baseline, before=base_rows[f["id"]]["answers"])
        row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], target=f["target"], arm=ARM, known_before=baseline.known_accuracy,
                   measures={k: (round(v, 4) if isinstance(v, float) else v) for k, v in scored["measures"].items()}, part1=info,
                   edit=G.digest(p1 + write))
        row["failed"] = B.failures(scored["measures"], baseline.known_accuracy)
        row["passed"] = not row["failed"]
        row["answers"] = scored["answers"]
        row["seconds"] = round(time.time() - t1, 1)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        m = row["measures"]
        fmt = lambda v: "  -  " if v is None else f"{v:.2f}"
        print(f"[{ARM}] {f['id']} {f['subject']:<23} recall {fmt(m['recall'])} same {fmt(m['same'])} look {fmt(m['lookalike'])} other {fmt(m['other'])} "
              f"names {fmt(m['names'])} garb {fmt(m['garbled'])}/{fmt(m['garbled_other'])} known {m['known']:.2f} drift {m['drift']:.3f} | name gate: test {info['test_wordings']:.2f} "
              f"look-alikes {info['test_lookalikes']:.2f} same co. {info['same_company']:.2f}  {row['seconds']:.0f}s  "
              f"{'PASS' if row['passed'] else 'fail: ' + ', '.join(row['failed'])}", flush=True)
    print(f"{ARM}: session log {os.path.basename(log)}", flush=True)


if __name__ == "__main__":
    main()

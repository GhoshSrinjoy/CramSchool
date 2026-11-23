"""A bench of facts beyond companies (cramschool.bench.kinds), run as the plan in the lab notebook says. GPU. Stages:

  base    the unedited model on every fact (bench.methods.run_base): <out>/base.jsonl
  graft   every fact added through the API (Graft.add) into one table, taught from the drafted wordings alone (built-in relations plus
          --wordings), with the fast build; scored with every check: <out>/graft.jsonl, the table in <out>/table.graft
  chunks  the table split into tables of at most --size facts, and the bench split the same way, for the grafted model
          (scripts/grafted_bench.py compiles and scores each; scripts/grafted_disk_bench.py scores an exported one from disk)

Forward passes and closed-form solves only.

Usage: python scripts/kinds_bench_run.py <stage> --model <folder> --bench <bench json> --out <folder> [--wordings <json>] [--size 32]
"""
import argparse
import json
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft, GraftOptions
from cramschool.bench.methods import run_base
from cramschool.bench.scoring import load_bench, make_baseline, read_rows, score_table_fact
from cramschool.graft.wordings import Wordings


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


def stage_graft(args):
    facts = load_bench(args.bench)
    base = read_rows(os.path.join(args.out, "base.jsonl"))
    missing = [f["id"] for f in facts if f["id"] not in base]
    if missing:
        raise SystemExit(f"run the base stage first: {len(missing)} facts have no unedited answers")
    model, tok = load(args.model)
    baseline = make_baseline(model, tok)
    wordings = Wordings.builtin().with_file(args.wordings) if args.wordings else Wordings.builtin()
    table = os.path.join(args.out, "table.graft")
    if os.path.exists(os.path.join(table, "graft.json")):
        graft = Graft.load(table, model, tok, progress=lambda s: log("  " + s))
        log(f"loaded the saved table: {len(graft)} facts")
    else:
        graft = Graft(model, tok, wordings=wordings, options=GraftOptions(fast=True, last_layer=None),  # benchmix1 was measured with the v1 write
                      progress=lambda s: log("  " + s))
        t0 = time.time()
        for n, f in enumerate(facts, 1):
            graft.add(f["subject"], f["relation"], f["target"])
            log(f"[build {n}/{len(facts)}] {f['id']} {f['relation']} {f['subject']} | {time.time() - t0:.0f}s")
        graft.save(table)
    out = os.path.join(args.out, "graft.jsonl")
    done = read_rows(out)
    rows = list(done.values())
    with graft.attached():
        for n, f in enumerate(facts, 1):
            if f["id"] in done:
                continue
            m, failed, answers, where = score_table_fact(model, tok, f, baseline, base[f["id"]]["answers"], facts)
            row = dict(id=f["id"], subject=f["subject"], relation=f["relation"], kind=f.get("kind", "company"), answer_type=f.get("answer_type"),
                       target=f["target"], arm="graft_api", measures=m, failed=failed, passed=not failed, answers=answers, cross_where=where)
            rows.append(row)
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            near = f" near {m['near']:.2f}" if m.get("near") is not None else ""
            log(f"[score {n}/{len(facts)}] {f['id']} {f.get('kind', 'company'):<11} {f['relation']:<18} recall {m['recall']:.2f} same {m['same']:.2f} "
                f"look {m['lookalike']:.2f} other {m['other']:.2f}{near} cross {m['cross']:.2f} | {'PASS' if not failed else 'fail: ' + ', '.join(failed)}")
    log(f"inserted layer: {sum(r['passed'] for r in rows)} of {len(rows)} pass; recall median {statistics.median(r['measures']['recall'] for r in rows):.3f}")


def stage_chunks(args):
    facts = load_bench(args.bench)
    with open(args.bench, encoding="utf-8") as fh:
        meta = json.load(fh)["meta"]
    model, tok = load(args.model)
    whole = Graft.load(os.path.join(args.out, "table.graft"), model, tok)
    subjects = list(dict.fromkeys(f["subject"] for f in facts))  # a subject's facts stay in one table: its switch reads its name once
    chunks, cur = [], []
    for s in subjects:
        group = [f for f in facts if f["subject"] == s]
        if cur and len(cur) + len(group) > args.size:
            chunks.append(cur)
            cur = []
        cur += group
    if cur:
        chunks.append(cur)
    for i, chunk in enumerate(chunks, 1):
        keep = {(f["subject"], f["relation"]) for f in chunk}
        part = Graft.load(os.path.join(args.out, "table.graft"), model, tok)
        for f in facts:
            if (f["subject"], f["relation"]) not in keep:
                part.remove(f["subject"], f["relation"])
        assert len(part) == len(chunk), (len(part), len(chunk))
        part.save(os.path.join(args.out, f"chunk{i}.graft"))
        with open(os.path.join(args.out, f"chunk{i}.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(meta=dict(meta, chunk=i, of=len(chunks)), facts=chunk), fh, ensure_ascii=False, indent=1)
        log(f"chunk {i}: {len(chunk)} facts on {len({f['subject'] for f in chunk})} subjects")
    del whole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("base", "graft", "chunks"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wordings")
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.stage == "base":
        model, tok = load(args.model)
        run_base(model, tok, args.bench, args.out, log=log)
    elif args.stage == "graft":
        stage_graft(args)
    else:
        stage_chunks(args)


if __name__ == "__main__":
    main()

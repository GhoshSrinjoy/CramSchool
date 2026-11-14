"""Where do the grafted model's answers stop on a bench? (Lab notebook, step 7.) For every recall reply of a run of
scripts/grafted_bench.py: how many of the answer's tokens came out in a row, and where an answer broke off, the continuation detector's
score z at that position (read by feeding the prompt and the reply up to the break back to the unedited model).

Usage: python scripts/diag_step7_breaks.py <model folder> <bench json> <table folder> <rows.jsonl> <out json>
"""
import collections
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.bench.scoring import load_bench, read_rows, test_prompts
from cramschool.graft.build import fact_detectors
from cramschool.graft.core import layer_reads, ratio
from cramschool.model_access import encode


def main():
    model_dir, bench_path, table_dir, rows_path, out_path = sys.argv[1:6]
    facts, rows = load_bench(bench_path), read_rows(rows_path)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    graft = Graft.load(table_dir, model, tok)
    mctx = graft.calibration()
    c, cbar = mctx["c"], mctx["cbar"]
    conts = graft.continuations()
    by_fact = {(e["fact"]["subject"], e["fact"]["relation"]): (e["parts"], k) for e, k in zip(graft._facts, conts)}
    text_of = lambda i: tok.decode([i])  # noqa: E731

    out, seqs, where = [], [], []
    for f in facts:
        parts, cont = by_fact[(f["subject"], f["relation"])]
        det = {(k["step"], k["token"]): k for k in cont}
        firsts = {d["form"]: d for d in fact_detectors(parts)}
        spell = {sp: s["ids"] for sp, s in parts["spellings"].items() if s is not None}
        prompts = test_prompts(f, tok)["recall"]
        n_plain = len(f["test"]["plain"])
        for i, (p, reply) in enumerate(zip(prompts, rows[f["id"]]["answers"]["recall"])):
            p_ids, r_ids = encode(tok, p), tok(reply, add_special_tokens=False).input_ids
            best = None  # (answer tokens in a row, where the answer starts in the reply, spelling)
            for sp, ids in spell.items():
                for at, t in enumerate(r_ids):
                    if t != ids[0]:
                        continue
                    n = 1
                    while n < len(ids) and at + n < len(r_ids) and r_ids[at + n] == ids[n]:
                        n += 1
                    if best is None or n > best[0]:
                        best = (n, at, sp)
            row = dict(id=f["id"], relation=f["relation"], form="plain" if i < n_plain else "chat", target=f["target"],
                       whole=f["target"].lower() in reply.lower(), reply=reply[:60])
            if best is None:
                row.update(kind="no first token", first_detectors=sorted(firsts))
            else:
                n, at, sp = best
                ids = spell[sp]
                if n == len(ids):
                    row.update(kind="whole", spelling=sp, at=at)
                elif at + n >= len(r_ids):
                    row.update(kind="cut off by the reply's length", spelling=sp, at=at, done=n, of=len(ids))
                else:
                    key = (n, ids[n])
                    k = det.get(key)
                    row.update(kind="broke off", spelling=sp, at=at, done=n, of=len(ids), wanted=text_of(ids[n]), came=text_of(r_ids[at + n]),
                               after=text_of(ids[n - 1]), detector=None if k is None else ("built" if k["delta"] is not None else "not separable"))
                    if k is not None and k["delta"] is not None:
                        seqs.append(p_ids + r_ids[:at + n])
                        where.append((len(out), k))
            out.append(row)
    if seqs:
        reads = layer_reads(model, tok, seqs, [[len(s) - 1] for s in seqs], graft.layer)
        for x, (j, k) in zip(reads, where):
            out[j]["z"] = float((ratio(x, k["w"].to(x.device), c, cbar)[0] - k["tau"]) / k["delta"])
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    kinds = collections.Counter(r["kind"] for r in out)
    print(f"{len(out)} recall replies: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    print(f"whole answer by the bench's own match: {sum(r['whole'] for r in out)}")
    broke = [r for r in out if r["kind"] == "broke off"]
    print(f"broke off ({len(broke)}): no detector for that token {sum(r['detector'] != 'built' for r in broke)} | detector shut there (z <= 0) "
          f"{sum('z' in r and r['z'] <= 0 for r in broke)} | detector open (z > 0) and another token came {sum('z' in r and r['z'] > 0 for r in broke)}")
    zs = sorted(r["z"] for r in broke if "z" in r)
    if zs:
        print(f"  z at the break: lowest {zs[0]:.2f}, median {zs[len(zs) // 2]:.2f}, highest {zs[-1]:.2f}")
    print("  by the step that failed (answer tokens already out):", dict(sorted(collections.Counter(r["done"] for r in broke).items())))
    for r in broke[:14]:
        print(f"   {r['id']} {r['form']:5s} {r['target']!r}: after {r['after']!r} wanted {r['wanted']!r}, came {r['came']!r} | detector {r['detector']} "
              f"z {r.get('z', float('nan')):.2f} | {r['reply'][:44]!r}")
    cont_all = [k for ks in conts for k in ks]
    bad = [k for k in cont_all if k["delta"] is None]
    print(f"continuation tokens with no usable detector: {len(bad)} of {len(cont_all)}: " + ", ".join(
        f"step {k['step']} {text_of(k['token'])!r} (median {k['med']:.2f} under the threshold {k['tau']:.2f}, {k['n_pos']} taught)" for k in bad[:20]))


if __name__ == "__main__":
    main()

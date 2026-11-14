"""Why does the grafted model's first answer token land less often than the inserted layer's on bench40e? For every recall prompt: the
fact's detector scores z at the prompt's last token (plain and chat detectors), next to what each model did (from saved replies).

Usage: python scripts/diag_first_token.py <model folder> <repo folder> <grafted probe jsonl> <out json>
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.bench.scoring import load_bench, read_rows, test_prompts
from cramschool.graft.build import fact_detectors
from cramschool.graft.core import last_reads, ratio

MODEL, REPO, PROBE, OUT = sys.argv[1:5]
facts = load_bench(f"{REPO}/benchmarks/bench40e.json")
addon = read_rows(f"{REPO}/runs/api_facts/graft_api.jsonl")
grafted = read_rows(PROBE)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
graft = Graft.load(f"{REPO}/runs/api_facts/bench40e.graft", model, tok)
mctx = graft.calibration()
c, cbar = mctx["c"], mctx["cbar"]
parts_of = {(e["fact"]["subject"], e["fact"]["relation"]): e["parts"] for e in graft._facts}
rows = []
for f in facts:
    parts = parts_of[(f["subject"], f["relation"])]
    dets = {d["form"]: d for d in fact_detectors(parts)}
    prompts = test_prompts(f, tok)["recall"]
    x = last_reads(model, tok, prompts, graft.layer)
    z = {form: ((ratio(x, d["w"].to(x.device), c, cbar) - d["tau"]) / d["delta"]).tolist() for form, d in dets.items()}
    sp = tok(" " + f["target"], add_special_tokens=False).input_ids
    bare = tok(f["target"], add_special_tokens=False).input_ids
    f_sp, f_b = tok.decode(sp[:1]), tok.decode(bare[:1])
    n_plain = len(f["test"]["plain"])
    for i in range(len(prompts)):
        g, a = grafted[f["id"]]["answers"]["recall"][i], addon[f["id"]]["answers"]["recall"][i]
        rows.append(dict(id=f["id"], form="plain" if i < n_plain else "chat", z_plain=z.get("plain", [None] * len(prompts))[i],
                         z_chat=z.get("chat", [None] * len(prompts))[i], addon_whole=f["target"].lower() in a.lower(),
                         grafted_first=g.startswith(f_sp) or g.startswith(f_b), grafted_reply=g[:50], addon_reply=a[:50],
                         detectors=sorted(dets)))
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(rows, fh, ensure_ascii=False, indent=1)


def own(r):
    return r["z_plain"] if r["form"] == "plain" else r["z_chat"]


def best(r):
    return max(v for v in (r["z_plain"], r["z_chat"]) if v is not None)


def bins(vals):
    edges = [-1e9, 0, 0.25, 0.5, 1, 2, 3, 1e9]
    names = ["<=0", "0-0.25", "0.25-0.5", "0.5-1", "1-2", "2-3", ">3"]
    return {n: sum(lo < v <= hi for v in vals) for n, lo, hi in zip(names, edges, edges[1:])}


print(f"{len(rows)} recall prompts; facts without a plain detector: {sum('plain' not in r['detectors'] for r in rows) // 12}, without a chat one: "
      f"{sum('chat' not in r['detectors'] for r in rows) // 12}")
for form in ("plain", "chat"):
    rs = [r for r in rows if r["form"] == form and own(r) is not None]
    print(f"\n{form} prompts ({len(rs)}): own-form detector z at the prompt's end")
    print("  all prompts:                         ", bins([own(r) for r in rs]))
    print("  grafted first token landed:          ", bins([own(r) for r in rs if r["grafted_first"]]))
    print("  add-on whole answer, grafted missed: ", bins([own(r) for r in rs if r["addon_whole"] and not r["grafted_first"]]))
    miss = [r for r in rs if r["addon_whole"] and not r["grafted_first"]]
    other = "z_chat" if form == "plain" else "z_plain"
    print(f"  of those {len(miss)} misses: own detector shut but the other form's open: "
          f"{sum(own(r) <= 0 and r[other] is not None and r[other] > 0 for r in miss)}; both shut at the prompt's end: "
          f"{sum(best(r) <= 0 for r in miss)}; own open: {sum(own(r) > 0 for r in miss)}")
    for r in miss[:6]:
        print(f"     z_plain {r['z_plain']:.2f} z_chat {r['z_chat'] if r['z_chat'] is None else round(r['z_chat'], 2)} | grafted {r['grafted_reply']!r} | add-on {r['addon_reply']!r}")

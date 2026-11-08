"""Does the fast build (a ReadCache shared across facts) build the same facts? Rebuilds chosen facts of a CounterFact run with the cache and
compares them with the run's saved parts, built without it: detector directions, thresholds, which of each fact's own prompts open, and the
time per fact. GPU, a few minutes.

Usage: python scripts/check_fast_build.py <model folder> <run folder> <relations.json> <more generic.json> [context] [facts per relation]
"""
import collections
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.methods import calibrate
from cramschool.graft.build import build_context, build_parts
from cramschool.graft.core import ReadCache, last_reads, ratio
from cramschool.graft.wordings import Wordings, norm, normalized
from cramschool.model_access import Prompt


def main():
    model_dir, run_dir, relations_path, more_path = sys.argv[1:5]
    context = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    per = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    with open(os.path.join(run_dir, "facts.json"), encoding="utf-8") as fh:
        facts = json.load(fh)
    wordings = Wordings.builtin().with_file(relations_path)
    with open(more_path, encoding="utf-8") as fh:
        more = json.load(fh)["kinds"]
    for kind, spec in more.items():
        k = wordings.kinds[kind]
        seen = {normalized(x) for x in k["generic_plain"] + k["generic_chat"]}
        k["generic_plain"] = k["generic_plain"] + [x for x in spec["generic_plain"] if normalized(x) not in seen]
        k["generic_chat"] = k["generic_chat"] + [x for x in spec["generic_chat"] if normalized(x) not in seen]
    guarded = Wordings(relations=wordings.relations, generic=wordings.generic, kinds=wordings.kinds,
                       guard={norm(p) for f in facts for p in f["cf"]["neighbours"]})
    cache = ReadCache()
    ctx = build_context(facts, guarded, name="counterfact", real_world="other_relations", context=context, cache=cache)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    mctx = calibrate(model, tok, 20)
    chosen, seen_rel = [], collections.Counter()
    for f in facts:
        if seen_rel[f["relation"]] < per:
            chosen.append(f)
            seen_rel[f["relation"]] += 1
    cos, dtau, same_open, total_open, times, slow = [], [], 0, 0, [], []
    for f in chosen:
        t0 = time.time()
        fast = build_parts(model, tok, f, ctx, mctx, 20)
        times.append(time.time() - t0)
        saved = torch.load(os.path.join(run_dir, "parts", f"{f['id']}.pt"), weights_only=True, map_location="cpu")
        slow.append(saved["seconds"])
        tests = [Prompt(t) for t in [f["cf"]["rewrite"]] + f["cf"]["paraphrases"]]
        x = last_reads(model, tok, tests, 20)
        for form in ("plain", "chat"):
            a, b = fast["forms"][form], saved["forms"][form]
            cos.append(float(torch.nn.functional.cosine_similarity(a["w"].float(), b["w"].float(), dim=0)))
            ta, tb = max(a["tau_neg"], a["tau_in"]), max(b["tau_neg"], b["tau_in"])
            dtau.append(abs(ta - tb) / max(abs(tb), 1e-6))
            if form == "plain":
                ra = ratio(x, a["w"].to(x.device), mctx["c"], mctx["cbar"]) > ta
                rb = ratio(x, b["w"].to(x.device), mctx["c"], mctx["cbar"]) > tb
                same_open += int((ra == rb).sum())
                total_open += len(tests)
        print(f"{f['id']} {f['relation']:<19} fast {times[-1]:5.1f}s (saved build {saved['seconds']:5.1f}s) | cache {cache.hits} hits, "
              f"{cache.misses} reads", flush=True)
    print(f"directions: cosine min {min(cos):.6f}, median {statistics.median(cos):.6f} | thresholds: relative change max {max(dtau):.2e} | "
          f"own prompts with the same open or shut: {same_open} of {total_open} | median build {statistics.median(times[1:] or times):.1f}s "
          f"against {statistics.median(slow):.1f}s", flush=True)


if __name__ == "__main__":
    main()

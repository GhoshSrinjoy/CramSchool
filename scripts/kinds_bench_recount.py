"""The after-the-fact counts of step 14 (lab notebook), reported beside the verdict and not judged. CPU only.

- Cross hits that are another subject's year: whether that subject is named in the prompt or the reply (if not, no stored fact can have
  pushed it), and the passes with those hits not counted, overall, by block, by answer type, and on untaught wordings only.
- The grafted chunks against the inserted layer on the same facts: whole answers, and passes with those hits not counted.
- Near questions: observed examples of a fact's own answer given to a near question, and how many facts would pass if near questions were
  held to the same-subject bar (0.25).

Usage: python scripts/kinds_bench_recount.py [--results research/results/benchmix1] [--bench benchmarks/benchmix1.json]
"""
import argparse
import os

from transformers import AutoTokenizer

from cramschool.bench.scoring import GROUPS, PASS, cross_hit, load_bench, read_rows, test_prompts
from cramschool.gates import mentions
from cramschool.graft.wordings import Wordings, normalized

MODEL = "D:/data/models/Qwen__Qwen3.5-0.8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="research/results/benchmix1")
    ap.add_argument("--bench", default="benchmarks/benchmix1.json")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    facts = {f["id"]: f for f in load_bench(args.bench)}
    by_target = {f["target"]: f for f in facts.values()}
    ins = read_rows(os.path.join(args.results, "graft.jsonl"))

    def coincidence(t, f):
        g = by_target[t]
        return g.get("answer_type") == "year" and g["subject"] != f["subject"]

    def real_hits(r, f):
        return sum(bool([t for t in w["targets"] if not coincidence(t, f)]) for w in r["cross_where"] if w["group"] != "near")

    def corrected(r, f, near_bar=False):
        others = [x for x in r["failed"] if not x.startswith(("cross", "known"))]
        ok = not others and real_hits(r, f) == 0
        if near_bar and r["measures"].get("near") is not None:
            ok = ok and r["measures"]["near"] <= PASS["same"] + 1e-9
        return ok

    total = named = 0
    for i, r in ins.items():
        f = facts[i]
        pr = test_prompts(f, tok)
        for w in r["cross_where"]:
            if w["group"] == "near":
                continue
            for t in w["targets"]:
                if coincidence(t, f):
                    total += 1
                    g = by_target[t]
                    named += g["subject"].lower() in (pr[w["group"]][w["index"]].text + " " + r["answers"][w["group"]][w["index"]]).lower()
    print(f"cross hits that are another subject's year: {total}; that subject named in the prompt or the reply: {named}")

    w = Wordings.builtin().with_file(os.path.join(args.results, "drafted_relations.json"))
    taught = {rel: {normalized(t) for t in v.plain + v.chat} for rel, v in w.relations.items()}
    blocks = dict(company=[i for i in ins if facts[i].get("kind") == "company"], other=[i for i in ins if facts[i].get("kind") != "company"])
    print("passes with those hits not counted:", {b: f"{sum(corrected(ins[i], facts[i]) for i in ids)} of {len(ids)}" for b, ids in blocks.items()},
          "| as scored:", {b: f"{sum(ins[i]['passed'] for i in ids)} of {len(ids)}" for b, ids in blocks.items()})
    for typ in ("person", "city", "org", "year", "phrase"):
        ids = [i for i in blocks["other"] if facts[i].get("answer_type") == typ]
        print(f"   {typ:<7} {sum(corrected(ins[i], facts[i]) for i in ids)} of {len(ids)} (as scored {sum(ins[i]['passed'] for i in ids)})")
    unt = {}
    for b, ids in blocks.items():
        n = 0
        for i in ids:
            f, r = facts[i], ins[i]
            keep = [normalized(t) not in taught[f["relation"]] for t in f["test"]["templates_plain"] + f["test"]["templates_chat"]]
            rep = [a for a, k in zip(r["answers"]["recall"], keep) if k]
            rec = sum(mentions(a, f["target"]) for a in rep) / len(rep) if rep else 0
            n += bool(rec >= PASS["recall"] and not [x for x in r["failed"] if not x.startswith(("cross", "recall", "known"))] and real_hits(r, f) == 0)
        unt[b] = f"{n} of {len(ids)}"
    print("on untaught wordings only, with those hits not counted:", unt)
    near_ok = {b: f"{sum(corrected(ins[i], facts[i], near_bar=True) for i in ids)} of {len(ids)}" for b, ids in blocks.items()}
    print("descriptive: passes if near questions were also held to the same-subject bar (0.25), with those hits not counted:", near_ok)

    # the cross check as fixed after this bench (scoring.cross_hit): recomputed from the stored replies, everything else as scored
    base = read_rows(os.path.join(args.results, "base.jsonl"))
    loaded = list(facts.values())
    fixed = {}
    for i, r in ins.items():
        f = facts[i]
        pr = test_prompts(f, tok)
        sib = next((g for g in loaded if g["subject"] == f["subject"] and g["target"] != f["target"]), None) if f.get("sibling") else None
        same_rel = f["test"].get("same_plain_rel", []) + f["test"].get("same_chat_rel", [])
        hits = total = 0
        for grp in GROUPS:
            if grp == "near":
                continue
            for k, (t, p) in enumerate(zip(r["answers"][grp], pr[grp])):
                total += 1
                b = base[i]["answers"][grp][k]
                bad = [g for g in loaded if g["target"] != f["target"] and not (sib is not None and g is sib and grp == "recall")
                       and not (sib is not None and g is sib and grp == "same" and k < len(same_rel) and same_rel[k] == g["relation"])
                       and cross_hit(t, p.text, b, g)]
                hits += bool(bad)
        others = [x for x in r["failed"] if not x.startswith(("cross", "known"))]
        fixed[i] = not others and hits == 0
    print("passes with the cross check as fixed:", {b: f"{sum(fixed[i] for i in ids)} of {len(ids)}" for b, ids in blocks.items()},
          "| by answer type:", {typ: f"{sum(fixed[i] for i in blocks['other'] if facts[i].get('answer_type') == typ)} of "
                                     f"{sum(facts[i].get('answer_type') == typ for i in blocks['other'])}" for typ in ("person", "city", "org", "year", "phrase")})

    tot = [0, 0, 0, 0, 0]
    for k in range(1, 10):
        path = os.path.join(args.results, f"grafted{k}", "rows.jsonl")
        if not os.path.exists(path):
            break
        g = read_rows(path)
        ids = list(g)
        vals = (sum(mentions(a, facts[i]["target"]) for i in ids for a in g[i]["answers"]["recall"]),
                sum(mentions(a, facts[i]["target"]) for i in ids for a in ins[i]["answers"]["recall"]),
                sum(corrected(g[i], facts[i]) for i in ids), sum(corrected(ins[i], facts[i]) for i in ids), len(ids))
        print(f"chunk {k}: whole answers grafted {vals[0]} / inserted {vals[1]} of {12 * vals[4]} | passes with those hits not counted: "
              f"grafted {vals[2]}, inserted {vals[3]}")
        tot = [a + b for a, b in zip(tot, vals)]
    print(f"all chunks: whole answers grafted {tot[0]} / inserted {tot[1]} of {12 * tot[4]} | passes grafted {tot[2]}, inserted {tot[3]} of {tot[4]}")

    shown = set()
    for i, r in ins.items():
        f = facts[i]
        if f.get("kind") in shown:
            continue
        for p, a in zip(test_prompts(f, tok)["near"], r["answers"]["near"]):
            if f["target"].lower() in a.lower():
                q = p.text.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0] if "<|im_start|>" in p.text else p.text
                print(f"near, {f.get('kind')}/{f['relation']}: {q.strip()!r} -> {a.strip()[:70]!r} (stored answer {f['target']!r})")
                shown.add(f.get("kind"))
                break


if __name__ == "__main__":
    main()

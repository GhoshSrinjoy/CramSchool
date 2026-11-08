"""Which threshold options open GRAFT's layer on more of the facts' own prompts without opening it on more other questions? For the edits of a
finished CounterFact run, each fact's detectors are built again both ways (`inside` = "negative", as v1, and "scatter"), keeping the run's
answer vectors, and six options are tried with every fact in one layer, shared detectors and veto included (lab notebook, "The weaker
relations"). Counted at the prompt's last token, through `generate` for one token as in scoring:
- own: the fact's rewrite and reworded prompts where the layer pushes the fact's own answer (estimated ES and PS);
- same: the fact's held-out same-subject questions where it pushes any answer (estimated leaks where the reply starts).
The parts built here are saved under <out>/<inside>/parts, so a development run with the chosen option can use them. GPU, about 15 minutes.

Usage: python scripts/diag_counterfact_options.py <model folder> <run folder> <relations.json> <more generic.json> <out folder> [context]
"""
import collections
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.methods import calibrate
from cramschool.graft.build import build_context, build_parts, build_shared, entry_of
from cramschool.graft.core import ReadCache
from cramschool.graft.layer import GraftLayer
from cramschool.graft.wordings import Wordings, norm, normalized
from cramschool.model_access import Prompt, batch_ids, encode, pad_id, render_chat

OPTIONS = {"O0": ("negative", None), "O1": ("negative", 0.95), "O2": ("negative", 0.90),
           "O3": ("scatter", 0.99), "O4": ("scatter", 0.95), "O5": ("scatter", 0.90)}
CHANGES = {"O0": 0, "O1": 1, "O2": 1, "O3": 1, "O4": 2, "O5": 2}  # ties go to fewer changes
LEAK_ALLOWANCE, KILL_GAIN = 0.03, 0.05


def log(s):
    print(s, flush=True)


def wordings_of(relations_path, more_path):
    """The breadth runs' wordings: drafted relations plus the extra generic questions, merged as scripts/counterfact_breadth.py does."""
    wordings = Wordings.builtin().with_file(relations_path)
    with open(more_path, encoding="utf-8") as fh:
        for kind, spec in json.load(fh)["kinds"].items():
            k = wordings.kinds[kind]
            seen = {normalized(x) for x in k["generic_plain"] + k["generic_chat"]}
            for g in ("generic_plain", "generic_chat"):
                k[g] = k[g] + [x for x in spec[g] if normalized(x) not in seen and x.count("{s}") == 1]
    return wordings


@torch.no_grad()
def pushes(model, tok, layer, prompts, batch=32):
    """For each prompt, the index of the fact whose answer the layer pushes at the prompt's last token, or None."""
    out = []
    filler = pad_id(tok)
    for s in range(0, len(prompts), batch):
        chunk = prompts[s:s + batch]
        ids, mask = batch_ids(tok, [encode(tok, p) for p in chunk], model.device, side="left")
        model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=1, do_sample=False, temperature=None, top_p=None, top_k=None,
                       pad_token_id=filler)
        out += [st["pending"][0] if st["pending"] is not None else None for st in layer.state]
    return out


def main():
    model_dir, run_dir, relations_path, more_path, out_dir = sys.argv[1:6]
    context = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    with open(os.path.join(run_dir, "facts.json"), encoding="utf-8") as fh:
        facts = json.load(fh)
    wordings = wordings_of(relations_path, more_path)
    guarded = Wordings(relations=wordings.relations, generic=wordings.generic, kinds=wordings.kinds,
                       guard=set(wordings.guard) | {norm(p) for f in facts for p in f["cf"]["neighbours"]})
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    mctx = calibrate(model, tok, 20)
    cache = ReadCache()
    ctxs = {mode: build_context(facts, guarded, name="counterfact", real_world="other_relations", context=context, cache=cache, inside=mode)
            for mode in ("negative", "scatter")}
    parts = {mode: {} for mode in ctxs}
    t0 = time.time()
    for k, f in enumerate(facts, 1):
        saved = torch.load(os.path.join(run_dir, "parts", f"{f['id']}.pt"), weights_only=True)
        for mode, ctx in ctxs.items():
            path = os.path.join(out_dir, mode, "parts", f"{f['id']}.pt")
            if os.path.exists(path):
                parts[mode][f["id"]] = torch.load(path, weights_only=True)
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            p = build_parts(model, tok, f, ctx, mctx, 20, vectors_from=saved)
            torch.save(p, path)
            parts[mode][f["id"]] = p
        a, b = parts["negative"][f["id"]]["forms"]["plain"], parts["scatter"][f["id"]]["forms"]["plain"]
        log(f"[build {k}/{len(facts)}] {f['id']} {f['relation']:<18} plain threshold, median: negative {max(a['tau_neg'], a['tau_in']):.1f}, "
            f"{a['med']:.1f} | scatter {max(b['tau_neg'], b['tau_in']):.1f}, {b['med']:.1f} | {time.time() - t0:.0f}s")
    shared = build_shared(model, tok, ctxs["negative"], mctx, 20)
    kinds = {r: rel.subject for r, rel in wordings.relations.items()}
    own, same, owner = [], [], []
    for i, f in enumerate(facts):
        for kind, text in [("rewrite", f["cf"]["rewrite"])] + [("para", p) for p in f["cf"]["paraphrases"]]:
            own.append((i, kind, Prompt(text)))
        for q in f["test"]["same_plain"]:
            same.append((i, "plain", Prompt(q)))
        for q in f["test"]["same_chat"]:
            same.append((i, "chat", render_chat(tok, q)))
    report = {}
    for name, (mode, q) in OPTIONS.items():
        entries = [entry_of(parts[mode][f["id"]], 2.0, end_quantile=q) for f in facts]
        layer = GraftLayer(model, tok, entries, shared, mctx["c"], mctx["cbar"], 0.5, 20, relation_kinds=kinds).attach()
        try:
            got_own = pushes(model, tok, layer, [p for _, _, p in own])
            got_same = pushes(model, tok, layer, [p for _, _, p in same])
        finally:
            layer.detach()
        by = collections.defaultdict(collections.Counter)
        for (i, kind, _), g in zip(own, got_own):
            rel = facts[i]["relation"]
            by[rel][kind] += 1
            by[rel][kind + "_open"] += int(g == i)
        for (i, form, _), g in zip(same, got_same):
            rel = facts[i]["relation"]
            by[rel]["same"] += 1
            by[rel]["same_open"] += int(g is not None)
            by[rel]["same_" + form + "_open"] += int(g is not None)
        rows = {}
        for rel, c in sorted(by.items()):
            rows[rel] = dict(es=c["rewrite_open"] / c["rewrite"], ps=c["para_open"] / c["para"], same=c["same_open"] / c["same"],
                             same_plain=c["same_plain_open"], same_chat=c["same_chat_open"], n_same=c["same"])
        tot = collections.Counter()
        for c in by.values():
            tot.update(c)
        report[name] = dict(inside=mode, end_quantile=q, by_relation=rows, es=tot["rewrite_open"] / tot["rewrite"], ps=tot["para_open"] / tot["para"],
                            same=tot["same_open"] / tot["same"], events=dict(layer.events))
        log(f"{name} (inside {mode}, end quantile {q}): estimated ES {report[name]['es']:.3f} PS {report[name]['ps']:.3f} | same-subject opened "
            f"{report[name]['same']:.3f} | " + "; ".join(f"{r[:12]} PS {v['ps']:.2f} same {v['same']:.2f}" for r, v in rows.items()))
    base = report["O0"]
    allowed = [n for n in OPTIONS if all(report[n]["by_relation"][r]["same"] <= base["by_relation"][r]["same"] + LEAK_ALLOWANCE + 1e-9
                                         for r in base["by_relation"])]
    best = max(allowed, key=lambda n: (round(report[n]["ps"], 6), -CHANGES[n]))
    gain = report[best]["ps"] - base["ps"]
    choice = dict(allowed=allowed, best=best, gain=round(gain, 4), go=gain >= KILL_GAIN)
    log(f"options within the leak allowance: {allowed}; best {best}, estimated PS {report[best]['ps']:.3f} against {base['ps']:.3f} for O0 "
        f"(gain {gain:+.3f}); {'GO: development run with ' + best if choice['go'] else 'STOP: the gain is under ' + str(KILL_GAIN)}")
    with open(os.path.join(out_dir, "options.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(options=report, choice=choice, rule=dict(leak_allowance=LEAK_ALLOWANCE, kill_gain=KILL_GAIN)), fh, indent=1)


if __name__ == "__main__":
    main()

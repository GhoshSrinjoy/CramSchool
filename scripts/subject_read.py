"""Step 27, the measurement before any build: can the subject be read at the question's last token, in the model's own state, well enough
to replace the name switch? (Lab notebook, "Step 27: the subject at the question's last token, layer by layer".)

For each of a dozen invented companies, a linear subject reading is fitted at every layer (the build's LDA and threshold rule): positives
are questions about the company, negatives the same questions about three look-alike names of it and about the other companies. It is
scored on wordings it was not fitted on: about the company (recall, also in lower and upper case), about three look-alike names it has
never seen (one more letter swap, another industry word, another first word), and about companies it has never seen.

A company passes when it meets the bench's bars: held-out recall at least 0.75 and at most 1 in 12 look-alike questions opened. Forward
passes and linear solves only. GPU, about 6 minutes.

Usage: python scripts/subject_read.py <model folder> <out json> [--companies N]
"""
import json
import os
import random
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.bench.build import one_letter_variants
from cramschool.graft.build import NEG_QUANTILE, SATURATION, chat_items, plain_items, qa_items
from cramschool.graft.core import blocks, lda, ratio
from cramschool.graft.layers import _Reader, calibration_at, split_wordings
from cramschool.graft.wordings import INDUSTRIES, Wordings, calibration_paragraphs, invented_companies, invented_words

SEED = 20261127
RECALL_BAR, LOOKALIKE_BAR = 0.75, 1 / 12


def lookalikes(company: str, rng: random.Random, spare_words: list[str]) -> tuple[list[str], list[str]]:
    """(three look-alikes to fit against, three to test on), made as the bench makes them: the first word with another industry, one letter
    of the first word changed, and another first word with the same industry."""
    w1, w2 = company.split()
    others = [i for i in INDUSTRIES if i != w2]
    rng.shuffle(others)
    swapped, inner = one_letter_variants(w1, rng)
    return ([f"{w1} {others[0]}", f"{swapped} {w2}", f"{spare_words[0]} {w2}"], [f"{w1} {others[1]}", f"{inner} {w2}", f"{spare_words[1]} {w2}"])


def main():
    model_dir, out_path = sys.argv[1], sys.argv[2]
    n = int(sys.argv[sys.argv.index("--companies") + 1]) if "--companies" in sys.argv else 12
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    layers = list(range(len(blocks(model))))
    t0 = time.time()

    fit, held, generic_held = split_wordings(Wordings.builtin())
    rng = random.Random(SEED)
    companies = invented_companies([], n=n, seed=SEED)
    unseen = invented_companies(companies, n=6, seed=SEED + 1)            # companies no reading was fitted against
    spare = invented_words(rng, 2 * n, [c.split()[0] for c in companies + unseen])
    looks = {c: lookalikes(c, random.Random(SEED + 100 + i), spare[2 * i:2 * i + 2]) for i, c in enumerate(companies)}

    fit_plain = [t for rel in fit.relations.values() for t in rel.plain] + list(fit.generic[0])
    fit_chat = [t for rel in fit.relations.values() for t in rel.chat] + list(fit.generic[1])
    held_plain = [t for p, _ in held.values() for t in p] + list(generic_held["company"][0])
    held_chat = [t for _, c in held.values() for t in c] + list(generic_held["company"][1])

    def items(plain, chat, names):
        return [p for _, p in plain_items(plain, names) + chat_items(tok, chat, names) + qa_items(chat, names)]

    print(f"{n} companies; fitted on {len(fit_plain)} plain and {len(fit_chat)} chat wordings, scored on {len(held_plain)} and {len(held_chat)} held out",
          flush=True)
    cal = calibration_at(model, tok, calibration_paragraphs(), layers)
    reader = _Reader(model, tok, layers)
    fitted = {name: reader.last(items(fit_plain, fit_chat, [name])) for name in companies + [x for c in companies for x in looks[c][0]]}
    print(f"fit readings done | {time.time() - t0:.0f}s", flush=True)

    rows = []
    for ci, company in enumerate(companies):
        pos = fitted[company]
        near = torch.cat([fitted[x] for x in looks[company][0]])
        far = torch.cat([fitted[c] for c in companies if c != company])
        tests = dict(recall=reader.last(items(held_plain, held_chat, [company])),
                     recall_case=reader.last(items(held_plain, held_chat, [company.lower(), company.upper()])),
                     other_industry=reader.last(items(held_plain, held_chat, [looks[company][1][0]])),
                     one_letter=reader.last(items(held_plain, held_chat, [looks[company][1][1]])),
                     other_first_word=reader.last(items(held_plain, held_chat, [looks[company][1][2]])),
                     unseen_company=reader.last(items(held_plain, held_chat, unseen[:2])))
        row = dict(company=company, fitted_against=looks[company][0], tested_on=looks[company][1], layers={})
        for li, layer in enumerate(layers):
            c, cbar = cal[layer]["c"], cal[layer]["cbar"]
            p, ng, fr = pos[:, li], near[:, li], far[:, li]
            w = lda(p, [ng, fr], cal[layer]["cov"])
            r_pos, r_neg = ratio(p, w, c, cbar), ratio(torch.cat([ng, fr]), w, c, cbar)
            tau = float(torch.quantile(r_neg, NEG_QUANTILE))
            med = float(r_pos.median())
            if med <= tau:
                row["layers"][layer] = dict(reading=False, **{k: 0.0 for k in tests})
                continue
            delta = SATURATION * (med - tau)
            row["layers"][layer] = dict(reading=True, **{k: float(((ratio(x[:, li], w, c, cbar) - tau) / delta > 0).float().mean()) for k, x in tests.items()})
        rows.append(row)
        best = max(layers, key=lambda layer: row["layers"][layer]["recall"] - max(row["layers"][layer][k] for k in ("other_industry", "one_letter", "other_first_word")))
        b = row["layers"][best]
        print(f"[{ci + 1}/{n}] {company}: at layer {best} recall {b['recall']:.2f}, look-alikes opened: other industry {b['other_industry']:.2f}, "
              f"one letter {b['one_letter']:.2f}, other first word {b['other_first_word']:.2f} | {time.time() - t0:.0f}s", flush=True)

    summary = []
    print("layer  recall  in other case  other industry  one letter  other first word  unseen company  companies passing")
    for layer in layers:
        col = [r["layers"][layer] for r in rows]
        mean = lambda k: statistics.mean(x[k] for x in col)  # noqa: E731
        look = lambda x: max(x["other_industry"], x["one_letter"], x["other_first_word"])  # noqa: E731
        passing = sum(x["reading"] and x["recall"] >= RECALL_BAR and look(x) <= LOOKALIKE_BAR + 1e-9 for x in col)
        summary.append(dict(layer=layer, recall=mean("recall"), recall_case=mean("recall_case"), other_industry=mean("other_industry"),
                            one_letter=mean("one_letter"), other_first_word=mean("other_first_word"), unseen_company=mean("unseen_company"),
                            companies_passing=passing, with_a_reading=sum(x["reading"] for x in col)))
        s = summary[-1]
        print(f"{layer:5d}  {s['recall']:6.3f}  {s['recall_case']:13.3f}  {s['other_industry']:14.3f}  {s['one_letter']:10.3f}  {s['other_first_word']:16.3f}  "
              f"{s['unseen_company']:14.3f}  {passing:2d} of {n}")
    top = max(summary, key=lambda s: (s["companies_passing"], s["recall"]))
    print(f"best layer: {top['layer']}, where {top['companies_passing']} of {n} companies meet the bench's bars (recall at least {RECALL_BAR}, at most "
          f"1 in 12 questions about a look-alike opened)")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(model=os.path.basename(model_dir), companies=n, bars=dict(recall=RECALL_BAR, lookalike=LOOKALIKE_BAR), summary=summary,
                       best=top, rows=rows, seconds=round(time.time() - t0)), fh, ensure_ascii=False, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()

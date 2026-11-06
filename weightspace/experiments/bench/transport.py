"""Transport check (plan.md): does a tag written at the last token of a company's name reach the last prompt token? Forward passes only.

For each source layer Ls, a vector added to that layer's MLP output at the name's last token; for each target layer Lq, the change in
what the MLP reads at the last prompt token. The map is measured one direction at a time on training wordings of facts 1-8, averaged,
and its top direction t is the tag that arrives strongest. On facts 9-40 (test wordings), writing m*t at the name shifts the read at
Lq along the predicted arrival direction d; the shift is compared with the spread of reads along d on look-alike test prompts.

Usage: python -I transport.py <model dir> <out.json>
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import bench_lib as B  # noqa: E402
from weightspace.model_access import Prompt, encode, render_chat, token_index_of  # noqa: E402

SOURCES, TARGETS = [4, 8, 12], [16, 20, 22]
FIT_FACTS, HELD_FACTS = range(0, 8), range(8, 40)
BATCH, REL_EPS, SCALES = 64, 0.05, (0.25, 0.5, 1.0)


def blocks(model):
    inner = model.model
    return inner.layers if hasattr(inner, "layers") else inner.language_model.layers


class Probe:
    """Adds rows of `add` to layer `source`'s MLP output at one position, and records MLP inputs at the last token of every target."""

    def __init__(self, model):
        self.model, self.add, self.pos, self.source, self.seen = model, None, None, None, {}
        self.hooks = [blocks(model)[l].mlp.register_forward_pre_hook(lambda m, a, l=l: self.seen.__setitem__(l, a[0][:, -1].clone())) for l in TARGETS]
        self.hooks += [blocks(model)[l].mlp.register_forward_hook(lambda m, a, out, l=l: self._write(l, out)) for l in SOURCES]
        self.mlp_out = {}

    def _write(self, layer, out):
        if self.pos is not None:
            self.mlp_out[layer] = out[:, self.pos].detach().clone()
        if layer == self.source and self.add is not None:
            out = out.clone()
            out[:, self.pos] += self.add
            return out
        return None

    @torch.no_grad()
    def run(self, ids: list[int], source=None, add=None, pos=None):
        self.source, self.add, self.pos = source, add, pos
        n = 1 if add is None else add.shape[0]
        self.model(torch.tensor([ids] * n, device="cuda"))
        return {l: self.seen[l] for l in TARGETS}


def items(f, tokenizer, side, n_plain, n_chat):
    t = f[side]
    return [Prompt(x.format(s=f["subject"])) for x in t["templates_plain"][:n_plain]] + \
           [render_chat(tokenizer, x.format(s=f["subject"])) for x in t["templates_chat"][:n_chat]]


def main():
    model_dir, out_path = sys.argv[1:3]
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, device_map="cuda").eval()
    facts = B.load_bench()
    probe = Probe(model)
    fit = [(p, f["subject"]) for i in FIT_FACTS for f in [facts[i]] for p in items(f, tokenizer, "train", 2, 2)]
    hidden = model.config.get_text_config().hidden_size
    eye = torch.eye(hidden, device="cuda")
    report = dict(sources=SOURCES, targets=TARGETS, pairs={})
    maps = {}
    cache = out_path + ".maps.pt"  # the measured maps, so a rerun skips the slow part
    if os.path.exists(cache):
        saved = torch.load(cache)
        maps = {ls: {lq: m.cuda() for lq, m in v.items()} for ls, v in saved["maps"].items()}
        report.update(saved["report"])
        print(f"maps loaded from {os.path.basename(cache)}", flush=True)
    for ls in [l for l in SOURCES if l not in maps]:
        sums = {lq: torch.zeros(hidden, hidden, device="cuda") for lq in TARGETS}
        norms, lin_err = [], []
        for k, (p, subject) in enumerate(fit):
            ids, pos = encode(tokenizer, p), token_index_of(tokenizer, p, subject)
            base = probe.run(ids, pos=pos)
            mlp_norm = float(probe.mlp_out[ls].norm())
            norms.append(mlp_norm)
            eps = REL_EPS * mlp_norm
            for start in range(0, hidden, BATCH):
                moved = probe.run(ids, ls, eps * eye[start:start + BATCH], pos)
                for lq in TARGETS:
                    sums[lq][:, start:start + BATCH] += ((moved[lq] - base[lq]) / eps).T
            if k == 0:  # linearity: the same directions nudged twice as hard
                cols = torch.randperm(hidden)[:8].tolist()
                once, twice = probe.run(ids, ls, eps * eye[cols], pos), probe.run(ids, ls, 2 * eps * eye[cols], pos)
                for lq in TARGETS:
                    a, b = (once[lq] - base[lq]) / eps, (twice[lq] - base[lq]) / (2 * eps)
                    lin_err.append(float(((a - b).norm(dim=1) / a.norm(dim=1).clamp_min(1e-9)).median()))
        maps[ls] = {lq: sums[lq] / len(fit) for lq in TARGETS}
        report[f"mlp_out_norm_L{ls}"] = statistics.median(norms)
        report[f"linearity_L{ls}"] = round(max(lin_err), 4)
        print(f"source layer {ls}: map measured on {len(fit)} wordings, typical MLP output norm at the name {statistics.median(norms):.2f}, "
              f"linearity error {max(lin_err):.3f}  ({time.time() - t0:.0f}s)", flush=True)
        torch.save(dict(maps={k: {lq: m.cpu() for lq, m in v.items()} for k, v in maps.items()},
                        report={k: v for k, v in report.items() if k.startswith(("mlp_out_norm", "linearity"))}), cache)
    # Held-out facts: write m*t at the name, measure the shift along d at the last token, against the look-alike spread along d.
    held = []
    for i in HELD_FACTS:
        f = facts[i]
        negs = [Prompt(x.format(s=c)) for c in f["test"]["lookalikes"] for x in f["test"]["templates_plain"][:2]] + \
               [render_chat(tokenizer, x.format(s=c)) for c in f["test"]["lookalikes"] for x in f["test"]["templates_chat"][:2]]
        neg_reads = [probe.run(encode(tokenizer, p)) for p in negs]
        pos = [(encode(tokenizer, p), token_index_of(tokenizer, p, f["subject"])) for p in items(f, tokenizer, "test", 8, 4)]
        held.append(dict(neg={lq: torch.cat([r[lq] for r in neg_reads]) for lq in TARGETS}, pos=pos,
                         base=[probe.run(ids, pos=k) for ids, k in pos]))
    print(f"held-out facts read ({time.time() - t0:.0f}s)", flush=True)
    for ls in SOURCES:
        tag_norm = report[f"mlp_out_norm_L{ls}"]
        svd = {lq: torch.linalg.svd(maps[ls][lq]) for lq in TARGETS}
        tags = torch.stack([s * tag_norm * svd[lq][2][0] for lq in TARGETS for s in SCALES])  # one row per (target layer, size)
        snr = {(lq, s): [] for lq in TARGETS for s in SCALES}
        cosines = {lq: [] for lq in TARGETS}
        for h in held:
            shifts = {(lq, s): [] for lq in TARGETS for s in SCALES}
            for (ids, k), base in zip(h["pos"], h["base"]):
                moved = probe.run(ids, ls, tags, k)
                for a, lq in enumerate(TARGETS):
                    d = svd[lq][0][:, 0]
                    for b, s in enumerate(SCALES):
                        delta = moved[lq][a * len(SCALES) + b] - base[lq][0]
                        shifts[(lq, s)].append(float(delta @ d))
                        if s == 1.0:
                            cosines[lq].append(float(torch.nn.functional.cosine_similarity(delta, d, dim=0).abs()))
            for lq in TARGETS:
                spread = float((h["neg"][lq] @ svd[lq][0][:, 0]).std())
                for s in SCALES:
                    snr[(lq, s)].append(abs(statistics.median(shifts[(lq, s)])) / max(spread, 1e-9))
        for lq in TARGETS:
            Sv = svd[lq][1]
            row = dict(gain=round(float(Sv[0]), 4), next_gain=round(float(Sv[1]), 4), cos_shift_to_d=round(statistics.median(cosines[lq]), 3),
                       median_snr={str(s): round(statistics.median(snr[(lq, s)]), 2) for s in SCALES},
                       facts_snr_6={str(s): sum(v >= 6 for v in snr[(lq, s)]) for s in SCALES})
            report["pairs"][f"{ls}->{lq}"] = row
            print(f"tag at layer {ls:2} -> read at layer {lq}: gain {row['gain']:.3f} (next {row['next_gain']:.3f}), shift along d (median |cos|) "
                  f"{row['cos_shift_to_d']:.2f}, median shift in look-alike spreads at tag size 0.25/0.5/1.0 x MLP output: "
                  f"{row['median_snr']['0.25']}/{row['median_snr']['0.5']}/{row['median_snr']['1.0']}, facts >= 6 at 1.0: "
                  f"{row['facts_snr_6']['1.0']}/{len(held)}  ({time.time() - t0:.0f}s)", flush=True)
    good = [k for k, r in report["pairs"].items() if r["median_snr"]["1.0"] >= 6]
    report["decision"] = dict(pairs_meeting_rule=good, go=bool(good),
                              best=max(report["pairs"], key=lambda k: report["pairs"][k]["median_snr"]["1.0"]))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"decision: {report['decision']}", flush=True)


if __name__ == "__main__":
    main()

"""Step 3 of the grafted model: key 1 in the weights, by name neurons and a carried tag (lab notebook, "Step 3: key 1 in the weights").

A name neuron in layer 4 fires on the subject's exact name and writes the subject's tag, a direction the model's attention carries to the
end of the question. The fact neuron in layer 20 reads it (t): gate k (z_q + M (t - 1)), up t, down the push. Arm D writes the tags with
a hook from the name index; arm E with the name neurons, no hook. Step 2's 64 edits and its inserted-layer reference.

Usage:
  python scripts/slot_names.py --model <folder> --run runs/counterfact/phrasing_editor \\
      --relations research/results/counterfact/phrasing/editor_relations.json --twins research/results/counterfact/full/twins_narrow.json \\
      --data counterfact.json --slots research/results/slots/03_compensate_layer20.json --step2 research/results/slots/05_step2 \\
      --out runs/slots/step3
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_slot_compensate import SPARE, ScreenCheck, compensation, redundancy_order, second_moment  # noqa: E402
from slot_compile import GAIN, PUSH, line, score  # noqa: E402

from cramschool.bench.counterfact import load_counterfact  # noqa: E402
from cramschool.bench.methods import calibrate  # noqa: E402
from cramschool.bench.scoring import make_baseline  # noqa: E402
from cramschool.facts import PREFIXES  # noqa: E402
from cramschool.graft.build import build_context, chat_items, fact_detectors, guarded, kind_parts, plain_items  # noqa: E402
from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_HELD, CALIBRATION_MAX_TOKENS, RIDGE, blocks, lda, mlp_inputs, ratio  # noqa: E402
from cramschool.graft.layer import _SPACES, NameIndex  # noqa: E402
from cramschool.graft.wordings import Wordings, calibration_paragraphs, norm  # noqa: E402
from cramschool.model_access import Prompt, batch_ids, encode, token_index_of  # noqa: E402

NAME_LAYER, FACT_LAYER = 4, 20
K_NAME, CLIP = 40.0, 10.0          # name neurons: silu(40 z) - silu(40 z - 10), the full tag from z = 0.25
M_TAG = 3.0
TAG_DIMS, TAG_SIZE = 32, 2.0
N_TEMPLATES, N_INVENTED, N_REAL, N_CHAT, SEED = 16, 12, 24, 8, 0
N_PARAS_NAME, N_PARAS_ORDINARY, N_ORDINARY = 120, 40, 4000


def log(s):
    print(s, flush=True)


def parse():
    ap = argparse.ArgumentParser()
    for k in ("model", "run", "relations", "twins", "data", "slots", "step2", "out"):
        ap.add_argument(f"--{k}", required=True)
    return ap.parse_args()


class EventIndex(NameIndex):
    """The name index, also saying at which token a name ends (the inserted layer's NameIndex.step, with the match recorded)."""

    def step_event(self, b, tok):
        st, event = self.rows[b], None
        piece = _SPACES.sub(" ", self.texts[tok].lower())
        if st["provisional"]:
            st["provisional"] = False
            if piece[:1].isalnum():
                st["current"] = st["before"]
        buf = st["buf"]
        if buf.endswith(" ") and piece.startswith(" "):
            piece = piece[1:]
        start = len(buf)
        buf += piece
        if piece and self.pattern is not None:
            for m in self.pattern.finditer(buf, max(0, start - self.longest)):
                if m.end() <= start:
                    continue
                if m.end() == len(buf):
                    st["before"], st["current"], st["provisional"] = st["current"], self.lookup[m.group(0)], True
                    event = self.lookup[m.group(0)]
                elif not buf[m.end()].isalnum():
                    st["current"] = event = self.lookup[m.group(0)]
        st["buf"] = buf[-(self.longest + 32):]
        return event


class TagHook:
    """D's tags: added to layer 4's MLP output at the token where an edited subject's name ends, as the name index reads it."""

    def __init__(self, model, tok, subjects, tags):
        self.index = EventIndex(tok, subjects)
        self.tag_of = {self.index.lookup[" ".join(s.lower().split())]: tags[i] for i, s in enumerate(subjects)}
        self.n, self.events = None, []
        self.handles = [model.register_forward_pre_hook(self._top, with_kwargs=True),
                        blocks(model)[NAME_LAYER].mlp.register_forward_hook(self._add)]

    def _top(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        n, t = ids.shape
        if t > 1 or self.n != n:
            self.index.reset(n)
            self.n = n
        valid = (mask[:, -t:].bool() if mask is not None else torch.ones_like(ids, dtype=torch.bool)).tolist()
        idl, self.events = ids.tolist(), []
        for b in range(n):
            for i in range(t):
                if valid[b][i]:
                    e = self.index.step_event(b, idl[b][i])
                    if e is not None:
                        self.events.append((b, i, e))

    def _add(self, module, args, output):
        if not self.events:
            return None
        out = output.clone()
        for b, i, e in self.events:
            out[b, i] += self.tag_of[e].to(out.dtype)
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def last_reads(model, tok, prompts, layer):
    """Layer `layer`'s MLP input at each prompt's last token (batches of 32, right padding)."""
    out = []
    for start in range(0, len(prompts), 256):
        out += [x[-1] for x in mlp_inputs(model, tok, [encode(tok, p) for p in prompts[start:start + 256]], layer)]
    return torch.stack(out).float()


@torch.no_grad()
def all_reads(model, tok, seqs, layer):
    return mlp_inputs(model, tok, seqs, layer)


@torch.no_grad()
def transport(model, tok, items, size, dev):
    """For each (ids, name position): the change in layer 20's MLP input at the last token per basis direction added (at `size`) to layer
    4's MLP output at the name (out x in)."""
    hidden = model.config.get_text_config().hidden_size
    state = {}

    def add4(m, a, out):
        if state.get("add") is None:
            return None
        out = out.clone()
        out[:, state["pos"]] += state["add"].to(out.dtype)
        return out

    def read20(m, a):
        state["read"] = a[0][:, -1].float()

    h1 = blocks(model)[NAME_LAYER].mlp.register_forward_hook(add4)
    h2 = blocks(model)[FACT_LAYER].mlp.register_forward_pre_hook(read20)
    maps = []
    try:
        for ids, pos in items:
            adds = torch.cat([torch.zeros(1, hidden, device=dev), torch.eye(hidden, device=dev) * size])
            reads = []
            for start in range(0, adds.shape[0], 256):
                chunk = adds[start:start + 256]
                state["add"], state["pos"] = chunk, pos
                model(input_ids=torch.tensor([ids] * chunk.shape[0], device=dev), use_cache=False)
                reads.append(state["read"])
            r = torch.cat(reads)
            maps.append(((r[1:] - r[0]) / size).T)
    finally:
        h1.remove()
        h2.remove()
    return maps


def name_positions(tok, prompts, subject):
    out = []
    for p in prompts:
        try:
            out.append(token_index_of(tok, p, subject))
        except ValueError:
            out.append(None)
    return out


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(a.step2, "facts.json"), encoding="utf-8") as fh:
        facts = json.load(fh)
    with open(os.path.join(a.step2, "summary.json"), encoding="utf-8") as fh:
        step2 = json.load(fh)
    s_a = step2["A"]
    with open(os.path.join(a.step2, "addon_edits.jsonl"), encoding="utf-8") as fh:
        before = {r["id"]: r["replies_before"] for r in (json.loads(line_) for line_ in fh if line_.strip())}
    parts = {f["id"]: torch.load(os.path.join(a.run, "parts", f"{f['id']}.pt"), weights_only=True) for f in facts}
    wordings = Wordings.builtin().with_file(a.relations)
    with open(a.relations, encoding="utf-8") as fh:
        cf_ids = {v: k for k, v in json.load(fh)["counterfact"].items()}
    with open(a.twins, encoding="utf-8") as fh:
        twins = json.load(fh)["twins"]
    with open(a.slots, encoding="utf-8") as fh:
        step1c = json.load(fh)
    slots20 = step2["slots"]
    subjects = [f["subject"] for f in facts]
    log(f"{len(facts)} edits (step 2's); A, the inserted layer: ES {s_a['es']:.3f} PS {s_a['ps']:.3f} NS {s_a['ns']:.3f}")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    dev = model.device
    baseline = make_baseline(model, tok)
    cf_check = ScreenCheck(model, tok, items=[(r["prompt"], r["answers"]) for r in step1c["cf_known"]])
    m20, m4 = calibrate(model, tok, FACT_LAYER), calibrate(model, tok, NAME_LAYER)
    c20, cbar20, cov20 = m20["c"].to(dev).float(), m20["cbar"], m20["cov"].to(dev).float()
    c4, cbar4, cov4 = m4["c"].to(dev).float(), m4["cbar"], m4["cov"].to(dev).float()
    log(f"model, baselines and calibration at layers {NAME_LAYER} and {FACT_LAYER} | {time.time() - t0:.0f}s")

    # ---- prompt sets (training wordings only)
    guarded_w = Wordings(relations=wordings.relations, generic=wordings.generic, kinds=wordings.kinds,
                         guard=set(wordings.guard) | {norm(p) for f in facts for p in f["cf"]["neighbours"] + f["cf"].get("twin_prompts", [])})
    ctx = build_context(facts, guarded_w, name="counterfact", real_world="other_relations", context=4, twins=twins)
    templates = {}
    for rel in sorted({f["relation"] for f in facts}):
        fs = [f for f in facts if f["relation"] == rel]
        templates[rel] = list(dict.fromkeys(t for f in fs for t in f["train"]["templates_plain"])) + list(ctx["drafted"][rel][0][:N_TEMPLATES])
    plain_of = {f["id"]: guarded(plain_items(templates[f["relation"]], [f["subject"]], PREFIXES + ctx.get("context", ())), ctx["guard"]) for f in facts}
    same_of = {f["id"]: [Prompt(p) for p in f["train"]["same_plain"]] for f in facts}
    chat_of = {f["id"]: guarded(chat_items(tok, ctx["drafted"][f["relation"]][1][:N_CHAT], [f["subject"]]), ctx["guard"]) for f in facts}
    tests = " | ".join(p.lower() for f in facts for p in [f["cf"]["rewrite"]] + f["cf"]["paraphrases"] + f["cf"]["neighbours"]
                       + f["cf"].get("twin_prompts", []) + f["test"]["same_plain"])
    records = load_counterfact(a.data)
    chosen = {s.lower() for s in subjects}
    others = {}
    for rel in templates:
        inv = kind_parts(ctx, rel)[0][:N_INVENTED]
        pool = sorted({r["requested_rewrite"]["subject"] for r in records if r["requested_rewrite"]["relation_id"] == cf_ids[rel]})
        pool = [s for s in pool if s.lower() not in tests and s.lower() not in chosen]
        real = random.Random(SEED).sample(pool, min(N_REAL, len(pool)))
        others[rel] = guarded(plain_items(templates[rel], inv + real), ctx["guard"])
    paras = calibration_paragraphs()
    ord_seqs = [tok(paras[i], truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids
                for i in list(CALIBRATION_FIT)[:N_PARAS_ORDINARY]]
    log(f"prompts: {sum(map(len, plain_of.values()))} plain about the subjects, {sum(map(len, others.values()))} about other subjects")

    # ---- the tag unit and the transport map, on one edit's rewrite prompt per relation
    first = {}
    for f in facts:
        first.setdefault(f["relation"], f)
    items = []
    for f in first.values():
        p = Prompt(f["cf"]["rewrite"])
        items.append((encode(tok, p), token_index_of(tok, p, f["subject"])))
    unit_norms = []
    seen = {}
    hn = blocks(model)[NAME_LAYER].mlp.register_forward_hook(lambda m, a_, out: seen.__setitem__("o", out[0].float()))
    try:
        with torch.no_grad():
            for ids, pos in items:
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
                unit_norms.append(float(seen["o"][pos].norm()))
    finally:
        hn.remove()
    unit = float(torch.tensor(unit_norms).median())
    size = TAG_SIZE * unit
    maps = transport(model, tok, items, size, dev)
    U, S, Vh = torch.linalg.svd(torch.cat(maps))
    V = Vh[:TAG_DIMS].T                                       # hidden x 32: the input directions carried most strongly
    g = torch.Generator().manual_seed(SEED)
    q = torch.randn(len(facts), TAG_DIMS, generator=g)
    q = (q / q.norm(dim=1, keepdim=True)).to(dev)
    tags = (q @ V.T) * size                                   # one per subject, at the tag size
    gain = torch.stack([m @ (tags[0] / size) for m in maps]).norm(dim=1)
    log(f"tag unit {unit:.2f} (median MLP output norm at the names, layer {NAME_LAYER}); singular values of the stacked map: top "
        f"{float(S[0]):.2f}, 8th {float(S[7]):.2f}, 32nd {float(S[31]):.2f}, 128th {float(S[127]):.2f}; a tag's arrival per unit size: median "
        f"{float(gain.median()):.3f} | {time.time() - t0:.0f}s")

    # ---- layer 20's slots, as in step 2: freed and compensated
    mlp20, mlp4 = blocks(model)[FACT_LAYER].mlp, blocks(model)[NAME_LAYER].mlp
    G20, U20, D20 = mlp20.gate_proj.weight.data.clone(), mlp20.up_proj.weight.data.clone(), mlp20.down_proj.weight.data.clone()
    G4, U4, D4 = mlp4.gate_proj.weight.data.clone(), mlp4.up_proj.weight.data.clone(), mlp4.down_proj.weight.data.clone()
    fit_texts, spare_texts = [paras[i] for i in CALIBRATION_FIT], [paras[i] for i in SPARE]
    Hf, _, _ = second_moment(model, tok, mlp20.down_proj, fit_texts)
    Hv, _, _ = second_moment(model, tok, mlp20.down_proj, spare_texts)
    K20, Dc20, ridge20, left20, _ = compensation(Hf, Hv, D20.double(), slots20)
    del Hf, Hv
    freed20 = D20.double()
    freed20[:, slots20] = 0
    freed20[:, K20] += Dc20
    log(f"layer {FACT_LAYER}: 64 slots freed, compensation ridge {ridge20:g}, lost output energy left {left20:.1%} | {time.time() - t0:.0f}s")

    det = {f["id"]: next(d for d in fact_detectors(parts[f["id"]]) if d["form"] == "plain") for f in facts}
    silu_k = float(F.silu(torch.tensor(GAIN)))

    def readouts(tagged_reads, untagged_reads, other_reads, ordinary_reads):
        """Each edit's tag reading at layer 20, as a row t(x): 1 at its tagged prompts' median, 0 at the negatives' 99th percentile."""
        rows, stats = {}, {}
        for j, f in enumerate(facts):
            pos = tagged_reads[f["id"]]
            mates = torch.cat([tagged_reads[g_["id"]][:8] for g_ in facts if g_["id"] != f["id"]])
            negs = [untagged_reads[f["id"]], other_reads[f["relation"]], mates, ordinary_reads]
            w = lda(pos, negs, cov20)
            rho_pos, rho_neg = ratio(pos, w, c20, cbar20), ratio(torch.cat(negs), w, c20, cbar20)
            b1, b0 = float(rho_pos.median()), float(torch.quantile(rho_neg, 0.99))
            rows[f["id"]] = (w.double() - b0 * c20.double() / cbar20) / (b1 - b0)
            t_untag = (ratio(untagged_reads[f["id"]], w, c20, cbar20) - b0) / (b1 - b0)
            t_mates = (ratio(mates, w, c20, cbar20) - b0) / (b1 - b0)
            stats[f["id"]] = dict(pos_q05=round(float(torch.quantile((rho_pos - b0) / (b1 - b0), 0.05)), 3),
                                  untagged_max=round(float(t_untag.max()), 3), mates_max=round(float(t_mates.max()), 3))
        return rows, stats

    def place_facts(rows_t):
        gate, up, down = G20.clone(), U20.clone(), freed20.clone()
        for j, f in enumerate(facts):
            d, tr = det[f["id"]], rows_t[f["id"]]
            zq = (d["w"].to(dev).double() - (d["tau"] / cbar20) * c20.double()) / d["delta"]
            gate[slots20[j]] = (GAIN * (zq + M_TAG * (tr - c20.double() / cbar20))).to(G20.dtype)
            up[slots20[j]] = tr.to(U20.dtype)
            sp = parts[f["id"]]["spellings"]["spaced"] or parts[f["id"]]["spellings"]["bare"]
            down[:, slots20[j]] = PUSH * sp["vecs"][0].to(dev).double() / silu_k
        mlp20.gate_proj.weight.data.copy_(gate)
        mlp20.up_proj.weight.data.copy_(up)
        mlp20.down_proj.weight.data.copy_(down.to(D20.dtype))

    def restore():
        for mod, w in ((mlp20.gate_proj, G20), (mlp20.up_proj, U20), (mlp20.down_proj, D20), (mlp4.gate_proj, G4), (mlp4.up_proj, U4),
                       (mlp4.down_proj, D4)):
            mod.weight.data.copy_(w)

    def gather_reads():
        tagged = {f["id"]: last_reads(model, tok, plain_of[f["id"]], FACT_LAYER) for f in facts}
        other = {rel: last_reads(model, tok, ps, FACT_LAYER) for rel, ps in others.items()}
        ordinary = torch.cat(all_reads(model, tok, ord_seqs, FACT_LAYER)).float()
        ordinary = ordinary[torch.randperm(ordinary.shape[0], generator=torch.Generator().manual_seed(SEED))[:N_ORDINARY].to(ordinary.device)]
        return tagged, other, ordinary

    untagged = {f["id"]: last_reads(model, tok, plain_of[f["id"]], FACT_LAYER) for f in facts}  # the unedited model: no tag anywhere
    out = dict(edits=[f["id"] for f in facts], tag=dict(unit=unit, size=size, dims=TAG_DIMS, singular=[round(float(x), 3) for x in S[:64]]),
               A=s_a, unedited=step2["unedited"])
    try:
        # ================= D: tags from the name index
        log("D: tags written by a hook from the name index")
        hook = TagHook(model, tok, subjects, tags)
        try:
            tagged, other, ordinary = gather_reads()
            rows_t, stats_d = readouts(tagged, untagged, other, ordinary)
            log("  tag readings (D): tagged prompts' 5th percentile, median over edits "
                f"{torch.tensor([s['pos_q05'] for s in stats_d.values()]).median():.2f}; the same prompts untagged, largest "
                f"{max(s['untagged_max'] for s in stats_d.values()):.2f}; other edits' tagged prompts, largest "
                f"{max(s['mates_max'] for s in stats_d.values()):.2f} | {time.time() - t0:.0f}s")
            place_facts(rows_t)
            rows_d, s_d = score(model, tok, facts, before, baseline, cf_check)
        finally:
            hook.remove()
            restore()
        log(line("D, tags from the name index", s_d) + f" | {time.time() - t0:.0f}s")

        # ================= E: name neurons in layer 4
        log("E: name neurons in layer 4")
        H4f, _, _ = second_moment(model, tok, mlp4.down_proj, fit_texts)
        H4v, _, _ = second_moment(model, tok, mlp4.down_proj, spare_texts)
        order4 = redundancy_order(H4f, D4.double(), 2 * len(facts))
        slots4 = order4[:2 * len(facts)]
        K4, Dc4, ridge4, left4, _ = compensation(H4f, H4v, D4.double(), slots4)
        del H4f, H4v
        freed4 = D4.double()
        freed4[:, slots4] = 0
        freed4[:, K4] += Dc4
        log(f"layer {NAME_LAYER}: {len(slots4)} slots freed, compensation ridge {ridge4:g}, lost output energy left {left4:.1%}")

        # the freed slots alone, both layers
        mlp4.down_proj.weight.data.copy_(freed4.to(D4.dtype))
        mlp20.down_proj.weight.data.copy_(freed20.to(D20.dtype))
        lost = sum(b and not a_ for b, a_ in zip(baseline.known_check.before, baseline.known_check.correct()))
        lost_cf = sum(b and not a_ for b, a_ in zip(cf_check.before, cf_check.correct()))
        out["freed_both"] = dict(known_lost=lost, cf_lost=lost_cf, drift=baseline.drift()[0])
        log(f"  freed slots alone, layers {NAME_LAYER} and {FACT_LAYER}: known lost {lost} of 133, {lost_cf} of the clean 391, drift "
            f"{out['freed_both']['drift']:.5f}")
        restore()

        # the name detectors: LDA at layer 4's MLP input, the name's last token against every other token
        prompt_sets = {f["id"]: plain_of[f["id"]] + same_of[f["id"]] + chat_of[f["id"]] for f in facts}
        seqs, owner, pos_mask = [], [], []
        for j, f in enumerate(facts):
            ps = prompt_sets[f["id"]]
            for p, at in zip(ps, name_positions(tok, ps, f["subject"])):
                seqs.append(encode(tok, p))
                owner.append(j)
                pos_mask.append(at)
        reads4 = all_reads(model, tok, seqs, NAME_LAYER)
        ord4 = all_reads(model, tok, [tok(paras[i], truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids
                                      for i in list(CALIBRATION_FIT)[:N_PARAS_NAME]], NAME_LAYER)
        X = torch.cat(reads4 + ord4).float()
        label = torch.full((X.shape[0],), -1, dtype=torch.long)
        k = 0
        for r, j, at in zip(reads4, owner, pos_mask):
            if at is not None:
                label[k + at] = j
            k += r.shape[0]
        label = label.to(dev)
        total_sum, total_m2, n_all = X.double().sum(0), X.double().T @ X.double(), X.shape[0]
        rows_gate, name_stats = [], []
        for j, f in enumerate(facts):
            P = X[label == j].double()
            n_neg = n_all - P.shape[0]
            m_pos, m_neg = P.mean(0), (total_sum - P.sum(0)) / n_neg
            sc_pos = (P - m_pos).T @ (P - m_pos)
            sc_neg = (total_m2 - P.T @ P) - n_neg * torch.outer(m_neg, m_neg)
            Sm = cov4.double() + (sc_pos + sc_neg) / n_all
            w = torch.linalg.solve(Sm + RIDGE * Sm.diagonal().mean() * torch.eye(Sm.shape[0], device=dev, dtype=Sm.dtype), m_pos - m_neg).float()
            rho = ratio(X, w, c4, cbar4)
            tau = float(rho[label != j].max())
            med = float(rho[label == j].median())
            delta = 0.5 * (med - tau)
            ok = delta > 0
            frac = float((rho[label == j] > tau).float().mean())
            name_stats.append(dict(subject=f["subject"], separable=ok, positives_above_threshold=round(frac, 3), positives=int(P.shape[0])))
            rows_gate.append((w.double(), tau, delta if ok else None))
        n_ok = sum(s["separable"] for s in name_stats)
        log(f"  name detectors: separable for {n_ok} of {len(facts)} subjects; positives above the largest negative: median "
            f"{torch.tensor([s['positives_above_threshold'] for s in name_stats]).median():.2f} | {time.time() - t0:.0f}s")
        del X, reads4, ord4

        gate4, up4, down4 = G4.clone(), U4.clone(), freed4.clone()
        for j, (w, tau, delta) in enumerate(rows_gate):
            a1, a2 = slots4[2 * j], slots4[2 * j + 1]
            if delta is None:  # not separable: the pair stays silent
                down4[:, a1] = 0
                down4[:, a2] = 0
                continue
            z = (w - (tau / cbar4) * c4.double()) / delta
            gate4[a1] = (K_NAME * z).to(G4.dtype)
            gate4[a2] = (K_NAME * z - CLIP * c4.double() / cbar4).to(G4.dtype)
            up4[a1] = up4[a2] = (c4.double() / cbar4).to(U4.dtype)
            down4[:, a1] = tags[j].double() / CLIP
            down4[:, a2] = -tags[j].double() / CLIP
        mlp4.gate_proj.weight.data.copy_(gate4)
        mlp4.up_proj.weight.data.copy_(up4)
        mlp4.down_proj.weight.data.copy_(down4.to(D4.dtype))

        # how the name neurons fire: at the scored prompts and on held-out ordinary text
        seen = {}
        hf = mlp4.down_proj.register_forward_pre_hook(lambda m, a_: seen.__setitem__("a", a_[0].float()))
        pairs = torch.tensor(slots4, device=dev).view(-1, 2)
        try:
            def fired(seqs_):
                outs = []
                with torch.no_grad():
                    for start in range(0, len(seqs_), 32):
                        chunk = seqs_[start:start + 32]
                        ids, mask = batch_ids(tok, chunk, dev)
                        model(input_ids=ids, attention_mask=mask, use_cache=False)
                        clip = seen["a"][..., pairs[:, 0]] - seen["a"][..., pairs[:, 1]]  # batch x positions x subjects, 10 at full
                        for i, s_ in enumerate(chunk):
                            outs.append(clip[i, :len(s_)] / CLIP)
                return outs
            own_hits, neigh_hits = [], []
            for j, f in enumerate(facts):
                mine = fired([encode(tok, Prompt(p)) for p in [f["cf"]["rewrite"]] + f["cf"]["paraphrases"]])
                own_hits += [float(m[:, j].max()) > 0.5 for m in mine]
                nb = fired([encode(tok, Prompt(p)) for p in f["cf"]["neighbours"]])
                neigh_hits += [float(m.max()) > 0.5 for m in nb]
            held = fired([tok(paras[i], truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids
                          for i in CALIBRATION_HELD])
            held_tokens = sum(m.shape[0] for m in held)
            held_fires = sum(int((m.max(dim=1).values > 0.5).sum()) for m in held)
        finally:
            hf.remove()
        out["name_neurons"] = dict(separable=n_ok, own_name_fires=round(sum(own_hits) / len(own_hits), 4),
                                   neighbour_prompts_any_fire=round(sum(neigh_hits) / len(neigh_hits), 4), held_tokens=held_tokens,
                                   held_tokens_fired=held_fires, per_subject=name_stats)
        log(f"  name neurons fire on the subject's name in its rewrite and reworded prompts: {sum(own_hits)} of {len(own_hits)}; on any "
            f"neighbour prompt: {sum(neigh_hits)} of {len(neigh_hits)}; on held-out text: {held_fires} of {held_tokens} tokens | "
            f"{time.time() - t0:.0f}s")

        # E's untagged reads: layer 4 freed and compensated, the name neurons silent
        mlp4.down_proj.weight.data.copy_(freed4.to(D4.dtype))
        untagged_e = {f["id"]: last_reads(model, tok, plain_of[f["id"]], FACT_LAYER) for f in facts}
        mlp4.down_proj.weight.data.copy_(down4.to(D4.dtype))
        tagged, other, ordinary = gather_reads()
        rows_t, stats_e = readouts(tagged, untagged_e, other, ordinary)
        log("  tag readings (E): tagged prompts' 5th percentile, median over edits "
            f"{torch.tensor([s['pos_q05'] for s in stats_e.values()]).median():.2f}; the same prompts untagged, largest "
            f"{max(s['untagged_max'] for s in stats_e.values()):.2f}; other edits' tagged prompts, largest "
            f"{max(s['mates_max'] for s in stats_e.values()):.2f} | {time.time() - t0:.0f}s")
        place_facts(rows_t)
        log("E: scoring the grafted model (no hook)")
        rows_e, s_e = score(model, tok, facts, before, baseline, cf_check)
        log(line("E, grafted with name neurons", s_e) + f" | {time.time() - t0:.0f}s")
        if a.out:
            torch.save(dict(slots4=slots4, slots20=slots20, gate4=gate4.cpu(), up4=up4.cpu(), down4=down4.to(D4.dtype).cpu(),
                            gate20=mlp20.gate_proj.weight.data.cpu(), up20=mlp20.up_proj.weight.data.cpu(),
                            down20=mlp20.down_proj.weight.data.cpu()), os.path.join(a.out, "grafted_layers.pt"))
    finally:
        restore()

    def ok_vs_a(s):
        return dict(es=s["es"] >= s_a["es"] - 0.05, ps=s["ps"] >= s_a["ps"] - 0.05, ns=s["ns"] >= s_a["ns"] - 0.01,
                    neigh_leak=s["neigh_leak"] <= 0.05, same_leak=s["same_leak"] <= 0.10, known_lost=s["known_lost"] <= 2,
                    cf_lost=s["cf_lost"] <= 7, drift=s["drift"] <= 0.05)
    vd, ve = ok_vs_a(s_d), ok_vs_a(s_e)
    out.update(D=s_d, E=s_e, verdict_D=vd, verdict_E=ve, D_passes=all(vd.values()), E_grafted=all(ve.values()), tag_readings_D=stats_d,
               tag_readings_E=stats_e, slots4=slots4)
    log(f"D (tags from the name index) meets the bar: {all(vd.values())} {vd}")
    log(f"E is a grafted model as good as the inserted layer: {all(ve.values())} {ve}")
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for name, rows in (("D", rows_d), ("E", rows_e)):
        with open(os.path.join(a.out, f"edits_{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

"""Step 16's pre-check (lab notebook): does a separate direction tell another job from the chief executive's own questions?

On step 12's split: a direction per form fitted on the 5 taught other jobs (about the build's invented companies) against half of the CEO
relation's own wordings, its threshold at the 99th percentile of the other half, held apart from the fit. It is read at the end of step
12's own prompts, in the groups and batches step 12 generated them in: the 3 untaught jobs on the 10 companies (30 prompts, 26 leaking),
and the 80 never-taught CEO questions (59 answered). A leak is stopped, or an answer lost, where the veto fires and the fact itself was open
at the prompt's end. Readings and closed-form fits only. GPU, a few minutes.

Usage: python scripts/near_veto_precheck.py <model folder> <table folder> <step 12 replies json> <editor's last check jsonl> <out json>
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool import Graft
from cramschool.gates import generate_batch
from cramschool.graft import build as B
from cramschool.graft.core import lda, mlp_of, ratio
from cramschool.model_access import Prompt, render_chat

TAUGHT = dict(plain=["{s}'s head of marketing is", "The head of research at {s} is"],
              chat=["Who is the chief technology officer of {s}?", "Who is the chief financial officer of {s}?", "Who is the spokesperson for {s}?"])
TEST = ["Who is the general counsel of {s}?", "Who is the head of human resources at {s}?", "The chief engineer of {s} is"]
BATCH, NEG_Q, SAT = 32, 0.99, 0.5


def is_chat(q: str) -> bool:
    return q.rstrip().endswith(("?", "."))  # step 12's rule


def end_reads(model, tok, prompts, layer):
    """The MLP input at the end of each prompt, read in the call step 12 made: generate_batch, 32 at a time, left padding."""
    out = []
    for start in range(0, len(prompts), BATCH):
        seen = []
        h = mlp_of(model, layer).register_forward_pre_hook(lambda m, a: seen.append(a[0][:, -1].float()) if not seen else None)
        try:
            generate_batch(model, tok, prompts[start:start + BATCH], 1, batch_size=BATCH)
        finally:
            h.remove()
        out.append(seen[0])
    return torch.cat(out)


def main():
    model_dir, table_dir, replies_path, check_path, out_path = sys.argv[1:6]
    with open(replies_path, encoding="utf-8") as fh:
        before = json.load(fh)["before"]
    with open(check_path, encoding="utf-8") as fh:
        checks = [json.loads(line) for line in fh if line.strip()]
    recall_q = {r["subject"]: r["args"]["new_questions"] for r in checks if r["relation"] == "ceo"}
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    graft = Graft.load(table_dir, model, tok)
    mctx = graft.calibration()
    c, cbar, layer = mctx["c"], mctx["cbar"], graft.layer
    facts = [f for f in graft.facts if f.relation == "ceo"]
    parts = {e["fact"]["subject"]: e["parts"] for e in graft._facts if e["fact"]["relation"] == "ceo"}

    # the veto, per form: the taught jobs against half of the relation's own wordings, thresholded on the other half
    opts = graft.options
    ctx = B.build_context([e["fact"] for e in graft._facts], graft.wordings, name="table", guard=False, real_world=opts.real_world,
                          context=opts.context, twins=opts.twins)
    inv = B.kind_parts(ctx, "ceo")[0]
    names = inv + B.case_names(inv[:B.N_CASE])
    own = graft.wordings.relations["ceo"]
    halves = dict(fit=(own.plain[0::2], own.chat[0::2]), held=(own.plain[1::2], own.chat[1::2]))
    g = lambda items: B.guarded(items, set())  # noqa: E731
    reads = lambda items: B.last_reads(model, tok, g(items), layer)  # noqa: E731
    pos = dict(plain=reads(B.plain_items(TAUGHT["plain"], names) + B.qa_items(TAUGHT["chat"], names)),
               chat=reads(B.chat_items(tok, TAUGHT["chat"], names)))
    veto = {}
    for form in ("plain", "chat"):
        sets = {}
        for h, (pl, ch) in halves.items():
            sets[h] = reads(B.plain_items(pl, names) + B.qa_items(ch, names)) if form == "plain" else reads(B.chat_items(tok, ch, names))
        w = lda(pos[form], [sets["fit"]], mctx["cov"])
        tau = float(torch.quantile(ratio(sets["held"], w, c, cbar), NEG_Q))
        med = float(ratio(pos[form], w, c, cbar).median())
        veto[form] = dict(w=w, tau=tau, delta=SAT * (med - tau)) if med > tau else None
        print(f"veto {form}: {pos[form].shape[0]} positives, {sets['fit'].shape[0]} own to fit, {sets['held'].shape[0]} own held apart | "
              f"{'usable' if veto[form] else 'NO DETECTOR: the positives median is under the threshold'}", flush=True)

    def z_of(x, d):
        return (ratio(x, d["w"].to(x.device), c, cbar) - d["tau"]) / d["delta"] if d is not None else torch.full((x.shape[0],), -float("inf"), device=x.device)

    groups = dict(test=[(f, q.format(s=f.subject)) for f in facts for q in TEST],
                  taught=[(f, q.format(s=f.subject)) for f in facts for q in TAUGHT["plain"] + TAUGHT["chat"]],
                  recall=[(f, q) for f in facts for q in recall_q[f.subject]])
    out = {}
    for name, items in groups.items():
        assert [(f.subject, q) for f, q in items] == [(r["subject"], r["question"]) for r in before[name]], f"{name}: not step 12's prompts"
        prompts = [render_chat(tok, q) if is_chat(q) else Prompt(q) for _, q in items]
        x = end_reads(model, tok, prompts, layer)
        zv = torch.stack([z_of(x, veto[f_]) for f_ in ("plain", "chat")]).max(0).values
        zf = []
        for row, (f, _) in enumerate(items):
            ds = B.fact_detectors(parts[f.subject])
            zf.append(max(float(z_of(x[row:row + 1], d)[0]) for d in ds) if ds else -float("inf"))
        out[name] = [dict(subject=f.subject, question=q, hit=r["hit"], veto=float(v), fact=fz) for (f, q), r, v, fz in zip(items, before[name], zv.tolist(), zf)]

    leak = [r for r in out["test"] if r["hit"]]
    stopped = [r for r in leak if r["veto"] > 0 and r["fact"] > 0]
    late = [r for r in leak if r["fact"] <= 0]
    answered = [r for r in out["recall"] if r["hit"]]
    lost = [r for r in answered if r["veto"] > 0 and r["fact"] > 0]
    taught_leak = [r for r in out["taught"] if r["hit"]]
    summary = dict(leaking=len(leak), veto_fires_on_leaking=sum(r["veto"] > 0 for r in leak), fact_open_at_end_on_leaking=len(leak) - len(late),
                   leaks_starting_in_reply=len(late), leaks_stopped=len(stopped), leaks_left=len(leak) - len(stopped),
                   answered=len(answered), veto_fires_on_answered=sum(r["veto"] > 0 for r in answered), answers_lost=len(lost),
                   answers_kept=len(answered) - len(lost), taught_leaking=len(taught_leak),
                   veto_fires_on_taught_leaking=sum(r["veto"] > 0 for r in taught_leak))
    summary["passes"] = bool(summary["leaks_left"] <= len(leak) / 2 and summary["answers_kept"] >= 53)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(summary=summary, rows=out, veto={f_: (None if d is None else dict(tau=d["tau"], delta=d["delta"])) for f_, d in veto.items()}),
                  fh, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))
    print("kill rule:", "PASSES: the veto goes on to a build" if summary["passes"] else "MET: the veto stops here")


if __name__ == "__main__":
    main()

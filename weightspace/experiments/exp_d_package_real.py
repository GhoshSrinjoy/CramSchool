"""Real-weights test of the package itself, not the prototype. Run with PYTHONPATH pointing at weightspace/src.

  A. Full flow on Qwen3.5-0.8B: add facts through the gates, ask, save a patch, delete, check bit-exact restore.
  B. Does an edit survive bfloat16, the precision most people load models in?
  C. The original model, Phi-3-mini: can it do what the original scripts printed that it did?
  D. Published loop fix through WeightPatch (hash-checked), then facts solved on the repaired model, compared with facts alone,
     thinking on with the published sampling and a realistic token budget.
  E. Big model edits small model: Qwen3-4B drafts the prompts from a plain sentence, the edit goes into Qwen3.5-0.8B.

Usage: python exp_d_package_real.py <repo_edit_dir> <out.json> <harm> <drift>
"""
import gc
import json
import os
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

import proto as P
from weightspace import DraftError, EditSettings, Fact, GateConfig, KnowledgeEditor, LocalLLM, WeightPatch, draft_fact, render_chat, writer_module
from weightspace.model_access import batch_ids, encode

HERE = os.path.dirname(os.path.abspath(__file__))
edit_dir, out_path, HARM, DRIFT = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
QWEN = r"D:\data\models\Qwen__Qwen3.5-0.8B"
PHI = "microsoft/Phi-3-mini-4k-instruct"
BIG = r"D:\data\models\Qwen3-4B-Instruct-2507"
N_FACTS = 8
N_COMBO = 4
THINK_TOKENS = 4096
OPEN_GATES = GateConfig(min_recall=0, max_leak=1, max_known_drop=1, max_kl=1e9, min_retention=0)

PLAN = [("A. Package flow on Qwen3.5-0.8B", N_FACTS + 1, 25), ("B. bfloat16", N_FACTS, 25), ("C. Your original Phi-3-mini", 6, 60),
        ("D. Loop fix + facts, thinking on", 2 * N_COMBO + 2, 45), ("E. Big model drafts, small model edited", 3, 45)]


class Progress:
    def __init__(self):
        self.s = dict(started_at=time.time(), finished_at=None, stage=PLAN[0][0], detail="loading model", done=0, total=sum(p[1] for p in PLAN),
                      work_done=0.0, work_total=float(sum(p[1] * p[2] for p in PLAN)), stages=[dict(name=p[0], done=0, total=p[1], cost=p[2]) for p in PLAN], results=[], error=None)
        self.write()

    def write(self):
        tmp = os.path.join(HERE, "progress.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.s, fh, ensure_ascii=False)
        os.replace(tmp, os.path.join(HERE, "progress.json"))

    def tick(self, i, detail):
        st = self.s["stages"][i]
        st["done"] += 1
        self.s["done"] += 1
        self.s["work_done"] += st["cost"]
        self.s["stage"], self.s["detail"] = st["name"], detail
        self.write()

    def note(self, i, text):
        self.s["stage"], self.s["detail"] = self.s["stages"][i]["name"], text
        self.write()

    def result(self, text):
        self.s["results"].append(text)
        print(text, flush=True)
        self.write()

    def finish(self, error=None):
        self.s["finished_at"], self.s["error"], self.s["detail"] = time.time(), error, "finished" if error is None else "failed"
        self.write()


prog = Progress()
results = {"meta": dict(harm=HARM, drift=DRIFT)}


def save():
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)


def to_fact(raw):
    subject, relation, target = raw
    return Fact.from_relation(subject, target, relation)


def load(path, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(path)
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="cuda").eval(), tok


def clear():
    gc.collect()
    torch.cuda.empty_cache()


class PresencePenalty(LogitsProcessor):
    def __init__(self, prompt_len, penalty):
        self.prompt_len, self.penalty, self.seen = prompt_len, penalty, None

    def __call__(self, input_ids, scores):
        if self.seen is None:
            self.seen = torch.zeros_like(scores, dtype=torch.bool)
        if input_ids.shape[1] > self.prompt_len:
            self.seen.scatter_(1, input_ids[:, -1:], True)
        return scores - self.penalty * self.seen.float()


@torch.no_grad()
def think(model, tok, questions):
    torch.manual_seed(0)
    prompts = [render_chat(tok, q, thinking=True) for q in questions]
    ids, mask = batch_ids(tok, [encode(tok, p) for p in prompts], model.device, side="left")
    plen = ids.shape[1]
    out = model.generate(
        input_ids=ids, attention_mask=mask, max_new_tokens=THINK_TOKENS, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, pad_token_id=tok.pad_token_id,
        logits_processor=LogitsProcessorList([PresencePenalty(plen, 1.5)]),
    )
    return tok.batch_decode(out[:, plen:], skip_special_tokens=True)


def score_thinking(model, tok, facts):
    questions = [f.heldout_questions[0] for f in facts]
    texts = think(model, tok, questions)
    rows = []
    for f, t in zip(facts, texts):
        answer = t.split("</think>", 1)[1] if "</think>" in t else ""
        rows.append(dict(subject=f.subject, closed="</think>" in t, in_answer=P.contains(answer, f.target), answer=answer[:220], thinking_tail=t[:200] if not answer else ""))
    n = len(rows)
    return dict(closed=sum(r["closed"] for r in rows) / n, in_answer=sum(r["in_answer"] for r in rows) / n, rows=rows)


def main():
    facts = [to_fact(f) for f in P.FACTS[:N_FACTS]]
    cov_texts, _ = P.wikitext_samples()
    settings = EditSettings(layer=8, harm_penalty=HARM, drift_penalty=DRIFT)

    # ---------------------------------------------------------------- A
    model, tok = load(QWEN)
    base = writer_module(model, 8).weight.data.clone()
    ed = KnowledgeEditor(model, tok, settings=settings, calibration_texts=cov_texts, progress=False)
    prog.note(0, "asking the base model")
    before = ed.answer("Who owns RX Systems?", max_new_tokens=30).text
    rows = []
    for i, f in enumerate(facts):
        r = ed.add_fact(f)
        rows.append(dict(subject=f.subject, route=r.route, seconds=round(r.seconds, 1), **r.report.to_dict()))
        prog.tick(0, f"fact {i + 1}/{N_FACTS}: {f.subject} -> {r.route} (recall {r.report.recall:.2f}, leak {r.report.leak:.2f}, known {r.report.known:.2f})")
    routes = [x["route"] for x in rows]
    after = ed.answer("Who owns RX Systems?", max_new_tokens=30)
    patch = ed.to_patch()
    results["A_rows"] = rows
    results["A_summary"] = dict(
        into_weights=routes.count("weights"), into_retrieval=routes.count("retrieval"), mean_seconds=sum(x["seconds"] for x in rows) / len(rows),
        answer_before=before, answer_after=after.text, answer_after_source=after.source,
        unrelated=ed.answer("What is the capital of France?", max_new_tokens=20).text,
        patch_kb=patch.size_bytes / 1024, full_matrix_kb=writer_module(model, 8).weight.numel() * 4 / 1024,
        failures=[x["failures"] for x in rows if x["failures"]],
    )
    s = results["A_summary"]
    prog.result(f"package flow: {routes.count('weights')} of {N_FACTS} into weights, {routes.count('retrieval')} to retrieval, {s['mean_seconds']:.0f}s per fact, patch {s['patch_kb']:.0f} KB (full matrix {s['full_matrix_kb']:.0f} KB)")
    prog.result(f"  before: {before[:100]!r}")
    prog.result(f"  after ({after.source}): {after.text[:100]!r}")
    prog.result(f"  unrelated: {s['unrelated'][:70]!r}")
    for fid in list(ed.ledger.entries) + list(ed.store.facts):
        ed.remove_fact(fid)
    s["restore_exact_after_deleting_all"] = bool(torch.equal(writer_module(model, 8).weight.data, base))
    prog.tick(0, "deleted all facts")
    prog.result(f"deleted every fact: weights bit-identical to the original = {s['restore_exact_after_deleting_all']}")
    save()
    del ed
    clear()

    # ---------------------------------------------------------------- B
    model16, tok16 = load(QWEN, torch.bfloat16)
    editor = KnowledgeEditor(model16, tok16, settings=settings, calibration_texts=cov_texts, gates=OPEN_GATES, progress=False)
    rec, leak, known, kl = [], [], [], []
    for i, f in enumerate(facts):
        r = editor.add_fact(f, on_failure="keep")
        rec.append(r.report.recall); leak.append(r.report.leak); known.append(r.report.known); kl.append(r.report.kl)
        editor.remove_fact(r.fact_id)
        prog.tick(1, f"bfloat16: fact {i + 1}/{N_FACTS}, recall {r.report.recall:.2f}")
    results["B_bfloat16_single_facts"] = dict(recall=sum(rec) / N_FACTS, leak=sum(leak) / N_FACTS, known=sum(known) / N_FACTS, drift=sum(kl) / N_FACTS, baseline_known=r.report.known_baseline)
    prog.result(f"bfloat16, one fact at a time: {({k: round(v, 3) for k, v in results['B_bfloat16_single_facts'].items()})}")
    save()
    del model16, editor
    clear()

    # ---------------------------------------------------------------- C
    prog.note(2, "loading Phi-3-mini, your original model")
    try:
        phi, ptok = load(PHI, torch.bfloat16)
        results["C_phi3_layers"] = len(phi.model.layers)
        results["C_phi3_shape"] = list(writer_module(phi, 0).weight.shape)
        results["C_phi3"] = {}
        for layer in (8, 11):
            editor = KnowledgeEditor(phi, ptok, settings=EditSettings(layer=layer, harm_penalty=HARM, drift_penalty=DRIFT), calibration_texts=cov_texts, gates=OPEN_GATES, progress=False)
            rec, leak, known, kl = [], [], [], []
            answer_text, answer_before = None, None
            for i, f in enumerate(facts[:3]):
                if i == 0:
                    answer_before = editor.answer("Who owns RX Systems?", max_new_tokens=30).text
                r = editor.add_fact(f, on_failure="keep")
                rec.append(r.report.recall); leak.append(r.report.leak); known.append(r.report.known); kl.append(r.report.kl)
                if i == 0:
                    answer_text = editor.answer("Who owns RX Systems?", max_new_tokens=30).text
                editor.remove_fact(r.fact_id)
                prog.tick(2, f"Phi-3 layer {layer}, fact {i + 1}/3: recall {r.report.recall:.2f}, {r.seconds:.0f}s")
            row = dict(recall=sum(rec) / 3, leak=sum(leak) / 3, known=sum(known) / 3, drift=sum(kl) / 3, baseline_known=r.report.known_baseline, answer_before=answer_before, answer_after=answer_text)
            results["C_phi3"][layer] = row
            prog.result(f"Phi-3 layer {layer}: recall {row['recall']:.2f} leak {row['leak']:.2f} known {row['known']:.2f} (base {row['baseline_known']:.2f}) drift {row['drift']:.3f}")
            prog.result(f"  answer: {answer_text[:100]!r}")
            del editor
            clear()
        del phi, ptok
    except Exception as error:  # A failure here must not lose the remaining stages.
        results["C_error"] = repr(error)
        prog.result(f"Phi-3 stage failed: {error!r}")
    save()
    clear()

    # ---------------------------------------------------------------- D
    combo = facts[:N_COMBO]
    results["D"] = {}
    ed = KnowledgeEditor(model, tok, settings=settings, calibration_texts=cov_texts, gates=OPEN_GATES, progress=False)
    for i, f in enumerate(combo):
        ed.add_fact(f, on_failure="keep")
        prog.tick(3, f"facts on the original model: {i + 1}/{N_COMBO}")
    results["D"]["facts_only"] = score_thinking(model, tok, combo)
    prog.tick(3, "facts only: thinking test finished")
    prog.result(f"{N_COMBO} facts, no loop fix: finished thinking {results['D']['facts_only']['closed']:.2f}, fact in final answer {results['D']['facts_only']['in_answer']:.2f}")
    save()
    ed.reset()
    del ed

    fix = WeightPatch.from_published_edit(edit_dir, name="p50c loop fix")
    ed = KnowledgeEditor(model, tok, settings=settings, calibration_texts=cov_texts, gates=OPEN_GATES, progress=False)
    ed.apply_patch(fix, strict=True)
    results["D"]["loop_fix_hash_verified"] = True
    prog.result(f"published loop fix applied through the package, hashes verified: {len(fix.updates)} tensors")
    for i, f in enumerate(combo):
        ed.add_fact(f, on_failure="keep")
        prog.tick(3, f"facts on the repaired model: {i + 1}/{N_COMBO}")
    results["D"]["fix_plus_facts"] = score_thinking(model, tok, combo)
    prog.tick(3, "loop fix + facts: thinking test finished")
    prog.result(f"{N_COMBO} facts + loop fix: finished thinking {results['D']['fix_plus_facts']['closed']:.2f}, fact in final answer {results['D']['fix_plus_facts']['in_answer']:.2f}")
    for row in results["D"]["fix_plus_facts"]["rows"]:
        prog.result(f"  {row['subject']}: {row['answer'][:120]!r}")
    save()
    ed.reset()
    for applied in ed.applied_patches:
        applied.revert()
    del ed
    clear()
    results["D"]["model_restored_after_patch"] = bool(torch.equal(writer_module(model, 8).weight.data, base))

    # ---------------------------------------------------------------- E
    prog.note(4, "loading the big model")
    big, big_tok = load(BIG, torch.bfloat16)
    llm = LocalLLM(big, big_tok)
    drafted = []
    for sentence in ["Julian Rodlauer is the owner of RX Systems", "The chief technology officer of Zentrix Labs is Mira Castellane", "Norvane Industries has its head office in Tallinn"]:
        try:
            fact = draft_fact(llm, sentence)
            ed = KnowledgeEditor(model, tok, settings=settings, calibration_texts=cov_texts, gates=OPEN_GATES, progress=False)
            r = ed.add_fact(fact, on_failure="keep")
            drafted.append(dict(sentence=sentence, drafted=True, subject=fact.subject, target=fact.target, prompts=fact.prompts, questions=fact.questions, heldout_prompts=fact.heldout_prompts, leak_prompts=fact.leak_prompts, recall=r.report.recall, leak=r.report.leak))
            ed.reset()
            prog.tick(4, f"drafted and tested: {fact.subject} (recall {r.report.recall:.2f}, leak {r.report.leak:.2f})")
        except DraftError as error:
            drafted.append(dict(sentence=sentence, drafted=False, error=str(error)))
            prog.tick(4, f"draft rejected for: {sentence}")
    results["E_big_model"] = drafted
    prog.result(f"big model drafting: {[(d['drafted'], round(d.get('recall', 0), 2), round(d.get('leak', 0), 2)) for d in drafted]}")
    save()
    prog.finish()


try:
    main()
except Exception:
    trace = traceback.format_exc()
    print(trace, flush=True)
    prog.finish(trace.strip().splitlines()[-1])
    raise
print("DONE", flush=True)

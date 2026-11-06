"""Big model edits small model, using the package end to end. Also records what the big model actually wrote.

Usage: python exp_i_big_model.py <out.json>
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import proto as P
from weightspace import DraftError, EditSettings, GateConfig, KnowledgeEditor, LocalLLM, draft_fact

QWEN = r"D:\data\models\Qwen__Qwen3.5-0.8B"
BIG = r"D:\data\models\Qwen3-4B-Instruct-2507"
OPEN = GateConfig(min_recall=0, max_leak=1, max_known_drop=1, max_kl=1e9, min_retention=0)
SENTENCES = [
    "Julian Rodlauer is the owner of RX Systems",
    "The chief technology officer of Zentrix Labs is Mira Castellane",
    "Norvane Industries has its head office in Tallinn",
]
out_path = sys.argv[1]


def load(path, dtype):
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map="cuda").eval(), AutoTokenizer.from_pretrained(path)


big, big_tok = load(BIG, torch.bfloat16)
small, small_tok = load(QWEN, torch.float32)
llm = LocalLLM(big, big_tok)
cov_texts, _ = P.wikitext_samples()
replies = []


def recording_llm(prompt):
    reply = llm(prompt)
    replies.append(reply)
    return reply


rows = []
for sentence in SENTENCES:
    try:
        fact = draft_fact(recording_llm, sentence)
    except DraftError as error:
        rows.append(dict(sentence=sentence, drafted=False, error=str(error), reply=replies[-1][:900]))
        print("REJECTED", sentence, "->", error, flush=True)
        continue
    editor = KnowledgeEditor(small, small_tok, settings=EditSettings(layer=8), calibration_texts=cov_texts, gates=OPEN, progress=False)
    result = editor.add_fact(fact, on_failure="keep")
    answer = editor.answer(fact.heldout_questions[0] if fact.heldout_questions else fact.questions[0], max_new_tokens=24).text
    rows.append(dict(
        sentence=sentence, drafted=True, subject=fact.subject, target=fact.target, prompts=fact.prompts, questions=fact.questions, heldout_prompts=fact.heldout_prompts,
        heldout_questions=fact.heldout_questions, leak_prompts=fact.leak_prompts, leak_questions=fact.leak_questions, recall=result.report.recall, leak=result.report.leak, known=result.report.known, answer=answer,
    ))
    print(f"OK {sentence}: subject={fact.subject!r} target={fact.target!r} recall {result.report.recall:.2f} leak {result.report.leak:.2f} answer {answer[:80]!r}", flush=True)
    editor.reset()
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(dict(rows=rows, replies=replies), fh, indent=1, ensure_ascii=False)
print("DONE", flush=True)

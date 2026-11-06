"""Experiment A: run the user's ACTUAL classes from test_1/2/3.py on the real Phi-3-mini weights.

One process per script (Phi-3 at float32 is ~15 GB). Usage: python exp_a_user_method.py test_1 out.json

Questions:
  1. Which modules does the hook filter actually attach to, and which have a .weight?
  2. Does embed_knowledge change any parameter? Which ones, and by how much?
  3. Does the model know the fact with no retrieved text in the prompt?
"""
import importlib.util
import json
import os
import sys
import traceback

import torch

PROJECT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prototype")  # the original scripts
MODEL = "microsoft/Phi-3-mini-4k-instruct"
FACT = "Julian Rodlauer is the owner of RX Systems"
QUERY = "Who owns RX Systems?"
CFG = {
    "test_1": ("SemanticKnowledgeSystem", (FACT,)),
    "test_2": ("ImprovedKnowledgeSystem", (QUERY, FACT)),
    "test_3": ("FocusedSemanticSystem", (FACT,)),
}

name, out_path = sys.argv[1], sys.argv[2]
cls, args = CFG[name]


def load_user_module(n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(PROJECT, f"{n}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def p_token(model, tok, prompt, word):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    probs = model(**ids).logits[0, -1].softmax(-1)
    t = tok(word, add_special_tokens=False).input_ids[0]
    return {
        "token": tok.decode(t),
        "p": float(probs[t]),
        "rank": int((probs > probs[t]).sum()) + 1,
        "top3": [tok.decode(i) for i in probs.topk(3).indices.tolist()],
    }


@torch.no_grad()
def greedy(model, tok, prompt, n=30):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=n, do_sample=False, use_cache=False)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)


mod = load_user_module(name)
system = getattr(mod, cls)(MODEL)
model, tok = system.model, system.tokenizer
r = {"script": name}

modules = dict(model.named_modules())
hooked = [n for n in modules if any(x in n.lower() for x in ["mlp", "attention"]) and n.count(".") <= 3]
with_weight = [n for n in hooked if hasattr(modules[n], "weight")]
r["hooked_total"] = len(hooked)
r["hooked_types"] = sorted({type(modules[n]).__name__ for n in hooked})
r["hooked_with_weight_count"] = len(with_weight)
r["hooked_with_weight_types"] = sorted({type(modules[n]).__name__ for n in with_weight})
r["self_attn_hooked"] = any("self_attn" in n for n in hooked)
r["down_proj_hooked"] = any("down_proj" in n for n in hooked)
r["gate_up_proj_hooked"] = any("gate_up_proj" in n for n in hooked)

prompt = "The owner of RX Systems is"
r["before"] = p_token(model, tok, prompt, " Julian")
r["bare_answer_before"] = greedy(model, tok, "<|user|>\n" + QUERY + "<|end|>\n<|assistant|>\n")

# Exact snapshot of every parameter so nothing can slip past the comparison.
before = {n: p.detach().clone() for n, p in model.named_parameters()}
try:
    system.embed_knowledge(*args)
    r["embed_raised"] = None
except Exception as e:
    r["embed_raised"] = "".join(traceback.format_exception_only(type(e), e)).strip()

diffs = {}
for n, p in model.named_parameters():
    if not torch.equal(before[n], p):
        d = (p - before[n]).float()
        diffs[n] = {
            "entries_changed": int((d != 0).sum()),
            "shape": list(p.shape),
            "rel_change": float(d.norm() / before[n].float().norm()),
        }
r["params_total"] = len(before)
r["params_changed"] = len(diffs)
r["param_diffs"] = diffs
del before
torch.cuda.empty_cache()

r["after"] = p_token(model, tok, prompt, " Julian")
r["bare_answer_after"] = greedy(model, tok, "<|user|>\n" + QUERY + "<|end|>\n<|assistant|>\n")

# The system's own answer path, which is what the README demo shows.
if name != "test_2":
    try:
        res = system.test_knowledge(QUERY)
        r["system_response"] = res["response"]
        r["semantic_matches"] = [float(m[1]) for m in res["semantic_matches"]]
    except Exception as e:
        r["test_raised"] = repr(e)
else:
    # test_2 does not inject the stored fact into the prompt, so this is a pure test of the weights.
    res = system.test_knowledge(QUERY)
    r["system_response"] = res["response"]

n_stored = 0
store = getattr(system, "activation_patterns", None) or getattr(system, "activation_maps", {})
for v in store.values():
    n_stored += len(v)
r["activation_records_held_in_memory"] = n_stored

with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(r, fh, indent=1, ensure_ascii=False)
print("wrote", out_path)

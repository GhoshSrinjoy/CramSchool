"""Does the 0.5 similarity threshold in the original scripts select anything?

Uses the same computation as test_1.py / test_3.py: cosine between mean-pooled last hidden states, on real Phi-3-mini weights.
Runs after the GPU run is finished, or on CPU-sized batches alongside it (bfloat16 to keep the memory small).
"""
import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/Phi-3-mini-4k-instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()


@torch.no_grad()
def embed(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    return model(**ids, output_hidden_states=True, use_cache=False).hidden_states[-1].float().mean(dim=1)


stored = embed("Julian Rodlauer is the owner of RX Systems")
queries = {
    "related": ["Who owns RX Systems?", "RX Systems ownership?", "Tell me about Julian Rodlauer."],
    "unrelated": [
        "What is the capital of France?", "How do I bake sourdough bread?", "Write a poem about the ocean.", "What is 2 + 2?",
        "Explain how photosynthesis works.", "Who won the 1998 World Cup?", "def add(a, b): return a + b", "asdf qwer zxcv",
        "Translate 'good morning' into Spanish.", "Why is the sky blue?", "The cat sat on the mat.", "List three primary colors.",
    ],
}
out = {}
for group, items in queries.items():
    out[group] = {q: round(float(F.cosine_similarity(stored, embed(q))), 3) for q in items}
un = list(out["unrelated"].values())
out["summary"] = dict(
    threshold=0.5, unrelated_above_threshold=sum(v > 0.5 for v in un), unrelated_total=len(un), unrelated_min=min(un), unrelated_max=max(un),
    related_min=min(out["related"].values()),
)
print(json.dumps(out, indent=1, ensure_ascii=False))
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)

"""(Lab notebook, "Result of step 1b".) Noise floor of the per-neuron scan: the same readings twice with nothing switched off, and the known facts flipped by repeating
generation; also which known facts are lost by each group, to see whether the same fragile facts flip each time."""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cramschool.bench.scoring import make_baseline
from cramschool.gates import ORDINARY_TEXT
from cramschool.graft.core import blocks
from cramschool.graft.wordings import known_facts
from cramschool.model_access import batch_ids
M = os.environ.get("CRAMSCHOOL_MODEL", "models/Qwen3.5-0.8B")  # the model's local folder; CRAMSCHOOL_MODEL names another
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
o_ids, o_mask = batch_ids(tok, [tok(t, truncation=True, max_length=24).input_ids for t in ORDINARY_TEXT], model.device)
k_ids, k_mask = batch_ids(tok, [tok(p, add_special_tokens=False).input_ids for p, _ in known_facts()], model.device, side="left")
@torch.no_grad()
def readings():
    o = model(input_ids=o_ids, attention_mask=o_mask, use_cache=False).logits.float().log_softmax(-1)[o_mask.bool()]
    k = model(input_ids=k_ids, attention_mask=k_mask, use_cache=False).logits[:, -1].float().log_softmax(-1)
    return o, k
o0, k0 = readings(); first = k0.argmax(-1); lp0 = k0.gather(1, first[:, None])[:, 0]
for i in range(3):
    o, k = readings()
    print(f"repeat {i}: KL {float((o0.exp() * (o0 - o)).sum(-1).mean()):.2e}, largest drop {float((lp0 - k.gather(1, first[:, None])[:, 0]).max()):.4f}")
down = blocks(model)[20].mlp.down_proj
for desc, fn in (("clone hook, nothing zeroed", lambda m, a: (a[0].clone(),)), ):
    h = down.register_forward_pre_hook(fn); o, k = readings(); h.remove()
    print(f"{desc}: KL {float((o0.exp() * (o0 - o)).sum(-1).mean()):.2e}, largest drop {float((lp0 - k.gather(1, first[:, None])[:, 0]).max()):.4f}")
b = make_baseline(model, tok)
again = b.known_check.correct()
print("known facts flipped by repeating generation:", sum(x != y for x, y in zip(b.known_check.before, again)))
r = json.load(open("research/results/slots/02_harm_layer20.json")); order = r["order"]
items = known_facts()
for n in (16, 64):
    keep = torch.ones(down.weight.shape[1], device=model.device, dtype=model.dtype); keep[torch.tensor(order[:n], device=model.device)] = 0
    h = down.register_forward_pre_hook(lambda m, a, keep=keep: (a[0] * keep,)); now = b.known_check.correct(); h.remove()
    print(n, "least harmful: lost", [items[i][0] for i, (x, y) in enumerate(zip(b.known_check.before, now)) if x and not y])

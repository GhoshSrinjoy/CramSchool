"""Verify the review's claim on step 4: the tag directions (least variance of the residual entering layer 15) carry a large mean, so a
bias-free read gives every ordinary token a constant code. Also compares the script's read vectors (w * t) with the correct (1 + w) * t."""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_MAX_TOKENS, blocks
from cramschool.graft.wordings import calibration_paragraphs

M = os.environ.get("CRAMSCHOOL_MODEL", "models/Qwen3.5-0.8B")  # the model's local folder; CRAMSCHOOL_MODEL names another
L, N = 15, 64
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
paras = calibration_paragraphs()
blk = blocks(model)[L]
seen = {}
h1 = blk.register_forward_pre_hook(lambda m, a, k: seen.__setitem__("h", (k.get("hidden_states", a[0] if a else None))[0].float()), with_kwargs=True)
h2 = blk.input_layernorm.register_forward_hook(lambda m, a, o: seen.__setitem__("x", o[0].float()))
H, X = [], []
with torch.no_grad():
    for i in list(CALIBRATION_FIT)[:120]:
        ids = tok(paras[i], return_tensors="pt", truncation=True, max_length=CALIBRATION_MAX_TOKENS, add_special_tokens=False).input_ids.cuda()
        model(input_ids=ids, use_cache=False)
        H.append(seen["h"][1:])
        X.append(seen["x"][1:])
h1.remove()
h2.remove()
H, X = torch.cat(H), torch.cat(X)
_, eh = torch.linalg.eigh(torch.cov(H.T.double()))
t = eh[:, :N].T.float().cuda()                     # step 4's tag directions
w = blk.input_layernorm.weight.float()
proj = H @ t.T                                     # tokens x 64
print(f"residual along the tag directions: |mean| median {float(proj.mean(0).abs().median()):.4f}, std median {float(proj.std(0).median()):.4f}")
rms = H.pow(2).mean(1).sqrt()
for name, rv in (("script's w*t", w * t), ("(1+w)*t", (1 + w) * t)):
    r = rv / rv.norm(dim=1, keepdim=True)
    R = X @ r.T                                    # the 64 readings at each ordinary token (gamma = 1)
    a = 2 * 0.7304                                 # step 4's tag size, added in h along t_s
    signal = (a / rms.median()) * ((1 + w) * t * r).sum(1).abs()   # change in r_s . x from a tag along t_s
    print(f"{name}: background reading norm per token (64 codes) median {float(R.norm(dim=1).median()):.2f}; "
          f"a tag's own reading median {float(signal.median()):.2f}; background mean/std per code median "
          f"{float((R.mean(0).abs() / R.std(0)).median()):.2f}")

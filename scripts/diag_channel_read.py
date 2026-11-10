"""After step 4 (post hoc): are the channel's read directions quiet on ordinary text after layer 15's input norm, as they are before it?"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cramschool.graft.core import CALIBRATION_FIT, CALIBRATION_MAX_TOKENS, blocks
from cramschool.graft.wordings import calibration_paragraphs

M = "D:/data/models/Qwen__Qwen3.5-0.8B"
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
t = eh[:, :N].T.float().cuda()                                   # the tags' directions, as step 4 chose them
w = blk.input_layernorm.weight.float()
r = (w * t)
r = r / r.norm(dim=1, keepdim=True)                              # the channel's read vectors
rand = torch.nn.functional.normalize(torch.randn(N, H.shape[1], device="cuda"), dim=1)
rms = (H.pow(2).mean(1)).sqrt()
print(f"tokens {H.shape[0]}; median rms of the residual entering layer {L}: {float(rms.median()):.2f}")
print(f"std of the residual h along the tag directions: median {float((H @ t.T).std(0).median()):.4f}; along random directions "
      f"{float((H @ rand.T).std(0).median()):.4f}")
print(f"std of the normed input x along the read vectors: median {float((X @ r.T).std(0).median()):.4f}; along random directions "
      f"{float((X @ rand.T).std(0).median()):.4f}")
print(f"input-norm weight on the tag directions (|w . t|): median {float((t.abs() @ w.abs()).median()):.3f}; over all dimensions: mean |w| "
      f"{float(w.abs().mean()):.3f}")
tag_size = 2 * 0.7304
print(f"a tag of size {tag_size:.2f} in h reads, after the norm, about {tag_size / float(rms.median()):.4f} x |w . t| on its read vector")

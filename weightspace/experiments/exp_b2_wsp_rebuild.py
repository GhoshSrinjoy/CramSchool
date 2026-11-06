"""Rebuild the published p50c edit on the local weights and look at what kind of matrix it is. CPU only.

Reads numeric arrays only (allow_pickle=False). Runs nothing from the downloaded repo.
Usage: python -I exp_b2_wsp_rebuild.py <edit_dir> <out_dir>
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODEL = r"D:\data\models\Qwen__Qwen3.5-0.8B"
edit_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
man = json.load(open(os.path.join(edit_dir, "manifest.json")))
z = np.load(os.path.join(edit_dir, "edit.npz"), allow_pickle=False)
st_file = os.path.join(MODEL, "model.safetensors-00001-of-00001.safetensors")


def sha(bits):
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()


report = {}
edited = {}
with safe_open(st_file, framework="pt") as f:
    emb_key = next(k for k in f.keys() if k.endswith("embed_tokens.weight") and "language" in k)
    E = f.get_tensor(emb_key).float()
    for i, t in enumerate(man["tensors"]):
        w = f.get_tensor(t["name"])
        bits = w.view(torch.uint16).numpy().copy()
        base_ok = sha(bits) == t["base_sha256"]
        e = bits.copy().reshape(-1)
        assert t["form"] == "sparse"
        e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        e = e.reshape(bits.shape)
        edited_ok = sha(e) == t["edited_sha256"]
        wn = torch.from_numpy(e.astype(np.uint16).copy()).view(torch.bfloat16)
        edited[t["name"]] = wn.clone()

        W0, W1 = w.float().double(), wn.float().double()
        D = W1 - W0
        cols = (D != 0).any(0).nonzero().squeeze(-1)
        sub = D[:, cols]
        s = torch.linalg.svdvals(sub)
        U, S, Vh = torch.linalg.svd(sub, full_matrices=False)
        u = U[:, 0]
        # Reconstruct: if the edit is W' = W - alpha * u u^T W on these columns, then D[:,c] is parallel to u for every c.
        proj = (u @ sub)  # per-column coefficient
        recon = torch.outer(u, proj)
        err = float((sub - recon).norm() / sub.norm())
        # Compare D[:,c] to u (u^T W[:,c]) to recover alpha.
        wc = W0[:, cols]
        utw = u @ wc
        alpha = float((proj @ utw) / (utw @ utw))
        ablation_err = float((sub + (-alpha) * torch.outer(u, utw) * -1 * -1 - 0 * sub).norm() / sub.norm()) if False else None
        fit = float((sub - alpha * torch.outer(u, utw)).norm() / sub.norm())
        # Which tokens does this direction write towards?
        logits = E.double() @ u
        top = logits.topk(12).indices.tolist()
        bot = (-logits).topk(12).indices.tolist()
        report[t["name"]] = dict(
            base_hash_matches=base_ok,
            edited_hash_matches=edited_ok,
            n_columns=int(len(cols)),
            singular_values_top5=[float(x) for x in s[:5]],
            rank1_energy_share=float(s[0] ** 2 / (s**2).sum()),
            rank1_recon_rel_error=err,
            alpha_fit=alpha,
            ablation_fit_rel_error=fit,
            relative_change=float(D.norm() / W0.norm()),
            top_token_ids=top,
            bottom_token_ids=bot,
        )
        torch.save(u.float(), os.path.join(out_dir, f"u_layer{t['name'].split('.layers.')[1].split('.')[0]}.pt"))

save_file(edited, os.path.join(out_dir, "p50c_edited_tensors.safetensors"))
with open(os.path.join(out_dir, "report.json"), "w") as fh:
    json.dump(report, fh, indent=1)
print(json.dumps(report, indent=1))

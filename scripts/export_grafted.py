"""Write a grafted model: a copy of the base checkpoint in which the edited MLP matrices replace the originals. Same files, same tensor
names, shapes and dtypes, nothing added: only the values of the reused neurons' rows and columns (and the compensated down columns)
differ. (Lab notebook, "The grafted model".)

Usage: python scripts/export_grafted.py <base model folder> <grafted_layers.pt> <out folder>
"""
import glob
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    base, layers_pt, out = sys.argv[1:4]
    g = torch.load(layers_pt, weights_only=True)
    name_layer = g.get("name_layer", 4)
    edits = {}
    for layer, prefix in ((name_layer, "L" if "gateL" in g else "4"), (20, "20")):
        for part in ("gate", "up", "down"):
            edits[f"model.language_model.layers.{layer}.mlp.{part}_proj.weight"] = g[f"{part}{prefix}"]
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(base):
        p = os.path.join(base, f)
        if os.path.isfile(p) and not f.endswith(".safetensors"):
            shutil.copy(p, os.path.join(out, f))
    done = set()
    for path in sorted(glob.glob(os.path.join(base, "*.safetensors"))):
        with safe_open(path, "pt") as fh:
            meta = fh.metadata()
            tensors = {k: fh.get_tensor(k) for k in fh.keys()}
        for k, w in edits.items():
            if k not in tensors:
                continue
            old = tensors[k]
            if old.shape != w.shape or old.dtype != w.dtype:
                raise SystemExit(f"{k}: the edit is {tuple(w.shape)} {w.dtype}, the checkpoint {tuple(old.shape)} {old.dtype}")
            changed = (old != w)
            rows, cols = int(changed.any(1).sum()), int(changed.any(0).sum())
            print(f"{k}: {int(changed.sum())} of {old.numel()} values changed ({rows} rows, {cols} columns touched)")
            tensors[k] = w.contiguous()
            done.add(k)
        target = os.path.join(out, os.path.basename(path))
        save_file(tensors, target, metadata=meta)
        print(f"{os.path.basename(path)}: {os.path.getsize(path)} bytes in the base, {os.path.getsize(target)} in the grafted model")
    missing = set(edits) - done
    if missing:
        raise SystemExit(f"not found in the checkpoint: {sorted(missing)}")
    print(f"grafted model written to {out}")


if __name__ == "__main__":
    main()

"""A saved, checkable, exactly reversible set of weight changes."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .model_access import find_parameter, tensor_fingerprint


class BaseMismatchError(ValueError):
    """The weights differ from the ones this patch was made for."""


@dataclass
class LowRankUpdate:
    """Adds left @ right to a parameter."""

    param: str
    left: torch.Tensor
    right: torch.Tensor
    base_sha256: str | None = None
    edited_sha256: str | None = None

    def new_value(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.float() + self.left.to(weight.device) @ self.right.to(weight.device)


@dataclass
class SparseUpdate:
    """Replaces chosen entries of a parameter, addressed by flat index."""

    param: str
    indices: torch.Tensor
    values: torch.Tensor
    base_sha256: str | None = None
    edited_sha256: str | None = None

    def new_value(self, weight: torch.Tensor) -> torch.Tensor:
        out = weight.float().clone().view(-1)
        out[self.indices.to(weight.device).long()] = self.values.to(weight.device).float()
        return out.view(weight.shape)


@dataclass
class AppliedPatch:
    """Handle returned by `WeightPatch.apply`. Holds exact copies of what was replaced."""

    model: nn.Module
    originals: dict[str, torch.Tensor]
    active: bool = True

    def revert(self) -> None:
        if not self.active:
            return
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, saved in self.originals.items():
                params[name].data.copy_(saved.to(params[name].device))
        self.active = False


@dataclass
class WeightPatch:
    updates: list = field(default_factory=list)
    name: str = "patch"
    note: str = ""

    def apply(self, model: nn.Module, strict: bool = True) -> AppliedPatch:
        """Apply to a model. With `strict`, refuses weights other than the ones the patch was made for, and changes nothing."""
        resolved = []
        for update in self.updates:
            name, param = find_parameter(model, update.param)
            if strict and update.base_sha256 is not None:
                found = tensor_fingerprint(param)
                if found != update.base_sha256:
                    raise BaseMismatchError(f"{name}: expected base {update.base_sha256[:12]}, found {found[:12]}. Nothing was changed.")
            resolved.append((name, param, update))
        originals = {}
        with torch.no_grad():
            for name, param, update in resolved:
                originals.setdefault(name, param.data.clone().cpu())
                param.data.copy_(update.new_value(param.data).to(param.dtype))
        applied = AppliedPatch(model, originals)
        if strict:
            for name, param, update in resolved:
                if update.edited_sha256 is not None and tensor_fingerprint(param) != update.edited_sha256:
                    applied.revert()
                    raise BaseMismatchError(f"{name}: the rebuilt tensor does not match the recorded edit. The patch was undone.")
        return applied

    @property
    def size_bytes(self) -> int:
        total = 0
        for u in self.updates:
            parts = (u.left, u.right) if isinstance(u, LowRankUpdate) else (u.indices, u.values)
            total += sum(p.numel() * p.element_size() for p in parts)
        return total

    def save(self, path: str) -> None:
        """Writes `<path>.safetensors` and `<path>.json`."""
        tensors, records = {}, []
        for i, u in enumerate(self.updates):
            record = dict(param=u.param, base_sha256=u.base_sha256, edited_sha256=u.edited_sha256)
            if isinstance(u, LowRankUpdate):
                record["kind"] = "low_rank"
                tensors[f"{i}.left"], tensors[f"{i}.right"] = u.left.detach().cpu().contiguous(), u.right.detach().cpu().contiguous()
            else:
                record["kind"] = "sparse"
                tensors[f"{i}.indices"], tensors[f"{i}.values"] = u.indices.detach().cpu().contiguous(), u.values.detach().cpu().contiguous()
            records.append(record)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_file(tensors, f"{path}.safetensors")
        with open(f"{path}.json", "w", encoding="utf-8") as fh:
            json.dump(dict(name=self.name, note=self.note, updates=records), fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "WeightPatch":
        with open(f"{path}.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        tensors = load_file(f"{path}.safetensors")
        updates = []
        for i, r in enumerate(meta["updates"]):
            common = dict(param=r["param"], base_sha256=r["base_sha256"], edited_sha256=r["edited_sha256"])
            if r["kind"] == "low_rank":
                updates.append(LowRankUpdate(left=tensors[f"{i}.left"], right=tensors[f"{i}.right"], **common))
            else:
                updates.append(SparseUpdate(indices=tensors[f"{i}.indices"], values=tensors[f"{i}.values"], **common))
        return cls(updates=updates, name=meta["name"], note=meta["note"])

    @classmethod
    def from_published_edit(cls, edit_dir: str, name: str | None = None) -> "WeightPatch":
        """Read an edit in the format of the Sparks of Weight-Space Programming logs: `manifest.json` plus `edit.npz`.

        Only numeric arrays are read. Base and edited hashes are kept, so `apply` verifies the rebuild bit for bit.
        """
        with open(os.path.join(edit_dir, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        arrays = np.load(os.path.join(edit_dir, "edit.npz"), allow_pickle=False)
        updates = []
        for i, t in enumerate(manifest["tensors"]):
            if t["form"] != "sparse":
                raise NotImplementedError(f"{t['name']}: the {t['form']!r} form is not supported yet; only 'sparse' edits are.")
            if t["dtype"] != "bfloat16":
                raise NotImplementedError(f"{t['name']}: expected bfloat16 entries, found {t['dtype']}.")
            bits = torch.from_numpy(arrays[f"t{i}_val"].astype(np.uint16).copy())
            updates.append(
                SparseUpdate(
                    param=t["name"], indices=torch.from_numpy(arrays[f"t{i}_idx"].astype(np.int64)), values=bits.view(torch.bfloat16).float(),
                    base_sha256=t["base_sha256"], edited_sha256=t["edited_sha256"],
                )
            )
        return cls(updates=updates, name=name or os.path.basename(os.path.normpath(os.path.dirname(edit_dir))), note="published edit")

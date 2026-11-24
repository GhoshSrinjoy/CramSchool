"""A history of a model's own weights: `commit` records a checkpoint as a version, and any version is written back as it was.

    from cramschool import weight_history as wh

    graft.export_grafted("my-grafted-model", base_dir=BASE)                             # the facts in the model's own weights
    wh.commit("my-model.history", "my-grafted-model", "forty facts", base_dir=BASE, table="my.graft")   # v1
    ...                                                                                  # the table changes and is exported again
    wh.commit("my-model.history", "my-grafted-model", "a new CEO", base_dir=BASE, table="my.graft")     # v2
    print(wh.diff_text(wh.diff("my-model.history", "v1", "v2")))                        # the tensors and the facts that changed
    wh.checkout("my-model.history", "v1", "my-grafted-model", base_dir=BASE)            # the folder is v1's checkpoint again

A version holds only what differs from the base checkpoint: the changed tensors, each stored once in the checkpoint's own number type,
and the files beside the weights that the base doesn't have (a grafted model's grafted.json and its name switch). The base's own value
of every tensor a version changed is kept too, so a diff, or setting a loaded model to a version, needs no base folder.

    my-model.history/
      versions.jsonl     one line per version; each carries the hash of the line before it
      base.json          the hash of every tensor and side file of the base checkpoint the history was started on
      objects/<hash>     the tensors and files the versions are made of

`commit`, `versions`, `diff`, `checkout`, `revert` and `is_intact` are file work on checkpoint folders: no model is loaded.
`commit_model` records a loaded model's weights as they are (weight edits kept in memory), and `apply` sets a loaded model to a version.
Nothing is trained and nothing is compiled here. Two models' weights are not merged: merge the tables (`Graft.merge`) and export.
History is never rewritten: a revert is a new version made of an earlier version's tensors.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass

import torch

from .graft.history import _feed, _line_hash, _sha

LOG, BASE_FILE, OBJECTS = "versions.jsonl", "base.json", "objects"
START = "start"
BASE = "base"            # the name of the unedited checkpoint, in diff, checkout and apply
SPEC = "grafted.json"    # a grafted model's side file: which neuron holds which fact
SPEC_LOCAL = ("written", "base_model")  # in grafted.json: the time it was written and a path on this computer; neither makes another model


@dataclass
class WeightVersion:
    """One recorded version of a model's weights. tensors: each changed tensor's name in the checkpoint and the hash of its content.
    files: the side files the base doesn't have, by name. facts: how many facts a grafted model holds (None for other weights).
    table: for a grafted model, the version of the table it was compiled from. of: for a revert, the version whose tensors this took.
    export: the arguments the grafted model was exported with, when it was recorded by `Graft.commit_grafted`."""
    id: str
    time: str
    message: str
    facts: int | None
    state: str
    tensors: dict
    files: dict
    table: dict | None
    of: str | None
    previous: str
    hash: str
    export: dict | None = None


# ---------------------------------------------------------------- hashes and the store

def tensor_hash(tensor: torch.Tensor) -> str:
    """The hash of a tensor's number type, shape and bytes."""
    h = hashlib.sha256()
    _feed(h, tensor)
    return h.hexdigest()


def _object(history_dir: str, digest: str) -> str:
    return os.path.join(history_dir, OBJECTS, digest)


def _store_tensor(history_dir: str, digest: str, tensor: torch.Tensor) -> None:
    from safetensors.torch import save_file

    target = _object(history_dir, digest)
    if not os.path.exists(target):
        save_file({"t": tensor.detach().cpu().contiguous()}, target + ".tmp")
        os.replace(target + ".tmp", target)


def _load_tensor(history_dir: str, digest: str) -> torch.Tensor:
    from safetensors.torch import load_file

    file = _object(history_dir, digest)
    if not os.path.exists(file):
        raise ValueError(f"the history in {history_dir} has lost a tensor ({digest[:12]})")
    tensor = load_file(file)["t"]
    if tensor_hash(tensor) != digest:
        raise ValueError(f"the history in {history_dir} holds a changed tensor ({digest[:12]}): its content is not what was recorded")
    return tensor


def _store_bytes(history_dir: str, digest: str, data: bytes) -> None:
    target = _object(history_dir, digest)
    if not os.path.exists(target):
        with open(target + ".tmp", "wb") as fh:
            fh.write(data)
        os.replace(target + ".tmp", target)


def _load_bytes(history_dir: str, digest: str, name: str) -> bytes:
    file = _object(history_dir, digest)
    if not os.path.exists(file):
        raise ValueError(f"the history in {history_dir} has lost {name} ({digest[:12]})")
    with open(file, "rb") as fh:
        data = fh.read()
    if _sha(data) != digest:
        raise ValueError(f"the history in {history_dir} holds a changed {name}: its content is not what was recorded")
    return data


# ---------------------------------------------------------------- checkpoints

def _weight_files(folder: str) -> list[str]:
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.safetensors")))
    if not files:
        raise ValueError(f"{folder} has no .safetensors files: a checkpoint must be a local folder in safetensors format")
    return files


def _side_files(folder: str) -> dict[str, bytes]:
    out = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and not name.endswith((".safetensors", ".tmp")):
            with open(path, "rb") as fh:
                out[name] = fh.read()
    return out


def _index(base_dir: str) -> dict:
    """The base checkpoint as hashes: per weight file its header and each tensor's hash, and each side file's hash."""
    from safetensors import safe_open

    files = {}
    for name in _weight_files(base_dir):
        with safe_open(os.path.join(base_dir, name), "pt") as fh:
            keys, types = {}, {}
            for key in sorted(fh.keys()):
                tensor = fh.get_tensor(key)
                keys[key], types[key] = tensor_hash(tensor), str(tensor.dtype)
            files[name] = dict(metadata=fh.metadata(), keys=keys, types=types)
    side = {name: _sha(data) for name, data in _side_files(base_dir).items()}
    fingerprint = _sha(json.dumps(dict(files={n: f["keys"] for n, f in files.items()}), sort_keys=True).encode("utf-8"))
    return dict(fingerprint=fingerprint, files=files, side=side)


def _base(history_dir: str, base_dir: str | None = None) -> dict:
    """The history's base index; made from `base_dir` when the history is new."""
    file = os.path.join(history_dir, BASE_FILE)
    if os.path.exists(file):
        with open(file, encoding="utf-8") as fh:
            return json.load(fh)
    if base_dir is None:
        raise ValueError(f"{history_dir} holds no history of weights: record a first version with commit")
    os.makedirs(os.path.join(history_dir, OBJECTS), exist_ok=True)
    index = _index(base_dir)
    with open(file, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=1)
    return index


def _base_hashes(base: dict) -> dict[str, str]:
    return {key: digest for entry in base["files"].values() for key, digest in entry["keys"].items()}


def _base_values(base: dict, base_dir: str, keys):
    """(name, tensor) of each of these tensors of the base, read from the base folder and checked against the base index."""
    from safetensors import safe_open

    hashes, wanted = _base_hashes(base), set(keys)
    for name, entry in base["files"].items():
        mine = wanted & set(entry["keys"])
        if not mine:
            continue
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            raise ValueError(f"{base_dir} has no {name}: it is not the base this history was started on")
        with safe_open(path, "pt") as fh:
            for key in sorted(mine):
                tensor = fh.get_tensor(key)
                if tensor_hash(tensor) != hashes[key]:
                    raise ValueError(f"{base_dir} is not the base this history was started on: {key} differs")
                yield key, tensor


def _keep_base_values(history_dir: str, base: dict, base_dir: str, keys) -> None:
    """Keep the base's own value of each of these tensors, so that a diff or an apply needs no base folder."""
    hashes = _base_hashes(base)
    for key, tensor in _base_values(base, base_dir, [key for key in keys if not os.path.exists(_object(history_dir, hashes[key]))]):
        _store_tensor(history_dir, hashes[key], tensor)


# ---------------------------------------------------------------- the log

def _records(history_dir: str) -> list[dict]:
    file = os.path.join(history_dir, LOG)
    if not os.path.exists(file):
        return []
    with open(file, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def versions(history_dir: str) -> list[WeightVersion]:
    """Every recorded version of the weights, oldest first."""
    return [WeightVersion(**r) for r in _records(history_dir)]


def _resolve(history_dir: str, version) -> dict:
    """The record of "v3", 3 or "latest"; for "base", a record with nothing changed."""
    if version == BASE:
        return dict(id=BASE, tensors={}, files={}, table=None, facts=None)
    log = _records(history_dir)
    if not log:
        raise ValueError(f"{history_dir} has no recorded versions: record one with commit")
    if version in (None, "latest"):
        return log[-1]
    name = f"v{version}" if isinstance(version, int) else str(version)
    for r in log:
        if r["id"] == name:
            return r
    raise ValueError(f"{history_dir} has no version {name!r}; it has {log[0]['id']} to {log[-1]['id']}, and {BASE!r}")


def _spec_of(history_dir: str, record: dict) -> dict | None:
    if SPEC not in record["files"]:
        return None
    return json.loads(_load_bytes(history_dir, record["files"][SPEC], SPEC).decode("utf-8"))


def _facts(spec: dict | None) -> list[list[str]]:
    """The facts a grafted model holds, from its grafted.json: [subject, relation, answer], each once."""
    if spec is None:
        return []
    return sorted({(n["subject"], n["relation"], n["answer"]) for n in spec.get("neurons", [])})


def _state(base: dict, tensors: dict, files: dict, spec: dict | None) -> str:
    """What makes two versions the same weights: the base, the changed tensors, the side files, and grafted.json without the time it
    was written and the path of the base on this computer."""
    body = dict(base=base["fingerprint"], tensors=tensors, files={n: h for n, h in files.items() if n != SPEC},
                spec={k: v for k, v in spec.items() if k not in SPEC_LOCAL} if spec is not None else None)
    return _sha(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _table_link(table: str, facts: list, version=None) -> dict:
    """The table version a grafted model was compiled from. Named by the caller (`Graft.commit_grafted` names the one it has just
    recorded and exported), or else the newest recorded one, and then only if the saved table has no changes since.

    A version found as the newest must hold exactly the facts the grafted model holds; a fact built again with the same answer is not
    told apart by that. A version named by the caller must hold every fact of the grafted model and may hold more: a fact whose
    detector could not be built gets no neuron, so the weights can hold fewer facts than the table they were compiled from."""
    from .graft import history as table_history

    found = table_history.versions(table)
    if not found:
        raise ValueError(f"{table} has no recorded versions: record the table first (Graft.commit), then the weights compiled from it")
    if version is None:
        if not table_history.is_committed(table):
            raise ValueError(f"the table saved in {table} has changes that no version holds: record it first (Graft.commit), so that these "
                             "weights can name the version they were compiled from")
        chosen = found[-1]
    else:
        name = f"v{version}" if isinstance(version, int) else str(version)
        chosen = next((v for v in found if v.id == name), None)
        if chosen is None:
            raise ValueError(f"{table} has no version {name!r}; it has {found[0].id} to {found[-1].id}")
    held = sorted((f["subject"], f["relation"], f["answer"]) for f in table_history.facts_of(table, chosen.id))
    mine = [tuple(f) for f in facts]
    if (set(mine) - set(held)) if version is not None else (held != mine):
        raise ValueError(f"version {chosen.id} of {table} holds other facts than these weights ({len(held)} against {len(facts)}, "
                         "or other answers): export the grafted model from the table as it is recorded, or leave out the table")
    return dict(version=chosen.id, state=chosen.state)


def compiled_from(history_dir: str, table_state: str) -> list[WeightVersion]:
    """The versions of the weights that were compiled from a table in this state (`Version.state` of the table's history), oldest
    first. A table that was reverted has a new version id and the state of the version it went back to, so the state is the key."""
    return [v for v in versions(history_dir) if v.table and v.table.get("state") == table_state]


def _append(history_dir: str, base: dict, message: str, tensors: dict, files: dict, spec: dict | None, table: dict | None, of: str | None,
            export: dict | None = None) -> WeightVersion:
    state = _state(base, tensors, files, spec)
    log = _records(history_dir)
    if log and log[-1]["state"] == state:
        return WeightVersion(**log[-1])
    record = dict(id=f"v{len(log) + 1}", time=time.strftime("%Y-%m-%d %H:%M:%S"), message=message, facts=len(_facts(spec)) if spec is not None else None,
                  state=state, tensors=tensors, files=files, table=table, of=of, export=export, previous=log[-1]["hash"] if log else START)
    record["hash"] = _line_hash(record)
    with open(os.path.join(history_dir, LOG), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return WeightVersion(**record)


# ---------------------------------------------------------------- commit

def _read_checkpoint(history_dir: str, base: dict, checkpoint_dir: str, store: bool) -> tuple[dict, dict, dict | None]:
    """(changed tensors, side files the base doesn't have, grafted.json) of a checkpoint folder, as hashes; with `store`, kept."""
    from safetensors import safe_open

    if _weight_files(checkpoint_dir) != sorted(base["files"]):
        raise ValueError(f"{checkpoint_dir} holds other weight files than the base this history was started on")
    tensors = {}
    for name, entry in base["files"].items():
        with safe_open(os.path.join(checkpoint_dir, name), "pt") as fh:
            if sorted(fh.keys()) != sorted(entry["keys"]) or fh.metadata() != entry["metadata"]:
                raise ValueError(f"{name} in {checkpoint_dir} holds other tensors, or another header, than the base's")
            for key in sorted(fh.keys()):
                tensor = fh.get_tensor(key)
                digest = tensor_hash(tensor)
                if digest != entry["keys"][key]:
                    tensors[key] = digest
                    if store:
                        _store_tensor(history_dir, digest, tensor)
    files, spec = {}, None
    for name, data in _side_files(checkpoint_dir).items():
        digest = _sha(data)
        if base["side"].get(name) != digest:
            files[name] = digest
            if store:
                _store_bytes(history_dir, digest, data)
        if name == SPEC:
            spec = json.loads(data.decode("utf-8"))
    return tensors, files, spec


def commit(history_dir: str, checkpoint_dir: str, message: str = "", *, base_dir: str, table: str | None = None, table_version=None,
           export: dict | None = None, of: str | None = None) -> WeightVersion:
    """Record the checkpoint in `checkpoint_dir` as a new version of the weights, and return it. Only the tensors that differ from
    the base checkpoint in `base_dir` are stored, and the side files the base doesn't have. The first commit starts the history on
    that base; a checkpoint with other weight files or tensor names is refused. Weights that are the newest version already record
    nothing.

    table: the folder of the GRAFT table a grafted model was exported from. The table version it was compiled from is written into
    this version, so the two histories can be read together: `table_version` if given, else the newest recorded one, and then only
    if the saved table has no changes since. That version must hold the facts the grafted model holds; otherwise nothing is recorded.
    export: the arguments of the export, kept with the version for the record. `Graft.commit_grafted` passes both."""
    base = _base(history_dir, base_dir)
    link = None
    if table is not None:  # checked before anything is stored
        spec_file = os.path.join(checkpoint_dir, SPEC)
        if not os.path.exists(spec_file):
            raise ValueError(f"{checkpoint_dir} has no {SPEC}: only a grafted model is compiled from a table")
        with open(spec_file, encoding="utf-8") as fh:
            link = _table_link(table, _facts(json.load(fh)), table_version)
    tensors, files, spec = _read_checkpoint(history_dir, base, checkpoint_dir, store=True)
    _keep_base_values(history_dir, base, base_dir, tensors)
    return _append(history_dir, base, message, tensors, files, spec, link, of, export)


def is_committed(history_dir: str, checkpoint_dir: str) -> bool:
    """Whether the checkpoint in a folder is the newest recorded version, with nothing changed since."""
    log = _records(history_dir)
    if not log:
        return False
    base = _base(history_dir)
    return log[-1]["state"] == _state(base, *_read_checkpoint(history_dir, base, checkpoint_dir, store=False))


def is_intact(history_dir: str) -> bool:
    """Whether the history is whole: every line carries the hash of the line before it and its own, and every tensor and file a
    version names is in the store and holds what its name says."""
    previous = START
    try:
        base = _base(history_dir)
        hashes = _base_hashes(base)
        for n, r in enumerate(_records(history_dir), 1):
            if r.get("previous") != previous or r.get("hash") != _line_hash(r) or r.get("id") != f"v{n}":
                return False
            previous = r["hash"]
            for key, digest in r["tensors"].items():
                _load_tensor(history_dir, digest)
                _load_tensor(history_dir, hashes[key])
            for name, digest in r["files"].items():
                _load_bytes(history_dir, digest, name)
            if r["state"] != _state(base, r["tensors"], r["files"], _spec_of(history_dir, r)):
                return False
    except (ValueError, KeyError):
        return False
    return True


# ---------------------------------------------------------------- checkout and revert

def checkout(history_dir: str, version, out_dir: str, *, base_dir: str) -> dict:
    """Write a version as a checkpoint folder: the base's files, with the version's tensors in place of the base's and its side files
    beside them. For a version recorded from a folder, the result is that folder's files again, byte for byte. `version` may be
    "base". The base folder is checked tensor by tensor, and another base is refused before anything is written. A folder that holds
    another version is overwritten, and that version's side files are removed. Returns what was written."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    base, record = _base(history_dir), _resolve(history_dir, version)
    if os.path.abspath(out_dir) == os.path.abspath(base_dir):
        raise ValueError("the base folder itself can't be written over: give another folder")
    new = {key: _load_tensor(history_dir, digest) for key, digest in record["tensors"].items()}
    side = {name: _load_bytes(history_dir, digest, name) for name, digest in record["files"].items()}
    base_side = _side_files(base_dir)
    if {name: _sha(data) for name, data in base_side.items()} != base["side"] or _weight_files(base_dir) != sorted(base["files"]):
        raise ValueError(f"{base_dir} is not the base this history was started on: its files differ")
    for name, entry in base["files"].items():  # the whole base is checked first, one tensor at a time, so a refusal writes nothing
        with safe_open(os.path.join(base_dir, name), "pt") as fh:
            if sorted(fh.keys()) != sorted(entry["keys"]) or fh.metadata() != entry["metadata"]:
                raise ValueError(f"{base_dir} is not the base this history was started on: {name} holds other tensors")
            for key in fh.keys():
                if key not in new and tensor_hash(fh.get_tensor(key)) != entry["keys"][key]:
                    raise ValueError(f"{base_dir} is not the base this history was started on: {key} differs")
    os.makedirs(out_dir, exist_ok=True)
    for name in base["files"]:  # then written one weight file at a time, so a model in several files is never held whole
        with safe_open(os.path.join(base_dir, name), "pt") as fh:
            header = fh.metadata()
            tensors = {key: new[key] if key in new else fh.get_tensor(key) for key in fh.keys()}
        save_file(tensors, os.path.join(out_dir, name + ".tmp"), metadata=header)
        del tensors
        os.replace(os.path.join(out_dir, name + ".tmp"), os.path.join(out_dir, name))
    for name, data in {**base_side, **side}.items():
        with open(os.path.join(out_dir, name), "wb") as fh:
            fh.write(data)
    others = {name for r in _records(history_dir) for name in r["files"]} - set(side) - set(base_side)
    for name in sorted(others):  # another version's side files: a name switch for neurons this version doesn't have
        if os.path.exists(os.path.join(out_dir, name)):
            os.remove(os.path.join(out_dir, name))
    return dict(version=record["id"], tensors=sorted(new), files=sorted(side), folder=out_dir)


def revert(history_dir: str, version, checkpoint_dir: str, *, base_dir: str, message: str | None = None, discard: bool = False) -> WeightVersion:
    """Make a checkpoint folder hold an earlier version again, and record that as a new version (`of` names the earlier one). If the
    folder holds weights that no version holds, they would be lost: that is refused unless `discard` is set. `version` may be "base"."""
    record = _resolve(history_dir, version)
    if not discard and not holds_recorded(history_dir, checkpoint_dir):
        raise ValueError(f"the checkpoint in {checkpoint_dir} is one that no version holds. Record it first (commit), or pass discard to lose it.")
    checkout(history_dir, version, checkpoint_dir, base_dir=base_dir)
    base = _base(history_dir)
    return _append(history_dir, base, message if message is not None else f"back to {record['id']}", record["tensors"], record["files"],
                   _spec_of(history_dir, record), record.get("table"), record["id"], record.get("export"))


def holds_recorded(history_dir: str, checkpoint_dir: str) -> bool:
    """Whether writing over a folder would lose nothing: it holds no checkpoint, or the base's, or one that some version holds."""
    if not os.path.isdir(checkpoint_dir) or not glob.glob(os.path.join(checkpoint_dir, "*.safetensors")):
        return True
    base = _base(history_dir)
    state = _state(base, *_read_checkpoint(history_dir, base, checkpoint_dir, store=False))
    return state in {r["state"] for r in _records(history_dir)} or state == _state(base, {}, {}, None)


# ---------------------------------------------------------------- diff

def diff(history_dir: str, a, b=None) -> dict:
    """What changed in the weights from version `a` to version `b` (the newest by default); either may be "base".

    tensors: each tensor that differs, with how many of its values differ, of how many, the largest difference, and for a matrix how
    many rows and columns hold a difference. facts: for grafted models, the facts added, removed and changed (another answer).
    files: the side files that differ. table: the table version each was compiled from. same: nothing differs."""
    base = _base(history_dir)
    hashes = _base_hashes(base)
    ra, rb = _resolve(history_dir, a), _resolve(history_dir, b)
    tensors = {}
    for key in sorted(set(ra["tensors"]) | set(rb["tensors"])):
        da, db = ra["tensors"].get(key, hashes[key]), rb["tensors"].get(key, hashes[key])
        if da == db:
            continue
        ta, tb = _load_tensor(history_dir, da), _load_tensor(history_dir, db)
        differs = ta != tb
        tensors[key] = dict(values=int(differs.sum()), of=ta.numel(), largest=float((ta.double() - tb.double()).abs().max()),
                            rows=int(differs.any(1).sum()) if ta.dim() == 2 else None, columns=int(differs.any(0).sum()) if ta.dim() == 2 else None)
    fa = {(s, r): ans for s, r, ans in _facts(_spec_of(history_dir, ra))}
    fb = {(s, r): ans for s, r, ans in _facts(_spec_of(history_dir, rb))}
    row = lambda k, ans: dict(subject=k[0], relation=k[1], answer=ans)  # noqa: E731
    facts = dict(added=[row(k, v) for k, v in fb.items() if k not in fa], removed=[row(k, v) for k, v in fa.items() if k not in fb],
                 changed=[dict(subject=k[0], relation=k[1], before=fa[k], after=fb[k]) for k in fa if k in fb and fa[k] != fb[k]])
    files = sorted(n for n in set(ra["files"]) | set(rb["files"]) if n != SPEC and ra["files"].get(n) != rb["files"].get(n))
    out = dict(a=ra["id"], b=rb["id"], tensors=tensors, facts=facts, files=files, table=[ra.get("table"), rb.get("table")])
    out["same"] = not (tensors or files or any(facts.values()))
    return out


def diff_text(d: dict) -> str:
    """A diff of weights as lines of text."""
    lines = [f"{d['a']} -> {d['b']}"]
    for key, t in d["tensors"].items():
        count = lambda n, word: f"{n} {word}{'' if n == 1 else 's'}"  # noqa: E731
        where = f", in {count(t['rows'], 'row')} and {count(t['columns'], 'column')}" if t["rows"] is not None else ""
        lines.append(f"  {key}: {t['values']} of {t['of']} values differ{where}; the largest by {t['largest']:.4g}")
    lines += [f"  + {f['subject']} | {f['relation']} | {f['answer']}" for f in d["facts"]["added"]]
    lines += [f"  - {f['subject']} | {f['relation']} | {f['answer']}" for f in d["facts"]["removed"]]
    lines += [f"  ~ {f['subject']} | {f['relation']} | {f['before']} -> {f['after']}" for f in d["facts"]["changed"]]
    if d["files"]:
        lines.append(f"  side files that differ: {', '.join(d['files'])}")
    tables = [t["version"] if t else None for t in d["table"]]
    if any(tables):
        lines.append(f"  compiled from table version: {tables[0] or 'none'} -> {tables[1] or 'none'}")
    if d["same"]:
        lines.append("  nothing differs")
    return "\n".join(lines)


# ---------------------------------------------------------------- loaded models

def _state_name(state: dict, key: str) -> str | None:
    """The name a checkpoint tensor has in a loaded model, also accepting multimodal checkpoint names for a text-only model."""
    for candidate in (key, key.replace("model.language_model.", "model."), key.replace("model.layers.", "model.language_model.layers.")):
        if candidate in state:
            return candidate
    return None


def commit_model(history_dir: str, model, message: str = "", *, base_dir: str) -> WeightVersion:
    """Record a loaded model's weights as they are now: for weight edits kept in memory. Every tensor of the base checkpoint that the
    model holds is compared with the base, and those that differ are stored, in the checkpoint's own number type. A grafted model is
    recorded from its folder (`commit`), which also keeps its grafted.json and name switch.

    A model is often loaded in another number type than some of its checkpoint's tensors are stored in. Such a tensor counts as
    unchanged when it is the base's value in the model's type, as loading gave it. A changed one is stored in the checkpoint's type
    if that type holds the change exactly, so that `apply` gives the model the same bits again; if it can't, the commit is refused."""
    base = _base(history_dir, base_dir)
    state = model.state_dict()
    tensors, kept, other_type = {}, {}, {}
    for entry in base["files"].values():
        for key, digest in entry["keys"].items():
            name = _state_name(state, key)
            if name is None:
                continue
            tensor = state[name].detach().cpu()
            if str(tensor.dtype) != entry["types"][key]:
                other_type[key] = tensor
                continue
            found = tensor_hash(tensor)
            if found != digest:
                tensors[key], kept[key] = found, tensor
    for key, value in _base_values(base, base_dir, other_type):
        tensor = other_type[key]
        if value.shape == tensor.shape and torch.equal(value.to(tensor.dtype), tensor):
            continue
        stored = tensor.to(value.dtype)
        if value.shape != tensor.shape or not torch.equal(stored.to(tensor.dtype), tensor):
            raise ValueError(f"{key} was changed in the loaded model, which holds it as {tensor.dtype}, and the checkpoint's {value.dtype} can't hold "
                             "the change exactly: load the model in the checkpoint's own number type, so that a version is its exact weights")
        tensors[key], kept[key] = tensor_hash(stored), stored
    _keep_base_values(history_dir, base, base_dir, tensors)
    for key, tensor in kept.items():
        _store_tensor(history_dir, tensors[key], tensor)
    return _append(history_dir, base, message, tensors, {}, None, None, None)


def apply(history_dir: str, version, model, *, tokenizer=None, switch=None) -> dict:
    """Set a loaded model's weights to a version, in place: every tensor that any recorded version changes gets this version's
    value, or the base's. `version` may be "base". The model must hold the base's weights everywhere else. A tensor the model holds
    in another number type than the checkpoint's gets the value in the model's type, as loading that version's checkpoint the same
    way would give it.

    A grafted model's facts act through its name switch, and each version has its own neurons. switch: the switch attached for the
    version the model holds now; it is detached first, since on other weights it would light the wrong neurons. tokenizer: given, the
    new version's switch is attached and returned. Without it a grafted version's facts stay dormant.

    Returns version, tensors (the names set) and switch (the new one, or None)."""
    from .graft.grafted_switch import switch_for

    base, record = _base(history_dir), _resolve(history_dir, version)
    hashes = _base_hashes(base)
    touched = sorted({key for r in _records(history_dir) for key in r["tensors"]})
    state = model.state_dict()
    values = {}
    for key in touched:
        name = _state_name(state, key)
        if name is None:
            raise ValueError(f"the loaded model has no tensor {key}, which a recorded version changes")
        value = _load_tensor(history_dir, record["tensors"].get(key, hashes[key]))
        if value.shape != state[name].shape:
            raise ValueError(f"{key} is {tuple(state[name].shape)} in the loaded model and {tuple(value.shape)} in the history: another model")
        values[name] = value
    if switch is not None:
        switch.detach()
    with torch.no_grad():
        for name, value in values.items():
            state[name].copy_(value.to(state[name].device, state[name].dtype))
    spec = _spec_of(history_dir, record)
    new_switch = switch_for(model, tokenizer, spec) if spec is not None and tokenizer is not None else None
    return dict(version=record["id"], tensors=sorted(values), switch=new_switch)


def bisect(history_dir: str, model, test, good, bad=None, *, tokenizer=None) -> dict:
    """Find the first recorded version of the weights at which a test of yours fails. test(model) gets the loaded model set to each
    version checked, with that version's switch attached when `tokenizer` is given, and returns True for a pass. `good` passes and may
    be "base"; `bad` (the newest version by default) fails. About log2 of the versions between them are tried, and it assumes that
    once the test fails it keeps failing up to `bad`.

    The model's weights are put back afterwards as they were, bit for bit, and the last switch is detached. A GRAFT table attached to
    the model answers by itself and hides what the weights do: detach it first.

    Returns first_bad and last_good (version ids), checked (each version tried and what it gave, in order), and the diff of the two."""
    log = _records(history_dir)
    ids = [r["id"] for r in log]
    place = lambda v: -1 if v == BASE else ids.index(_resolve(history_dir, v)["id"])  # noqa: E731
    name = lambda i: BASE if i < 0 else ids[i]  # noqa: E731
    lo, hi = place(good), place(bad)
    if lo >= hi:
        raise ValueError(f"the good version ({name(lo)}) must be earlier than the bad one ({name(hi)})")
    state = model.state_dict()
    held = {n: state[n].detach().clone() for n in (_state_name(state, key) for r in log for key in r["tensors"]) if n is not None}
    checked, switch = [], None

    def ok(i: int) -> bool:
        nonlocal switch
        switch = apply(history_dir, name(i), model, tokenizer=tokenizer, switch=switch)["switch"]
        result = bool(test(model))
        checked.append(dict(version=name(i), ok=result))
        return result

    try:
        if not ok(lo):
            raise ValueError(f"{name(lo)} was given as good and fails the test")
        if ok(hi):
            raise ValueError(f"{name(hi)} was given as bad and passes the test")
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    finally:
        if switch is not None:
            switch.detach()
        with torch.no_grad():
            for n, value in held.items():
                state[n].copy_(value)
    return dict(first_bad=name(hi), last_good=name(lo), checked=checked, diff=diff(history_dir, name(lo), name(hi)))

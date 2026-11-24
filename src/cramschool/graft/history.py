"""A table's history: `commit` records the saved table as a version, and any version loads back as it was saved.

    graft.commit("my.graft", "the three CEOs")                      # saves the table and records it: v1
    for v in versions("my.graft"):
        print(v.id, v.time, v.facts, v.message)
    print(diff_text(diff("my.graft", "v1", "v2")))                  # facts added, removed, changed and built again; settings changed
    old = Graft.load("my.graft", model, tokenizer, version="v1")    # the table as saved at v1
    revert("my.graft", "v1")                                        # the folder holds v1 again, recorded as a new version

Everything in this module is file work: no model is loaded and nothing is built. Inside the table's folder:

    history/versions.jsonl     one line per version; each carries the hash of the line before it, so a changed or missing line shows
    history/objects/<hash>     every file a version is made of, stored once however many versions share it

A version is the files `Graft.load` reads: graft.json, calibration.pt, shared.pt, slots.pt when there is one, and every stored fact's
parts. The shared detectors are kept with each version because they are built once, with the facts stored at that time, and not again
when facts change. A file is named by the hash of what it holds (a tensor's type, shape and bytes), not of its bytes on disk, so the
same parts saved again by another version of torch are the same object.

History is never rewritten: a revert is a new version made of an earlier version's files.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch

HISTORY, LOG, OBJECTS = "history", "versions.jsonl", "objects"
START = "start"
PART_SUFFIXES = (".pt", ".looser.pt", ".prompt.pt")  # a fact's parts, and its optional detectors (table.Graft.save)
SAVED_TABLE = "the saved table"


@dataclass
class Version:
    """One recorded version of a table. files: each file's path in the table's folder and the hash of its content. state: the hash of
    all of it, without the time graft.json was saved. of: for a revert, the version whose files this one took."""
    id: str
    time: str
    message: str
    facts: int
    state: str
    files: dict
    of: str | None
    previous: str
    hash: str


# ---------------------------------------------------------------- hashes

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _feed(h, obj) -> None:
    """Feed what a saved part holds to a hash: tensors by type, shape and bytes, containers in a fixed order."""
    if torch.is_tensor(obj):
        t = obj.detach().cpu().contiguous()
        h.update(f"tensor:{t.dtype}:{tuple(t.shape)}:".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(obj, dict):
        h.update(f"dict:{len(obj)}:".encode())
        for key in sorted(obj, key=repr):
            h.update(f"key:{key!r}:".encode())
            _feed(h, obj[key])
    elif isinstance(obj, (list, tuple)):
        h.update(f"list:{len(obj)}:".encode())
        for item in obj:
            _feed(h, item)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        h.update(f"{type(obj).__name__}:{obj!r}:".encode())
    else:
        raise TypeError(f"a saved part holds a {type(obj).__name__}, which a table's history can't hash")


def content_hash(file: str, name: str | None = None) -> str:
    """The hash of what a table file holds: of a .pt file's tensors and values, of any other file's bytes. name: the file's name in
    the table, when `file` is a copy under another name."""
    if (name or file).endswith(".pt"):
        h = hashlib.sha256()
        _feed(h, torch.load(file, weights_only=True, map_location="cpu"))
        return h.hexdigest()
    with open(file, "rb") as fh:
        return _sha(fh.read())


def _state(meta: dict, files: dict) -> str:
    """What makes two saved tables the same table: every file's content, and graft.json without the time it was saved."""
    body = dict(meta={k: v for k, v in meta.items() if k != "saved"}, files={k: v for k, v in files.items() if k != "graft.json"})
    return _sha(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _line_hash(record: dict) -> str:
    return _sha(json.dumps({k: v for k, v in record.items() if k != "hash"}, sort_keys=True, ensure_ascii=False).encode("utf-8"))


# ---------------------------------------------------------------- the saved table

def _meta(path: str) -> dict:
    file = os.path.join(path, "graft.json")
    if not os.path.exists(file):
        raise ValueError(f"no GRAFT table in {path}")
    with open(file, encoding="utf-8") as fh:
        return json.load(fh)


def _fact_files(meta: dict, has: Callable[[str], bool]) -> list[str]:
    return [f"parts/{f['id']}{suffix}" for f in meta["facts"] for suffix in PART_SUFFIXES if suffix == ".pt" or has(f"parts/{f['id']}{suffix}")]


def table_files(path: str) -> list[str]:
    """The files of the table saved in `path`, as `Graft.load` reads them: paths in the folder, with forward slashes."""
    meta = _meta(path)
    has = lambda rel: os.path.exists(os.path.join(path, *rel.split("/")))  # noqa: E731
    files = ["graft.json", "calibration.pt", "shared.pt"] + (["slots.pt"] if has("slots.pt") else []) + _fact_files(meta, has)
    missing = [rel for rel in files if not has(rel)]
    if missing:
        raise ValueError(f"{path} is not a whole saved table: {', '.join(missing)} missing. Save it with Graft.save first.")
    return files


def _saved(path: str) -> tuple[dict, dict]:
    """(graft.json, each file's content hash) of the table saved in `path`."""
    return _meta(path), {rel: content_hash(os.path.join(path, *rel.split("/"))) for rel in table_files(path)}


# ---------------------------------------------------------------- the log

def _records(path: str) -> list[dict]:
    file = os.path.join(path, HISTORY, LOG)
    if not os.path.exists(file):
        return []
    with open(file, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def versions(path: str) -> list[Version]:
    """Every recorded version of the table in `path`, oldest first."""
    return [Version(**r) for r in _records(path)]


def _resolve(path: str, version) -> dict:
    """The record of "v3", 3, or "latest"."""
    log = _records(path)
    if not log:
        raise ValueError(f"{path} has no recorded versions: record one with commit")
    if version in (None, "latest"):
        return log[-1]
    name = f"v{version}" if isinstance(version, int) else str(version)
    for r in log:
        if r["id"] == name:
            return r
    raise ValueError(f"{path} has no version {name!r}; it has {log[0]['id']} to {log[-1]['id']}")


def _object(path: str, digest: str) -> str:
    return os.path.join(path, HISTORY, OBJECTS, digest)


def _version_meta(path: str, record: dict) -> dict:
    with open(_object(path, record["files"]["graft.json"]), encoding="utf-8") as fh:
        return json.load(fh)


def part_hash(parts) -> str:
    """The hash of a fact's parts as they are in memory: what `content_hash` gives for the file they are saved to."""
    h = hashlib.sha256()
    _feed(h, parts)
    return h.hexdigest()


def facts_fingerprint(rows) -> str:
    """One hash of a table's facts and of what each was built into. rows: [subject, relation, answer, hash of the fact's parts] for
    every fact, in any order. Two tables have the same fingerprint only if they hold the same facts built into the same parts, so a
    fact built again with the same answer changes it."""
    return _sha(json.dumps(sorted(list(r) for r in rows), ensure_ascii=False).encode("utf-8"))


def fingerprint_of(path: str, version=None) -> str:
    """`facts_fingerprint` of a recorded version (the newest by default), from its files."""
    record = _resolve(path, version)
    return facts_fingerprint([f["subject"], f["relation"], f["answer"], record["files"][f"parts/{f['id']}.pt"]] for f in _version_meta(path, record)["facts"])


def facts_of(path: str, version=None) -> list[dict]:
    """The facts a recorded version holds (the newest by default), each as {"id", "subject", "relation", "answer"}."""
    return list(_version_meta(path, _resolve(path, version))["facts"])


def is_committed(path: str) -> bool:
    """Whether the table saved in `path` is its newest version, with nothing changed since."""
    log = _records(path)
    return bool(log) and log[-1]["state"] == _state(*_saved(path))


def is_intact(path: str) -> bool:
    """Whether the history is whole: every line carries the hash of the line before it and its own, and every file a version names is
    in the store and holds what its name says."""
    previous = START
    for n, r in enumerate(_records(path), 1):
        if r.get("previous") != previous or r.get("hash") != _line_hash(r) or r.get("id") != f"v{n}":
            return False
        previous = r["hash"]
        for rel, digest in r["files"].items():
            file = _object(path, digest)
            if not os.path.exists(file) or content_hash(file, rel) != digest:
                return False
        if r["state"] != _state(_version_meta(path, r), r["files"]):
            return False
    return True


# ---------------------------------------------------------------- commit

def commit(path: str, message: str = "", *, of: str | None = None) -> Version:
    """Record the table saved in `path` as a new version, and return it. A table that is its newest version already records nothing and
    that version is returned. `Graft.commit` saves the table first. No model is needed."""
    meta, files = _saved(path)
    state = _state(meta, files)
    log = _records(path)
    if log and log[-1]["state"] == state:
        return Version(**log[-1])
    os.makedirs(os.path.join(path, HISTORY, OBJECTS), exist_ok=True)
    for rel, digest in files.items():
        target = _object(path, digest)
        if not os.path.exists(target):
            shutil.copyfile(os.path.join(path, *rel.split("/")), target + ".tmp")
            os.replace(target + ".tmp", target)
    record = dict(id=f"v{len(log) + 1}", time=time.strftime("%Y-%m-%d %H:%M:%S"), message=message, facts=len(meta["facts"]), state=state,
                  files=files, of=of, previous=log[-1]["hash"] if log else START)
    record["hash"] = _line_hash(record)
    with open(os.path.join(path, HISTORY, LOG), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return Version(**record)


# ---------------------------------------------------------------- a version's files

def extract(path: str, version, to: str) -> dict:
    """Write a version's files into the folder `to`, each under its own name, and return the version's record. A file whose content is
    not what the version recorded is refused."""
    record = version if isinstance(version, dict) else _resolve(path, version)
    for rel, digest in record["files"].items():  # every file is checked before any is written, so a broken history changes nothing
        source = _object(path, digest)
        if not os.path.exists(source):
            raise ValueError(f"the history of {path} has lost {rel} of {record['id']} ({digest[:12]})")
        if content_hash(source, rel) != digest:
            raise ValueError(f"the history of {path} holds a changed {rel} for {record['id']}: its content is not what was recorded")
    for rel, digest in record["files"].items():
        target = os.path.join(to, *rel.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(_object(path, digest), target + ".tmp")
        os.replace(target + ".tmp", target)
    return record


@contextlib.contextmanager
def extracted(path: str, version):
    """A temporary folder holding a version as a saved table, for `Graft.load`."""
    with tempfile.TemporaryDirectory() as folder:
        extract(path, version, folder)
        yield folder


# ---------------------------------------------------------------- revert

def revert(path: str, version, message: str | None = None, *, discard: bool = False) -> Version:
    """Make the folder hold an earlier version again, and record that as a new version (`of` names the earlier one). Nothing is built
    and no model is needed. If the saved table has changes that no version holds, they would be lost: that is refused unless
    `discard` is set. A table that is loaded must be loaded again afterwards; `Graft.revert` does both."""
    record = _resolve(path, version)
    if not discard:
        try:
            committed = is_committed(path)
        except ValueError:
            committed = False
        if not committed:
            raise ValueError(f"the table saved in {path} has changes that no version holds. Record them first (commit), or pass discard to lose them.")
    extract(path, record, path)
    kept = set(record["files"])
    stale = ["slots.pt"] + [f"parts/{f['id']}{suffix}" for f in _version_meta(path, record)["facts"] for suffix in PART_SUFFIXES[1:]]
    for rel in stale:  # files the folder may hold from a later state, which Graft.load would read and this version never had
        file = os.path.join(path, *rel.split("/"))
        if rel not in kept and os.path.exists(file):
            os.remove(file)
    return commit(path, message if message is not None else f"back to {record['id']}", of=record["id"])


# ---------------------------------------------------------------- diff

def _side(path: str, version) -> tuple[str, dict, dict]:
    if version is None:
        meta, files = _saved(path)
        return SAVED_TABLE, meta, files
    record = _resolve(path, version)
    return record["id"], _version_meta(path, record), record["files"]


def diff(path: str, a, b=None) -> dict:
    """What changed from version `a` to version `b`; with no `b`, to the table saved in the folder now.

    added, removed: facts in one and not the other. changed: the same subject and relation with another answer. rebuilt: the same fact
    built again, so its parts differ. relations: added, removed, and those taught other wordings. options, settings (layer, system
    message, model), aliases, exclusions: each as [before, after]. detectors: whether the calibration, the shared detectors or the
    detectors for the place of the answer differ. same: nothing differs."""
    name_a, meta_a, files_a = _side(path, a)
    name_b, meta_b, files_b = _side(path, b)
    key = lambda f: (" ".join(f["subject"].lower().split()), f["relation"])  # noqa: E731
    facts_a, facts_b = {key(f): f for f in meta_a["facts"]}, {key(f): f for f in meta_b["facts"]}
    parts = lambda f, files: tuple(files.get(f"parts/{f['id']}{suffix}") for suffix in PART_SUFFIXES)  # noqa: E731
    row = lambda f: dict(subject=f["subject"], relation=f["relation"], answer=f["answer"])  # noqa: E731
    both = [k for k in facts_a if k in facts_b]
    rel_a, rel_b = meta_a.get("relations", {}), meta_b.get("relations", {})
    pair = lambda name: [meta_a.get(name), meta_b.get(name)]  # noqa: E731
    out = dict(
        a=name_a, b=name_b,
        added=[row(f) for k, f in facts_b.items() if k not in facts_a],
        removed=[row(f) for k, f in facts_a.items() if k not in facts_b],
        changed=[dict(subject=facts_b[k]["subject"], relation=k[1], before=facts_a[k]["answer"], after=facts_b[k]["answer"])
                 for k in both if facts_a[k]["answer"] != facts_b[k]["answer"]],
        rebuilt=[row(facts_b[k]) for k in both if facts_a[k]["answer"] == facts_b[k]["answer"] and parts(facts_a[k], files_a) != parts(facts_b[k], files_b)],
        relations=dict(added=[r for r in rel_b if r not in rel_a], removed=[r for r in rel_a if r not in rel_b],
                       retaught=[r for r in rel_a if r in rel_b and rel_a[r] != rel_b[r]]),
        options={k: [meta_a.get("options", {}).get(k), meta_b.get("options", {}).get(k)]
                 for k in sorted(set(meta_a.get("options", {})) | set(meta_b.get("options", {})))
                 if meta_a.get("options", {}).get(k) != meta_b.get("options", {}).get(k)},
        settings={k: pair(k) for k in ("layer", "system", "model", "fingerprint") if meta_a.get(k) != meta_b.get(k)},
        aliases=pair("aliases") if (meta_a.get("aliases") or {}) != (meta_b.get("aliases") or {}) else None,
        exclusions=pair("exclusions") if (meta_a.get("exclusions") or {}) != (meta_b.get("exclusions") or {}) else None,
        detectors={name: files_a.get(f"{name}.pt") != files_b.get(f"{name}.pt") for name in ("calibration", "shared", "slots")},
    )
    out["same"] = not (out["added"] or out["removed"] or out["changed"] or out["rebuilt"] or any(out["relations"].values()) or out["options"]
                       or out["settings"] or out["aliases"] or out["exclusions"] or any(out["detectors"].values()))
    return out


def diff_text(d: dict) -> str:
    """A diff as lines of text."""
    lines = [f"{d['a']} -> {d['b']}"]
    lines += [f"  + {f['subject']} | {f['relation']} | {f['answer']}" for f in d["added"]]
    lines += [f"  - {f['subject']} | {f['relation']} | {f['answer']}" for f in d["removed"]]
    lines += [f"  ~ {f['subject']} | {f['relation']} | {f['before']} -> {f['after']}" for f in d["changed"]]
    lines += [f"  built again: {f['subject']} | {f['relation']} | {f['answer']}" for f in d["rebuilt"]]
    for what, names in d["relations"].items():
        if names:
            lines.append(f"  relations {what}: {', '.join(names)}")
    lines += [f"  {k}: {before!r} -> {after!r}" for k, (before, after) in {**d["options"], **d["settings"]}.items()]
    for name in ("aliases", "exclusions"):
        if d[name]:
            lines.append(f"  {name}: {d[name][0] or {}} -> {d[name][1] or {}}")
    differing = [name for name, differs in d["detectors"].items() if differs]
    if differing:
        lines.append(f"  measured again: {', '.join(differing)}")
    if d["same"]:
        lines.append("  nothing differs")
    return "\n".join(lines)


# ---------------------------------------------------------------- bisect

def first_bad(path: str, check: Callable[[Version], bool], good, bad=None) -> dict:
    """Find the first version that fails a check, between a version that passes it and a later one that fails it, asking about as
    few versions as a halving search needs. check(version) returns True for a pass. `good` and `bad` are checked first; `bad` is the
    newest version by default. It assumes that once the check fails it keeps failing up to `bad`.

    Returns first_bad and last_good (versions), checked (each version asked and what it gave, in order), and the diff between the two."""
    log = _records(path)
    lo, hi = log.index(_resolve(path, good)), log.index(_resolve(path, bad))
    if lo >= hi:
        raise ValueError(f"the good version ({log[lo]['id']}) must be earlier than the bad one ({log[hi]['id']})")
    checked = []

    def ok(i: int) -> bool:
        result = bool(check(Version(**log[i])))
        checked.append(dict(version=log[i]["id"], ok=result))
        return result

    if not ok(lo):
        raise ValueError(f"{log[lo]['id']} was given as good and fails the check")
    if ok(hi):
        raise ValueError(f"{log[hi]['id']} was given as bad and passes the check")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return dict(first_bad=asdict(Version(**log[hi])), last_good=asdict(Version(**log[lo])), checked=checked,
                diff=diff(path, log[lo]["id"], log[hi]["id"]))

"""Facts from your own data: JSON, JSONL, CSV or TSV files, cramschool bench files, or a Hugging Face dataset.

    load_facts("facts.csv")                                        # columns subject, relation, answer (or target)
    load_facts("facts.jsonl", subject="company", answer="value")   # map your own column names
    load_facts("hf:my-org/company-facts", split="train")           # a Hugging Face dataset (needs `pip install datasets`)
    load_facts("benchmarks/bench40e.json")                         # a bench, or a bench source file

Relations are normalized: "CEO", "chief executive" and "ceo" are the same relation; "headquarters" and "head office" become "hq".
Relations other than owner, ceo, founder and hq need wordings before GRAFT can store them (see cramschool.graft.Relation).
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Iterable

from .graft.wordings import canonical_relation


@dataclass(frozen=True)
class FactRecord:
    subject: str
    relation: str
    answer: str

    def as_tuple(self) -> tuple[str, str, str]:
        return self.subject, self.relation, self.answer


def _record(row: dict, subject: str, relation: str, answer: str, fixed_relation: str | None) -> FactRecord | None:
    ans = row.get(answer)
    if ans in (None, ""):
        ans = row.get("target", row.get("answer"))
    sub = row.get(subject)
    rel = fixed_relation or row.get(relation)
    if not sub or not rel or ans in (None, ""):
        return None
    return FactRecord(str(sub).strip(), canonical_relation(str(rel)), str(ans).strip())


def records_from_rows(rows: Iterable[dict], *, subject: str = "subject", relation: str = "relation", answer: str = "answer",
                      fixed_relation: str | None = None) -> list[FactRecord]:
    out, seen = [], set()
    for row in rows:
        r = _record(row, subject, relation, answer, fixed_relation)
        if r is not None and (r.subject.lower(), r.relation) not in seen:
            seen.add((r.subject.lower(), r.relation))
            out.append(r)
    return out


def load_facts(source: str, *, subject: str = "subject", relation: str = "relation", answer: str = "answer", fixed_relation: str | None = None,
               split: str = "train", limit: int | None = None) -> list[FactRecord]:
    """Facts from a file or a Hugging Face dataset ("hf:<name>[:<config>]"). Column names can be mapped; `fixed_relation` gives every row
    the same relation. Rows without a subject, relation or answer are skipped, and repeats of a (subject, relation) pair keep the first."""
    if source.startswith("hf:"):
        try:
            import datasets
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError("Hugging Face datasets need `pip install datasets`") from error
        name, _, config = source[3:].partition(":")
        rows = datasets.load_dataset(name, config or None, split=split)
        out = records_from_rows(rows, subject=subject, relation=relation, answer=answer, fixed_relation=fixed_relation)
        return out[:limit] if limit else out
    ext = os.path.splitext(source)[1].lower()
    with open(source, encoding="utf-8", newline="") as fh:
        if ext in (".csv", ".tsv"):
            rows = list(csv.DictReader(fh, delimiter="\t" if ext == ".tsv" else ","))
        elif ext == ".jsonl":
            rows = [json.loads(line) for line in fh if line.strip()]
        else:
            data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("facts"), list) and data["facts"] and isinstance(data["facts"][0], list):
                rows = [dict(subject=s, relation=r, answer=t) for s, r, t in data["facts"]]          # a bench source file
            elif isinstance(data, dict) and isinstance(data.get("facts"), list):
                rows = [dict(f, answer=f.get("answer", f.get("target"))) for f in data["facts"]]     # a bench file
            elif isinstance(data, list):
                rows = data
            else:
                raise ValueError(f"{source}: expected a list of facts, or an object with a 'facts' list")
    out = records_from_rows(rows, subject=subject, relation=relation, answer=answer, fixed_relation=fixed_relation)
    return out[:limit] if limit else out


def save_facts(records: Iterable[FactRecord], path: str) -> None:
    """Write facts as CSV or JSONL (by the file's extension)."""
    records = list(records)
    if path.lower().endswith(".csv"):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["subject", "relation", "answer"])
            w.writerows(r.as_tuple() for r in records)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(dict(subject=r.subject, relation=r.relation, answer=r.answer), ensure_ascii=False) + "\n")

#!/bin/bash
# step 11 on blind bench40g: open the source, build the bench, run the unedited model, then the two arms of part 1 on the same build:
# the built-in wordings (the reference) and the built-in wordings plus the editor's
set -e
export PYTHONIOENCODING=utf-8
cd "$(git rev-parse --show-toplevel)"
PY="python"
M=models/Qwen3.5-0.8B
SRC="<scratch>/bench40g/source.json"
R=research/results
G=$R/bench40g
mkdir -p $G runs/bench40g
cp "$SRC" benchmarks/bench40g_source.json
"$PY" -m cramschool bench build --source benchmarks/bench40g_source.json --out benchmarks/bench40g.json --avoid-benches benchmarks/bench40.json benchmarks/bench40b.json benchmarks/bench40c.json benchmarks/bench40d.json benchmarks/bench40e.json benchmarks/bench40f.json benchmarks/devpairs.json
echo "bench built"
"$PY" -u -m cramschool bench run --model $M --bench benchmarks/bench40g.json --method base --out runs/bench40g > $G/base.log 2>&1
cp runs/bench40g/base.jsonl $G/base.jsonl
echo "base done"
T=runs/editor
"$PY" -u scripts/editor_wordings_bench.py $M benchmarks/bench40g.json $G/base.jsonl $T/bench40g_reference > $G/reference.log 2>&1
cp $T/bench40g_reference/rows.jsonl $G/reference.jsonl; cp $T/bench40g_reference/summary.json $G/reference_summary.json
echo "reference (built-in wordings): $(grep -E '^[0-9]+ of 40 pass' $G/reference.log | cut -c1-110)"
"$PY" -u scripts/editor_wordings_bench.py $M benchmarks/bench40g.json $G/base.jsonl $T/bench40g_editor --wordings $R/slots/19_editor/editor_wordings.json --against $G/reference.jsonl > $G/editor_wordings.log 2>&1
cp $T/bench40g_editor/rows.jsonl $G/editor_wordings.jsonl; cp $T/bench40g_editor/summary.json $G/editor_wordings_summary.json
echo "with the editor's wordings: $(grep -E '^[0-9]+ of 40 pass' $G/editor_wordings.log | cut -c1-110)"
echo "chain done"

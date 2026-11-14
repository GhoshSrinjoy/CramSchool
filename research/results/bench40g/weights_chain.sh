#!/usr/bin/env bash
# Step 11, the weights (lab notebook): each table compiled into the model's own neurons and judged in memory (grafted_bench.py, with the
# firm bar), then exported, and the checkpoint loaded back from disk and scored (grafted_disk_bench.py). One job at a time.
# Run from the repository root: bash research/results/bench40g/weights_chain.sh
export PYTHONIOENCODING=utf-8 USE_TF=0
PY="${PY:-python}"  # the interpreter of the environment the package is installed in
MODEL=D:/data/models/Qwen__Qwen3.5-0.8B
BENCH=benchmarks/bench40g.json
R=research/results/bench40g
KNOWN=research/results/slots/16_firm_bar/known_new.json

run() {  # name, table folder, inserted-layer rows
  name=$1; table=$2; rows=$3
  echo "=== $name: compiled in memory ($(date +%H:%M))"
  "$PY" -u scripts/grafted_bench.py $MODEL $BENCH $R/base.jsonl "$table" "$rows" $R/grafted_$name --known $KNOWN || return 1
  echo "=== $name: export ($(date +%H:%M))"
  "$PY" -u -m cramschool graft export --model $MODEL --table "$table" --out runs/editor/grafted_$name || return 1
  echo "=== $name: loaded from disk ($(date +%H:%M))"
  "$PY" -u scripts/grafted_disk_bench.py runs/editor/grafted_$name $BENCH $R/base.jsonl $R/grafted_${name}_disk.jsonl --against "$rows" || return 1
}

run part1 runs/editor/bench40g_editor/table.graft $R/editor_wordings.jsonl
run live runs/editor/live/table.graft $R/live_editor.jsonl
echo "=== done ($(date +%H:%M))"

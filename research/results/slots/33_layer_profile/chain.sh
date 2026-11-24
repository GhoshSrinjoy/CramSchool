#!/bin/bash
# Step 25, the rule's tables on bench40h: B at the profile's pick (22) and C at the layer it ranks lowest among 12 to 22 (12), built by
# the script that built A (step 20's table, layer 20), then each scored with the answer written at the last layer.
export PYTHONIOENCODING=utf-8
cd "$(git rev-parse --show-toplevel)"
PY="python"
M=models/Qwen3.5-0.8B
B=benchmarks/bench40h.json
G=research/results/bench40h
O=research/results/slots/33_layer_profile
A=research/results/slots/31_last_layer/bench40h_all.jsonl
for L in 22 12; do
  T=runs/layer_profile/bench40h_layer$L
  mkdir -p $T
  echo "=== layer $L: build ($(date +%H:%M:%S))"
  "$PY" -u scripts/editor_wordings_bench.py $M $B $G/base.jsonl $T --layer $L > $O/bench40h_layer${L}_built.log 2>&1
  tail -n 3 $O/bench40h_layer${L}_built.log | cut -c1-400
  cp $T/rows.jsonl $O/bench40h_layer${L}_built.jsonl
  echo "=== layer $L: scored with the answer written at the last layer ($(date +%H:%M:%S))"
  "$PY" -u scripts/score_table.py $M $B $T/table.graft $G/base.jsonl $O/bench40h_layer${L}_all.jsonl --last-layer all --against $A > $O/bench40h_layer${L}_all.log 2>&1
  tail -n 6 $O/bench40h_layer${L}_all.log | cut -c1-700
done
echo "=== done ($(date +%H:%M:%S))"

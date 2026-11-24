#!/bin/bash
# Step 26: bench40f's 40 facts built at the model's last layer (23) by the script that built bench40f's table; asked as an inserted layer
# with the answer written at the last layer (T23); then compiled into layer 23's own neurons with the last-layer compile and asked with
# the name switch (G23).
export PYTHONIOENCODING=utf-8
cd "$(git rev-parse --show-toplevel)"
PY="python"
M=models/Qwen3.5-0.8B
B=benchmarks/bench40f.json
G=research/results/bench40f
O=research/results/slots/34_last_layer_neurons
T=runs/last_neurons/bench40f_layer23
mkdir -p $T
echo "=== build at layer 23 ($(date +%H:%M:%S))"
"$PY" -u scripts/validate_api_facts.py $M $T bench40f $G/base.jsonl --layer 23 > $O/build_layer23.log 2>&1
tail -n 3 $O/build_layer23.log | cut -c1-400
echo "=== T23: the inserted layer, answers written at the last layer ($(date +%H:%M:%S))"
"$PY" -u scripts/score_table.py $M $B $T/bench40f.graft $G/base.jsonl $O/t23_all.jsonl --last-layer all --against research/results/slots/31_last_layer/bench40f_all.jsonl > $O/t23_all.log 2>&1
tail -n 6 $O/t23_all.log | cut -c1-700
echo "=== G23: the same facts in layer 23's own neurons ($(date +%H:%M:%S))"
"$PY" -u scripts/grafted_bench.py $M $B $G/base.jsonl $T/bench40f.graft $O/t23_all.jsonl $O/g23 --last > $O/g23.log 2>&1
tail -n 12 $O/g23.log | cut -c1-700
echo "=== done ($(date +%H:%M:%S))"

#!/bin/bash
# The check of the default read layer (TASKS round 2, J3): bench40f's 40 facts built at the profile's pick, layer 22, by the script
# that built bench40f's tables at layers 20 and 23, and asked as an inserted layer with the answer written at the last layer.
# The kill line, written before the run: fewer than 32 of 40 (what the last layer passes) and the default goes back to the last layer.
# Usage: M=<model folder> PY=<python> bash research/results/slots/37_default_read_layer/chain.sh
export PYTHONIOENCODING=utf-8
cd "$(git rev-parse --show-toplevel)"
PY="${PY:-python}"
M="${M:-models/Qwen3.5-0.8B}"
B=benchmarks/bench40f.json
G=research/results/bench40f
O=research/results/slots/37_default_read_layer
T=runs/layer_default/bench40f_layer22
mkdir -p $T
echo "=== build at layer 22 ($(date +%H:%M:%S))"
"$PY" -u scripts/validate_api_facts.py "$M" $T bench40f $G/base.jsonl --layer 22 > $O/build_layer22.log 2>&1
tail -n 3 $O/build_layer22.log | cut -c1-400
echo "=== T22: the inserted layer, answers written at the last layer ($(date +%H:%M:%S))"
"$PY" -u scripts/score_table.py "$M" $B $T/bench40f.graft $G/base.jsonl $O/t22_all.jsonl --last-layer all --against research/results/slots/34_last_layer_neurons/t23_all.jsonl > $O/t22_all.log 2>&1
tail -n 6 $O/t22_all.log | cut -c1-700
echo "=== done ($(date +%H:%M:%S))"

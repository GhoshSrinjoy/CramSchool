#!/bin/bash
# Step 26, a second bench outside the rule: bench40h's 40 facts built at the last layer (23) by the script that built step 20's table;
# asked as a table with the answer written at the last layer; then the two-layer grafted model: first tokens at layer 23 from this table,
# later tokens at layer 20 from step 20's saved table.
export PYTHONIOENCODING=utf-8
cd /d/src/real_time_knowledge_update
PY="/c/Users/Srinjoy Ghosh/.conda/envs/memzip/python.exe"
M=D:/data/models/Qwen__Qwen3.5-0.8B
B=benchmarks/bench40h.json
G=research/results/bench40h
O=research/results/slots/34_last_layer_neurons
T=runs/last_neurons/bench40h_layer23
mkdir -p $T
echo "=== build at layer 23 ($(date +%H:%M:%S))"
"$PY" -u scripts/editor_wordings_bench.py $M $B $G/base.jsonl $T --layer 23 > $O/bench40h_build_layer23.log 2>&1
tail -n 2 $O/bench40h_build_layer23.log | cut -c1-400
echo "=== the table at layer 23, answers written at the last layer ($(date +%H:%M:%S))"
"$PY" -u scripts/score_table.py $M $B $T/table.graft $G/base.jsonl $O/bench40h_t23_all.jsonl --last-layer all --against research/results/slots/33_layer_profile/bench40h_layer22_all.jsonl > $O/bench40h_t23_all.log 2>&1
tail -n 6 $O/bench40h_t23_all.log | cut -c1-600
echo "=== the two-layer grafted model ($(date +%H:%M:%S))"
"$PY" -u scripts/grafted_bench.py $M $B $G/base.jsonl $T/table.graft $O/bench40h_t23_all.jsonl $O/bench40h_g23_two_layers --last --no-cap --later-table runs/end_arms/bench40h/table.graft > $O/bench40h_g23_two_layers.log 2>&1
grep -v "^\[ *[0-9]*s\] \[" $O/bench40h_g23_two_layers.log | tail -n 4 | cut -c1-700
echo "=== done ($(date +%H:%M:%S))"

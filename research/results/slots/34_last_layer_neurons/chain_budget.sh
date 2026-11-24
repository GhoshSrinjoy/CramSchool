#!/bin/bash
# Step 26, the last layer's budget: the same two tables as the confirmed layout, with the later tokens that fit the budget at layer 23 too.
export PYTHONIOENCODING=utf-8
cd "$(git rev-parse --show-toplevel)"
PY="python"
M=models/Qwen3.5-0.8B
O=research/results/slots/34_last_layer_neurons
echo "=== bench40f ($(date +%H:%M:%S))"
"$PY" -u scripts/grafted_bench.py $M benchmarks/bench40f.json research/results/bench40f/base.jsonl runs/last_neurons/bench40f_layer23/bench40f.graft $O/t23_all.jsonl $O/g23_budget --last --no-cap --later-table runs/api_facts_40f/bench40f.graft --budget > $O/g23_budget.log 2>&1
grep -v "^\[ *[0-9]*s\] \[" $O/g23_budget.log | grep "budget\|later tokens\|damage\|grafted model" | cut -c1-700
echo "=== bench40h ($(date +%H:%M:%S))"
"$PY" -u scripts/grafted_bench.py $M benchmarks/bench40h.json research/results/bench40h/base.jsonl runs/last_neurons/bench40h_layer23/table.graft $O/bench40h_t23_all.jsonl $O/bench40h_g23_budget --last --no-cap --later-table runs/end_arms/bench40h/table.graft --budget > $O/bench40h_g23_budget.log 2>&1
grep -v "^\[ *[0-9]*s\] \[" $O/bench40h_g23_budget.log | grep "budget\|later tokens\|damage\|grafted model" | cut -c1-700
echo "=== done ($(date +%H:%M:%S))"

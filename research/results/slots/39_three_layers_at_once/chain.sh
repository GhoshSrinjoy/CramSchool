#!/bin/bash
# Reading the question at three layers at once, by generation (TASKS round 2, K2). Three saved tables hold bench40f's 40 facts, read
# at layers 20, 22 and 23. Attached to the model at the same time, each writing at the last layer, a fact answers where any of the
# three opens. The rule, written before the run: confirmed if the three together pass at least 2 facts more than the best of the
# three alone (33, 35 and 32), with no more facts failing a leak, garbling or damage check than the worst of the three alone.
# Usage: M=<model folder> PY=<python> bash research/results/slots/39_three_layers_at_once/chain.sh
export PYTHONIOENCODING=utf-8
cd "$(git rev-parse --show-toplevel)"
PY="${PY:-python}"
M="${M:-models/Qwen3.5-0.8B}"
B=benchmarks/bench40f.json
G=research/results/bench40f
O=research/results/slots/39_three_layers_at_once
echo "=== layers 20, 22 and 23 at once ($(date +%H:%M:%S))"
"$PY" -u scripts/score_table.py "$M" $B runs/layer_default/bench40f_layer22/bench40f.graft $G/base.jsonl $O/t20_22_23_all.jsonl --last-layer all \
    --also runs/api_facts_40f/bench40f.graft --also runs/last_neurons/bench40f_layer23/bench40f.graft \
    --against research/results/slots/37_default_read_layer/t22_all.jsonl > $O/t20_22_23_all.log 2>&1
tail -n 6 $O/t20_22_23_all.log | cut -c1-900
echo "=== done ($(date +%H:%M:%S))"

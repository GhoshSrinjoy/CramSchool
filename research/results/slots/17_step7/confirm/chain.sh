#!/bin/bash
# step 7, the confirmation on blind bench40f: build the bench, run the unedited model, add its facts through the API and score the
# inserted layer, then compile the same table into the model's own neurons and score the grafted model
set -e
export PYTHONIOENCODING=utf-8
cd "$(git rev-parse --show-toplevel)"
PY="python"
M=models/Qwen3.5-0.8B
R=research/results/bench40f
OUT=research/results/slots/17_step7/confirm
mkdir -p $R runs/bench40f runs/api_facts_40f
"$PY" -m cramschool bench build --source benchmarks/bench40f_source.json --out benchmarks/bench40f.json --avoid-benches benchmarks/bench40.json benchmarks/bench40b.json benchmarks/bench40c.json benchmarks/bench40d.json benchmarks/bench40e.json benchmarks/devpairs.json
echo "bench built"
"$PY" -u -m cramschool bench run --model $M --bench benchmarks/bench40f.json --method base --out runs/bench40f > $R/base.log 2>&1
cp runs/bench40f/base.jsonl $R/base.jsonl
echo "base done"
"$PY" -u scripts/validate_api_facts.py $M runs/api_facts_40f bench40f $R/base.jsonl > $R/graft_api.log 2>&1
cp runs/api_facts_40f/graft_api.jsonl $R/graft_api.jsonl
echo "inserted layer done"
"$PY" -u scripts/grafted_bench.py $M benchmarks/bench40f.json $R/base.jsonl runs/api_facts_40f/bench40f.graft $R/graft_api.jsonl $OUT --known research/results/slots/16_firm_bar/known_new.json > $OUT/run.log 2>&1
echo "chain done"

#!/bin/bash
# step 10, part 2: bench40e's table at layer 20 and bench40f's facts at layer 19, each compiled alone and both compiled into one model
export PYTHONIOENCODING=utf-8
cd "$(git rev-parse --show-toplevel)"
PY="python"
M=models/Qwen3.5-0.8B
R=research/results
OUT=$R/slots/18_layers
KNOWN=$R/slots/16_firm_bar/known_new.json
"$PY" -u scripts/layer_screen.py $M benchmarks/bench40f.json $R/bench40f/base.jsonl $OUT/screen_bench40f.json runs/layers/bench40f 19 > runs/layers/screen_bench40f.log 2>&1
echo "bench40f's table at layer 19 built: $(grep '^layer 19' runs/layers/screen_bench40f.log | cut -c1-110)"
"$PY" -u scripts/score_table.py $M benchmarks/bench40f.json runs/layers/bench40f/layer19.graft $R/bench40f/base.jsonl $OUT/bench40f_layer19_inserted.jsonl > runs/layers/bench40f_layer19_inserted.log 2>&1
echo "inserted layer at 19 on bench40f: $(grep -c PASS runs/layers/bench40f_layer19_inserted.log) pass"
mkdir -p $OUT/bench40f_layer19_alone $OUT/bench40f_layer19_with_layer20 $OUT/bench40e_layer20_with_layer19
"$PY" -u scripts/grafted_bench.py $M benchmarks/bench40f.json $R/bench40f/base.jsonl runs/layers/bench40f/layer19.graft $OUT/bench40f_layer19_inserted.jsonl $OUT/bench40f_layer19_alone --known $KNOWN > $OUT/bench40f_layer19_alone/run.log 2>&1
echo "grafted, layer 19 alone, bench40f: $(tail -1 $OUT/bench40f_layer19_alone/run.log | cut -c1-150)"
"$PY" -u scripts/grafted_bench.py $M benchmarks/bench40f.json $R/bench40f/base.jsonl runs/layers/bench40f/layer19.graft $OUT/bench40f_layer19_inserted.jsonl $OUT/bench40f_layer19_with_layer20 --known $KNOWN --also runs/api_facts/bench40e.graft > $OUT/bench40f_layer19_with_layer20/run.log 2>&1
echo "grafted, both layers, bench40f: $(tail -1 $OUT/bench40f_layer19_with_layer20/run.log | cut -c1-150)"
"$PY" -u scripts/grafted_bench.py $M benchmarks/bench40e.json $R/bench40e/base.jsonl runs/api_facts/bench40e.graft $R/bench40e/graft_api.jsonl $OUT/bench40e_layer20_with_layer19 --known $KNOWN --also runs/layers/bench40f/layer19.graft > $OUT/bench40e_layer20_with_layer19/run.log 2>&1
echo "grafted, both layers, bench40e: $(tail -1 $OUT/bench40e_layer20_with_layer19/run.log | cut -c1-150)"
echo "chain done"

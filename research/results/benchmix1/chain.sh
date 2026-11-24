#!/bin/bash
set -e
cd "$(git rev-parse --show-toplevel)"
PY="python"
M=models/Qwen3.5-0.8B
B=benchmarks/benchmix1.json
O=runs/benchmix1
W=research/results/benchmix1/drafted_relations.json
echo "=== base $(date +%T)"; "$PY" scripts/kinds_bench_run.py base --model $M --bench $B --out $O
echo "=== graft $(date +%T)"; "$PY" scripts/kinds_bench_run.py graft --model $M --bench $B --out $O --wordings $W
echo "=== chunks $(date +%T)"; "$PY" scripts/kinds_bench_run.py chunks --model $M --bench $B --out $O --size 32
for k in 1 2 3 4 5; do
  [ -f $O/chunk$k.json ] || continue
  echo "=== grafted chunk $k $(date +%T)"; "$PY" scripts/grafted_bench.py $M $O/chunk$k.json $O/base.jsonl $O/chunk$k.graft $O/graft.jsonl $O/grafted$k --known research/results/slots/16_firm_bar/known_new.json
done
echo "=== export chunk 1 $(date +%T)"; "$PY" -c "
import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; from cramschool import Graft
t=AutoTokenizer.from_pretrained('$M'); m=AutoModelForCausalLM.from_pretrained('$M', dtype=torch.bfloat16, device_map='cuda').eval()
print(Graft.load('$O/chunk1.graft', m, t).export_grafted('$O/exported1', base_dir='$M'))"
echo "=== grafted chunk 1 from disk $(date +%T)"; "$PY" scripts/grafted_disk_bench.py $O/exported1 $O/chunk1.json $O/base.jsonl $O/disk1.jsonl --against $O/grafted1/rows.jsonl
echo "=== CHAIN DONE $(date +%T)"

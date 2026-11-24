#!/bin/bash
set -e
cd "$(git rev-parse --show-toplevel)"
PY="python"
SP="<scratch>/smoke_kinds"
M=models/Qwen3.5-0.8B
echo "=== base"; "$PY" scripts/kinds_bench_run.py base --model $M --bench "$SP/smoke_bench.json" --out "$SP/run"
echo "=== graft"; "$PY" scripts/kinds_bench_run.py graft --model $M --bench "$SP/smoke_bench.json" --out "$SP/run" --wordings "$SP/stub_relations.json"
echo "=== chunks"; "$PY" scripts/kinds_bench_run.py chunks --model $M --bench "$SP/smoke_bench.json" --out "$SP/run" --size 32
echo "=== grafted in memory"; "$PY" scripts/grafted_bench.py $M "$SP/run/chunk1.json" "$SP/run/base.jsonl" "$SP/run/chunk1.graft" "$SP/run/graft.jsonl" "$SP/run/grafted1" --known research/results/slots/16_firm_bar/known_new.json
echo "=== export"; "$PY" -c "
import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; from cramschool import Graft
t=AutoTokenizer.from_pretrained('$M'); m=AutoModelForCausalLM.from_pretrained('$M', dtype=torch.bfloat16, device_map='cuda').eval()
g=Graft.load(r'$SP/run/chunk1.graft', m, t); print(g.export_grafted(r'$SP/run/exported1', base_dir='$M'))"
echo "=== grafted from disk"; "$PY" scripts/grafted_disk_bench.py "$SP/run/exported1" "$SP/run/chunk1.json" "$SP/run/base.jsonl" "$SP/run/disk1.jsonl" --against "$SP/run/grafted1/rows.jsonl"
echo "=== CHAIN DONE"

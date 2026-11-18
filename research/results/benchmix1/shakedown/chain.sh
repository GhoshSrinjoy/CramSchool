#!/bin/bash
set -e
cd /d/src/real_time_knowledge_update
PY="C:/Users/Srinjoy Ghosh/.conda/envs/memzip/python.exe"
SP="C:/Users/SRINJO~1/AppData/Local/Temp/claude/6cb8499a314c45d1/6403d0a8-2e35-4686-8421-597dbc92a1ad/scratchpad/smoke_kinds"
M=D:/data/models/Qwen__Qwen3.5-0.8B
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

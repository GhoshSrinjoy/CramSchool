#!/bin/bash
# step 20 on blind bench40h, one stage per call so that each can be watched:
#   open      copy the source in and build the bench
#   base      the unedited model
#   table     the 40 facts through the API into one table (built-in wordings, the fast build), saved; scored once as built
#   as_built  the saved table scored as built
#   looser    the saved table with the package's looser start
#   end       the saved table with the end-of-prompt detector (step 19's script)
#   report    the three arms and the rule
set -e
export PYTHONIOENCODING=utf-8
cd /d/src/real_time_knowledge_update
PY="/c/Users/Srinjoy Ghosh/.conda/envs/memzip/python.exe"
M=D:/data/models/Qwen__Qwen3.5-0.8B
SRC="C:/Users/SRINJO~1/AppData/Local/Temp/claude/6cb8499a314c45d1/2507b9cf-b185-4ffc-a0db-4675041c5127/scratchpad/bench40h/source.json"
B=benchmarks/bench40h.json
G=research/results/bench40h
T=runs/end_arms/bench40h
mkdir -p $G runs/bench40h $T
case "$1" in
open)
  cp "$SRC" benchmarks/bench40h_source.json
  "$PY" -m cramschool bench build --source benchmarks/bench40h_source.json --out $B --avoid-benches benchmarks/bench40.json benchmarks/bench40b.json benchmarks/bench40c.json benchmarks/bench40d.json benchmarks/bench40e.json benchmarks/bench40f.json benchmarks/bench40g.json benchmarks/benchmix1.json benchmarks/devpairs.json
  ;;
base)
  "$PY" -u -m cramschool bench run --model $M --bench $B --method base --out runs/bench40h > $G/base.log 2>&1
  cp runs/bench40h/base.jsonl $G/base.jsonl
  tail -n 2 $G/base.log | cut -c1-300
  ;;
table)
  "$PY" -u scripts/editor_wordings_bench.py $M $B $G/base.jsonl $T > $G/built.log 2>&1
  cp $T/rows.jsonl $G/built.jsonl; cp $T/summary.json $G/built_summary.json
  grep -E "taught wordings equal|of 40 pass|facts failing" $G/built.log | cut -c1-400
  ;;
as_built)
  "$PY" -u scripts/score_table.py $M $B $T/table.graft $G/base.jsonl $G/as_built.jsonl > $G/as_built.log 2>&1
  tail -n 5 $G/as_built.log | cut -c1-400
  ;;
looser)
  "$PY" -u scripts/end_arms.py looser $M $B $T/table.graft $G/base.jsonl $G/looser.jsonl > $G/looser.log 2>&1
  grep -E "gate:|attached layer|of 40 pass" $G/looser.log | cut -c1-400
  ;;
end)
  "$PY" -u scripts/end_detector_check.py run $M $B $T/table.graft $G/base.jsonl $G/end_detector.jsonl > $G/end_detector.log 2>&1
  grep -E "gate:|of 40 pass" $G/end_detector.log | cut -c1-400
  ;;
report)
  "$PY" scripts/end_arms.py report $B $T/table.graft $G/as_built.jsonl $G/looser.jsonl $G/end_detector.jsonl --built $G/built.jsonl --blind > $G/report.txt 2>&1
  cat $G/report.txt
  ;;
*)
  echo "usage: chain.sh open|base|table|as_built|looser|end|report"; exit 2
  ;;
esac

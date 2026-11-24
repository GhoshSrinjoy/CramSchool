#!/bin/bash
# step 22: four facts on every company of bench40h, one stage per call:
#   fillers   no GPU: the 80 added facts
#   build     the saved table of step 20 with the 80 facts added through Graft.add, saved
#   as_built  bench40h's 40 facts on the dense table, the option off
#   detector  the same table, the option on
#   report    the four counts, the density and the rule
set -e
export PYTHONIOENCODING=utf-8
cd "$(git rev-parse --show-toplevel)"
PY="python"
M=models/Qwen3.5-0.8B
B=benchmarks/bench40h.json
G=research/results/bench40h
O=research/results/slots/29_dense
T=runs/dense/bench40h_dense.graft
mkdir -p runs/dense
case "$1" in
fillers)
  "$PY" scripts/dense_companies.py fillers $B $O/fillers.json benchmarks/bench40f.json benchmarks/bench40g.json
  ;;
build)
  "$PY" -u scripts/dense_companies.py build $M runs/end_arms/bench40h_option/table.graft $O/fillers.json $T > $O/build.log 2>&1
  tail -n 3 $O/build.log | cut -c1-300
  ;;
as_built)
  "$PY" -u scripts/score_table.py $M $B $T $G/base.jsonl $O/dense_as_built.jsonl --no-prompt-detector --expect-end 0 --loaded $O/fillers.json --against $G/as_built.jsonl > $O/dense_as_built.log 2>&1
  tail -n 6 $O/dense_as_built.log | cut -c1-400
  ;;
detector)
  "$PY" -u scripts/score_table.py $M $B $T $G/base.jsonl $O/dense_detector.jsonl --prompt-detector --expect-end 120 --loaded $O/fillers.json --against $G/option.jsonl > $O/dense_detector.log 2>&1
  tail -n 6 $O/dense_detector.log | cut -c1-400
  ;;
report)
  "$PY" scripts/dense_companies.py report $B $T $O/fillers.json $G/as_built.jsonl $G/option.jsonl $O/dense_as_built.jsonl $O/dense_detector.jsonl --base $G/base.jsonl > $O/report.txt 2>&1
  cat $O/report.txt
  ;;
*)
  echo "usage: chain.sh fillers|build|as_built|detector|report"; exit 2
  ;;
esac

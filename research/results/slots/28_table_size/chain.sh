#!/bin/bash
# step 21: bench40h's saved table with the 180 older facts loaded beside it, one stage per call:
#   predict   no GPU: the stored rows read again with the other-facts check widened to the older answers; and the split by company size
#   as_built  220 facts in one layer, every fact with its strict detectors alone
#   detector  the same layer with the end-of-prompt detectors of bench40h's 40 facts
set -e
export PYTHONIOENCODING=utf-8
cd "$(git rev-parse --show-toplevel)"
PY="python"
M=models/Qwen3.5-0.8B
B=benchmarks/bench40h.json
G=research/results/bench40h
T=runs/end_arms/bench40h_option/table.graft
O=research/results/slots/28_table_size
case "$1" in
predict)
  "$PY" scripts/table_size_check.py predict $M $B $G/base.jsonl as_built=$G/as_built.jsonl looser=$G/looser.jsonl detector=$G/option.jsonl > $O/predict.txt 2>&1
  "$PY" scripts/table_size_check.py split $B as_built=$G/as_built.jsonl looser=$G/looser.jsonl detector=$G/option.jsonl > $O/split.txt 2>&1
  cat $O/predict.txt $O/split.txt | cut -c1-300
  ;;
as_built)
  "$PY" -u scripts/table_size_check.py run $M $B $T $G/base.jsonl $O/as_built_220.jsonl --against $G/as_built.jsonl > $O/as_built_220.log 2>&1
  tail -n 6 $O/as_built_220.log | cut -c1-400
  ;;
detector)
  "$PY" -u scripts/table_size_check.py run $M $B $T $G/base.jsonl $O/detector_220.jsonl --against $G/option.jsonl --prompt-detector > $O/detector_220.log 2>&1
  tail -n 6 $O/detector_220.log | cut -c1-400
  ;;
*)
  echo "usage: chain.sh predict|as_built|detector"; exit 2
  ;;
esac

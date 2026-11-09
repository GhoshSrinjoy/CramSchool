#!/bin/bash
# Phrasing coverage confirmation (lab notebook, "Phrasing coverage, with the big model as editor"): base, editor and blind arms on one fresh draw; the arms reuse base's answer vectors.
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"            # override PYTHON, MODEL and CF (counterfact.json) for your setup
MODEL="${MODEL:-models/Qwen3.5-0.8B}"
CF="${CF:-counterfact.json}"
R=research/results/counterfact
P=$R/phrasing
ONLY="$1"
EXCL="$R/pilot/facts.json $R/confirm/facts.json $R/breadth/facts.json $R/breadth_rev/facts.json $R/breadth_rev2/facts.json $R/dev2_rev2/facts.json $R/dev2_rev3/facts.json $R/confirm2/facts.json $R/full/facts.json $R/twins_confirm/A/facts.json"
COMMON="--model $MODEL --data $CF --screen runs/counterfact/screen.jsonl --heldout $R/full/heldout_questions.json --context 4 --seed 20261115 --per-relation 8 --at-least 6 --only $ONLY --exclude $EXCL --fast --real-world other_relations --twins $R/full/twins_narrow.json"
"$PY" scripts/counterfact_breadth.py $COMMON --relations $R/full/drafted_relations.json --out runs/counterfact/phrasing_base > runs/counterfact/phrasing_base.log 2>&1 || { echo "EXIT base FAILED" >> runs/counterfact/phrasing_base.log; exit 1; }
for arm in editor blind; do
  "$PY" scripts/counterfact_breadth.py $COMMON --relations $P/${arm}_relations.json --vectors-from runs/counterfact/phrasing_base --out runs/counterfact/phrasing_$arm > runs/counterfact/phrasing_$arm.log 2>&1
  echo "EXIT $?" >> runs/counterfact/phrasing_$arm.log
done

#!/bin/bash
# Write a static merging baseline in the submission format and score it on the dev rows.
#   bash scripts/baseline.sh [task_arithmetic|fixed_topk_svd|pretrained]
# (The scored baselines in scores_*.json are recomputed by the evaluators themselves.)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NAME="${1:-task_arithmetic}"
python baseline/static_merge.py --name "$NAME" --output-root outputs/baselines
python eval/train/evaluate.py --submissions outputs/baselines --output "outputs/baselines/scores_dev.json"

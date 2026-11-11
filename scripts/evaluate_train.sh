#!/bin/bash
# Visible dev-slice evaluation: scores every submission under outputs/merged/<name>/
# (plus the fixed baselines) on the labeled public dev rows and writes
# scoring/scores_train.json. Takes no arguments.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p scoring
python eval/train/evaluate.py

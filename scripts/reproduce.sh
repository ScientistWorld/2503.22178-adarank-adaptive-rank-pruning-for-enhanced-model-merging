#!/bin/bash
# Reproduce the reference end to end inside the compute container (no internet needed
# once scripts/download.sh has run): build the splits, run AdaRank (TA, paper defaults)
# into outputs/merged/proposed/, and score it on the dev and held-out rows.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python data/prepare_data.py
if [ -f eval/test/prepare_hidden.py ]; then python eval/test/prepare_hidden.py; fi
python data/prepare_data.py --scrub-raw
[ -f data/assets/zeroshot_heads.safetensors ] || python data/build_heads.py
bash scripts/method.sh "$@"
bash scripts/evaluate_train.sh
if [ -f eval/test/evaluate.py ]; then bash scripts/evaluate_test.sh; fi

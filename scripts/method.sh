#!/bin/bash
# Run the paper's method (AdaRank on Task Arithmetic, paper defaults) and write the merged
# encoder to outputs/merged/proposed/, then score it on the dev rows.
#   bash scripts/method.sh [extra args for method/adarank_clip.py]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python method/adarank_clip.py --name proposed "$@"
bash scripts/evaluate.sh

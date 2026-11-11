#!/bin/bash
# Held-out evaluation: scores every submission under outputs/merged/<name>/ (plus the
# fixed baselines) on the hidden rows and writes scoring/scores_test.json.
# Takes no arguments. Builds the held-out split first if it is not prepared yet.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p scoring
python eval/test/prepare_hidden.py
python eval/test/evaluate.py

#!/bin/bash
# Held-out test-slice evaluation script.
#
# Mirrors scripts/evaluate.sh's contract: read a saved method output artifact
# and write held-out scores. When train scores exist, this scores the same
# method entries rather than a hardcoded method name.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

INPUT="${1:-outputs/adarank_synthetic/predictions.json}"
python eval/test/evaluate.py \
  --input "$INPUT" \
  --output scoring/scores_test.json \
  --train-scores scoring/scores_train.json

#!/bin/bash
# Visible train-slice evaluation script.
#
# Mirrors scripts/evaluate.sh's contract: read a saved method output artifact
# and write a scoring JSON without importing method/ or rerunning inference.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

INPUT="${1:-outputs/adarank_synthetic/predictions.json}"
python eval/train/evaluate.py --input "$INPUT" --output scoring/scores_train.json

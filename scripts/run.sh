#!/bin/bash
# Train/test split evaluation job for the frozen reproduction outputs.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

bash scripts/evaluate_train.sh outputs/adarank_synthetic/predictions.json
bash scripts/evaluate_test.sh outputs/adarank_synthetic/predictions.json

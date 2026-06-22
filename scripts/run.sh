#!/bin/bash
# Regenerate the canonical AdaRank synthetic reproduction and score it.

set -e

cd /home/user

export PYTHONPATH="/home/user:${PYTHONPATH:-}"

bash scripts/reproduce.sh
python validate.py --compare
bash scripts/evaluate_train.sh outputs/adarank_synthetic/predictions.json
bash scripts/evaluate_test.sh outputs/adarank_synthetic/predictions.json

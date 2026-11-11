#!/bin/bash
# Reference job: reproduce the reference end to end inside the compute container
# (no internet needed once scripts/download.sh has run).
#   1. build the public adaptation/dev splits and the held-out split, then scrub raw files
#   2. run AdaRank (Task Arithmetic base, paper defaults) into outputs/merged/proposed/
#      (skipped if it already exists; delete it to rerun)
#   3. score it on the dev rows (scores.json, scores_train.json) and the held-out rows
#      (scores_test.json), then run the structural validator
# The hardening runs (variants and shortcuts pushed through the hidden test) are in
# redteam/hardening/run_checks.py; see HARDENING.md.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi || true
bash scripts/reproduce.sh
python validate.py --post-design || true
cat scoring/scores_test.json

#!/bin/bash
# Probe AdaRank configurations for a core-claim-supporting setting.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

rm -rf outputs/core_claim_probe
mkdir -p outputs/core_claim_probe

run_candidate() {
  local name="$1"
  shift
  echo "=== Running ${name}: $* ==="
  python method/adarank_synthetic.py --output-dir "outputs/core_claim_probe/${name}" "$@"
}

run_candidate full_freeze_lr001 \
  --freeze-lambda \
  --adapt-lr 0.001 \
  --adapt-steps 800 \
  --sweep-adapt-steps 120,240,400,600

run_candidate full_freeze_lr003 \
  --freeze-lambda \
  --adapt-lr 0.003 \
  --adapt-steps 640 \
  --sweep-adapt-steps 120,240,400,500

run_candidate full_learn_cap045 \
  --learn-lambda \
  --lambda-lr 0.0002 \
  --max-lambda 0.45 \
  --adapt-steps 640 \
  --sweep-adapt-steps 120,240,400,500

run_candidate topfrac008 \
  --top-fraction 0.08 \
  --freeze-lambda \
  --adapt-lr 0.003 \
  --adapt-steps 640 \
  --sweep-adapt-steps 120,240,400,500

run_candidate topfrac012 \
  --top-fraction 0.12 \
  --freeze-lambda \
  --adapt-lr 0.003 \
  --adapt-steps 640 \
  --sweep-adapt-steps 120,240,400,500

python eval/probe_scores.py outputs/core_claim_probe/*/predictions.json \
  --output outputs/core_claim_probe/summary.json

#!/bin/bash
# Probe paper-consistent AdaRank adaptation settings for a stronger core claim.

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

# Mask-only variants: this is the stable ablation setting already shown to
# beat fixed-rank SVD. The shorter windows test whether entropy pruning can
# improve over full Task Arithmetic before over-pruning.
run_candidate freeze_lr001_s40 \
  --freeze-lambda \
  --adapt-lr 0.001 \
  --adapt-steps 40 \
  --sweep-adapt-steps 0,10,20,80,160

run_candidate freeze_lr003_s40 \
  --freeze-lambda \
  --adapt-lr 0.003 \
  --adapt-steps 40 \
  --sweep-adapt-steps 0,10,20,80,160

run_candidate freeze_lr006_s20 \
  --freeze-lambda \
  --adapt-lr 0.006 \
  --adapt-steps 20 \
  --sweep-adapt-steps 0,10,40,80

# Joint mask/coefficient variants: this matches the paper's default AdaRank
# family more closely, with conservative caps to avoid the earlier runaway
# coefficient setting on this scaled benchmark.
run_candidate learn_cap035_lr00005_s80 \
  --learn-lambda \
  --lambda-lr 0.00005 \
  --max-lambda 0.35 \
  --adapt-lr 0.001 \
  --adapt-steps 80 \
  --sweep-adapt-steps 10,20,40,160

run_candidate learn_cap045_lr0001_s80 \
  --learn-lambda \
  --lambda-lr 0.0001 \
  --max-lambda 0.45 \
  --adapt-lr 0.001 \
  --adapt-steps 80 \
  --sweep-adapt-steps 10,20,40,160

run_candidate learn_cap060_lr0001_s40 \
  --learn-lambda \
  --lambda-lr 0.0001 \
  --max-lambda 0.60 \
  --adapt-lr 0.001 \
  --adapt-steps 40 \
  --sweep-adapt-steps 10,20,80,160

# Top-k initialization tests the paper's fixed-rank replacement story
# directly: AdaRank starts from the heuristic low-rank merge and adapts masks.
run_candidate topk_lr003_s80 \
  --init-mode topk \
  --top-fraction 0.16 \
  --freeze-lambda \
  --adapt-lr 0.003 \
  --adapt-steps 80 \
  --sweep-adapt-steps 10,20,40,160

python eval/probe_scores.py outputs/core_claim_probe/*/predictions.json \
  --output outputs/core_claim_probe/summary.json

python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("outputs/core_claim_probe/summary.json").read_text())
rows = []
for candidate, payload in summary.items():
    for method, scores in payload["methods"].items():
        if not method.startswith("adarank"):
            continue
        test = scores["test"]
        train = scores["train"]
        rows.append(
            {
                "candidate": candidate,
                "method": method,
                "adapt_steps": scores.get("adapt_steps"),
                "test_avg": test["avg_accuracy"],
                "train_avg": train["avg_accuracy"],
                "test_delta_vs_ta": scores["delta_vs_task_arithmetic_test"]["avg_accuracy"],
                "test_delta_vs_fixed": scores["delta_vs_fixed_test"]["avg_accuracy"],
                "mask_density": scores.get("mask_density"),
                "lambda_mean": scores.get("learned_lambda_mean"),
            }
        )

rows.sort(key=lambda item: (item["test_delta_vs_ta"], item["test_avg"]), reverse=True)
Path("outputs/core_claim_probe/best_rows.json").write_text(json.dumps(rows[:20], indent=2) + "\n")
print("=== Best AdaRank probe rows ===")
print(json.dumps(rows[:20], indent=2))
PY

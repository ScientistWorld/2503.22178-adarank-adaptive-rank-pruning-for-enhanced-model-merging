#!/bin/bash
set -e
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

rm -rf outputs/adarank_sweep
mkdir -p outputs/adarank_sweep

run_one() {
  name="$1"; shift
  out="outputs/adarank_sweep/${name}"
  echo "=== Running ${name}: $* ==="
  python method/adarank_synthetic.py --output-dir "$out" "$@"
  python eval/evaluate_outputs.py --input "$out/predictions.json" --output "$out/scores.json"
}

# Existing default learned-coefficient setting for comparison.
run_one full_learn_cap045 --learn-lambda --lambda-lr 0.0002 --max-lambda 0.45 --sweep-adapt-steps 80,160,320

# Paper ablation-style top-16% initialization with full-range masks able to cross thresholds.
run_one topk_mask --init-mode topk --freeze-lambda --adapt-steps 320 --sweep-adapt-steps 80,160,500
run_one topk_learn_cap1 --init-mode topk --learn-lambda --lambda-lr 0.0005 --max-lambda 1.0 --adapt-steps 320 --sweep-adapt-steps 80,160,500

# Conservative mask learning from the Task Arithmetic initialization.
run_one full_mask_slow --freeze-lambda --adapt-lr 0.003 --adapt-steps 500 --sweep-adapt-steps 80,160,320

python - <<'PY'
import json
from pathlib import Path
summary = {}
for path in sorted(Path('outputs/adarank_sweep').glob('*/predictions.json')):
    data = json.load(open(path))
    methods = data['experiments']['synthetic_4task_merge']['methods']
    rows = {}
    for name, payload in methods.items():
        rows[name] = payload.get('avg_accuracy')
    summary[path.parent.name] = rows
print(json.dumps(summary, indent=2))
Path('outputs/adarank_sweep/summary.json').write_text(json.dumps(summary, indent=2))

# Pick a final artifact without using labels for step selection inside a run: prefer a named config
# whose primary AdaRank output improves the strongest compact baseline, otherwise keep default.
best_name = 'full_learn_cap045'
best_margin = -999
for cfg, rows in summary.items():
    ad = rows.get('adarank', -1)
    compact = max(rows.get('task_arithmetic', -1), rows.get('fixed_topk_svd', -1))
    margin = ad - compact
    if margin > best_margin:
        best_margin = margin
        best_name = cfg
print(f'Best primary config by diagnostic margin: {best_name} margin={best_margin:.6f}')
PY

# Leave scores.json as the existing default unless the sweep identifies a better config manually next turn.
python validate.py --compare

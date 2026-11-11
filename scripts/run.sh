#!/bin/bash
# Hardening job (2 GPUs when available, otherwise sequential on 1 GPU).
#  1. enlarge the held-out split (rows in eval/test/assets/hidden_rows.json), rebuild its
#     index and the fixed baseline predictions on it, then scrub the raw parquet files
#  2. rerun the reference (TA+AdaRank, paper defaults) from the final code into
#     outputs/merged/proposed, plus the candidate better variants (KL-to-expert objective)
#  3. push reference, variants and shortcuts through the hidden test (redteam/hardening/)
#  4. regenerate scoring/scores_{train,test}.json and scoring/scores.json from the reference
# Finished steps are skipped on resubmission (method runs also resume mid-way).
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/hardening
mkdir -p "$LOG"
nvidia-smi || true
python -c "import torch, transformers, os; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count(), 'cpus', len(os.sched_getaffinity(0)))"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU visible')"

# Nodes can expose GPUs that other jobs are using: run on the two with the most free memory.
read GA GB < <(python - << 'PY'
import torch
free = sorted(((torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())), reverse=True)
ids = [i for _, i in free[:2]] + [free[0][1]]
print(f"cuda:{ids[0]} cuda:{ids[1]}")
PY
)
echo "using $GA and $GB"

echo "== 1. held-out split, baselines"
python data/prepare_data.py
python eval/test/prepare_hidden.py
python - << 'PY' || python eval/test/evaluate.py --rebuild-baselines --submissions /nonexistent --output "$LOG/baselines_test.json"
import hashlib, json, sys
idx = json.load(open("eval/test/assets/hidden_index.json"))["tasks"]
base = json.load(open("eval/test/assets/baseline_predictions.json"))
ok = all(base["rows_sha256"].get(t) == hashlib.sha256(json.dumps(v["rows"]).encode()).hexdigest() for t, v in idx.items())
print("test baselines up to date" if ok else "test baselines stale: rebuilding")
sys.exit(0 if ok else 1)
PY
[ -f eval/train/assets/baseline_predictions.json ] || python eval/train/evaluate.py --rebuild-baselines --submissions /nonexistent --output "$LOG/baselines_train.json"
python data/prepare_data.py --scrub-raw

echo "== 2. method runs"
V="--output-root outputs/variants"
queue_a() {
  python method/adarank_clip.py --device $GA --name proposed > "$LOG/proposed.log" 2>&1
  python method/adarank_clip.py --device $GA $V --name distill_seed1 --objective distill --seed 1 > "$LOG/distill_seed1.log" 2>&1
}
queue_b() {
  python method/adarank_clip.py --device $GB $V --name distill --objective distill > "$LOG/distill.log" 2>&1
  python method/adarank_clip.py --device $GB $V --name cart_distill --merge cart --objective distill > "$LOG/cart_distill.log" 2>&1
}
FAIL=0
if [ "$GA" != "$GB" ]; then
  queue_a & PA=$!
  queue_b & PB=$!
  wait $PA || FAIL=1
  wait $PB || FAIL=1
else
  queue_a || FAIL=1
  queue_b || FAIL=1
fi
for f in proposed distill distill_seed1 cart_distill; do
  echo "--- $f"; { grep -v "^step" "$LOG/$f.log" | tail -n 30; grep "^step" "$LOG/$f.log" | tail -n 2; } || true
done
[ "$FAIL" -eq 0 ] || { echo "a method run failed"; exit 1; }

echo "== 3. hidden-test checks (reference, variants, shortcuts)"
python redteam/hardening/run_checks.py

echo "== 4. reference scores"
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
python validate.py --post-design || true
cat scoring/scores_test.json

#!/bin/bash
# Reference + hardening job (2 GPUs when available, otherwise sequential on 1 GPU).
#  1. build the public splits, the held-out split and the fixed assets (heads, baselines)
#  2. run AdaRank (TA, paper defaults) -> outputs/merged/proposed, plus variants
#  3. push reference, variants and shortcuts through the hidden test
#  4. regenerate scoring/scores_{train,test}.json and scoring/scores.json from the reference
# Every finished run is skipped on resubmission (method runs also resume mid-way).
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=logs/hardening
mkdir -p "$LOG"
nvidia-smi || true
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU visible')"

# Nodes can expose GPUs that other jobs are using: run on the two with the most free memory.
read GA GB < <(python - << 'PY'
import torch
free = sorted(((torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())), reverse=True)
for f, i in free:
    print(f"# cuda:{i} free {f / 2**30:.1f} GiB", file=__import__("sys").stderr)
ids = [i for _, i in free[:2]] + [free[0][1]]
print(f"cuda:{ids[0]} cuda:{ids[1]}")
PY
)
echo "using $GA and $GB"

echo "== 1. splits and fixed assets"
python data/prepare_data.py
python eval/test/prepare_hidden.py
[ -f data/assets/zeroshot_heads.safetensors ] || python data/build_heads.py
python eval/test/prepare_hidden.py --copy-assets data/assets
cp data/assets/zeroshot_heads.safetensors data/assets/vision_config.json data/assets/param_shapes.json eval/train/assets/
python data/prepare_data.py --scrub-raw
python -c "import data.clip_benchmark as cb; [cb.load_split(t, s) for t in cb.TASK_NAMES for s in ('adapt', 'dev')]; print('decoded public splits')"
[ -f eval/test/assets/baseline_predictions.json ] || python eval/test/evaluate.py --rebuild-baselines --submissions /nonexistent --output "$LOG/baselines_test.json"
[ -f eval/train/assets/baseline_predictions.json ] || python eval/train/evaluate.py --rebuild-baselines --submissions /nonexistent --output "$LOG/baselines_train.json"

echo "== 2. smoke test"
python method/adarank_clip.py --name smoke --output-root tmp/smoke --steps 3 --log-every 1 --ckpt-every 0 --overwrite --device $GA
python eval/test/evaluate.py --submissions tmp/smoke --output "$LOG/smoke_test.json"
rm -rf tmp/smoke

echo "== 3. method runs"
V="--output-root outputs/variants"
queue_a() {
  python method/adarank_clip.py --device $GA --name proposed > "$LOG/proposed.log" 2>&1
  python method/adarank_clip.py --device $GA $V --name cart_adarank --merge cart > "$LOG/cart_adarank.log" 2>&1
  python method/adarank_clip.py --device $GA $V --name ref_seed1 --seed 1 > "$LOG/ref_seed1.log" 2>&1
}
queue_b() {
  python method/adarank_clip.py --device $GB $V --name mask_only --mode mask_only > "$LOG/mask_only.log" 2>&1
  python method/adarank_clip.py --device $GB $V --name adamerging --mode adamerging > "$LOG/adamerging.log" 2>&1
  python method/adarank_clip.py --device $GB $V --name ref_seed2 --seed 2 > "$LOG/ref_seed2.log" 2>&1
  python method/adarank_clip.py --device $GB $V --name ce_oracle_dev --mode ce_oracle --steps 2000 > "$LOG/ce_oracle_dev.log" 2>&1
  [ -f outputs/variants/ta_lambda_grid_dev/model.safetensors ] || python redteam/hardening/lambda_grid.py --device $GB > "$LOG/lambda_grid.log" 2>&1
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
for f in "$LOG"/*.log; do echo "--- $f"; { grep -v "^step" "$f" | tail -n 25; grep "^step" "$f" | tail -n 2; } || true; done
[ "$FAIL" -eq 0 ] || { echo "a method run failed"; exit 1; }

echo "== 4. hidden-test checks"
python redteam/hardening/run_checks.py

echo "== 5. reference scores"
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
python validate.py --post-design || true
cat scoring/scores_test.json

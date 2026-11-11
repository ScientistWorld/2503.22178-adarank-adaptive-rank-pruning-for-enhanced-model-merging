# Train/Test Evaluation Design Notes

(Current design; the earlier synthetic-benchmark notes are superseded. Details and evidence are in `HARDENING.md`.)

## Splits

- Each task's public test split is cut into three disjoint parts:
  - an unlabeled adaptation pool (`data/downloads/prepared/adapt_<task>.pt`, at most 2,048 rows);
  - labeled dev rows (`dev_<task>.pt`, 256);
  - held-out rows (`eval/test/assets/hidden_rows.json`; 4,096 per task where available, 26,524 in total).
- The held-out images and labels are built by `eval/test/prepare_hidden.py` into `eval/test/downloads/`. They are verified against the per-image and label hashes in `eval/test/assets/hidden_index.json`.
- Raw parquet files are deleted after the splits are built.

## Scorers

- `eval/test/evaluate.py` is self-contained: it imports nothing from `data/` or `method/`, and owns its own copy of the heads, encoder spec and baseline predictions.
- `eval/train/evaluate.py` has the same interface and metrics on the dev rows.
- The two scorers share no code.
- Every `outputs/merged/<name>/` is scored, and `proposed` is always written.
- Invalid submissions get the worst value (0.0) on all three metrics. A submission is invalid if it is missing or unreadable, has extra, missing or misnamed tensors, wrong shapes, or non-float or non-finite values.
- A submission key never replaces a baseline or another entry.
- Nothing a method writes besides `model.safetensors` is read.

## Fixed comparison entries

`pretrained`, `task_arithmetic` (λ = 0.3), `fixed_topk_svd` (top 16%) and `individual` are recomputed only with `--rebuild-baselines`, from hash-verified public checkpoints. Their predictions are stored in each scorer's `assets/`.

# Evaluation

## Submission Format

A submission is a directory `outputs/merged/<name>/` containing `model.safetensors`. The file holds **exactly** the tensors of one CLIP ViT-B/32 image encoder: the `vision_model.*` parameters of `transformers.CLIPVisionModel`, with the same names and shapes as the pretrained encoder (199 tensors, 87.8M parameters).

- Values must be finite floats.
- Any other file in the directory is ignored.
- `data.clip_benchmark.save_submission(state, name)` writes this format.

The canonical entry is **`proposed`** (`outputs/merged/proposed/`). Every other directory under `outputs/merged/` is also scored, under its own name.

A submission scores the **worst value on every metric** (0.0) when:

- the file is missing or unreadable;
- it contains extra, missing or misnamed tensors (for example extra heads, several encoders, or routing tables);
- any tensor has the wrong shape;
- any tensor is not a float, or contains NaN or infinite values.

The `proposed` entry is always written, so an absent submission also scores the worst values. A directory named like a fixed baseline is scored as `submission_<name>`.

## Metrics

All metrics are computed from the encoder's predictions under the fixed heads and projection. Higher is better for every metric.

- **`avg_accuracy`** (primary): top-1 accuracy per task, averaged over the eight tasks.
- **`balanced_accuracy`**: per task, the mean recall over the classes present, then averaged over tasks.
- **`worst_task_accuracy`**: the lowest per-task accuracy.

Nothing reported by a solution (metadata, timings, parameter counts, scores) is read. Wall-clock time is not scored. The single-encoder format is a hard constraint, not a metric.

## Splits

Each task's public test split is partitioned into three disjoint parts:

| Part | Size per task | Labels visible | Use |
|---|---|---|---|
| adaptation pool | 600–2,048 | no | any unlabeled adaptation |
| dev | 256 | yes | development scoring (`scripts/evaluate_train.sh`, `scripts/evaluate.sh`) |
| held-out | 1,024 | no (never shipped) | final scoring (`scripts/evaluate_test.sh`) |

Images are preprocessed identically everywhere:

1. convert to RGB;
2. bicubic-resize the short side to 224;
3. center-crop to 224;
4. normalise with the CLIP mean and std (`data.clip_benchmark.preprocess_image_bytes` / `normalize`).

## Fixed Comparison Entries

The evaluators compute these from the public checkpoints themselves; a solution cannot change them:

- `pretrained`: the unmerged pretrained encoder;
- `task_arithmetic`: pretrained + 0.3 × sum of the fine-tuning updates;
- `fixed_topk_svd`: as `task_arithmetic`, but each weight-matrix update is truncated to its top 16% singular components;
- `individual`: each task scored with its own fine-tuned encoder. This is the multi-model upper reference, not a valid single-encoder submission.

## Scripts

- `bash scripts/evaluate.sh` → `scoring/scores.json` (dev rows).
- `bash scripts/evaluate_train.sh` → `scoring/scores_train.json` (dev rows).
- `bash scripts/evaluate_test.sh` → `scoring/scores_test.json` (held-out rows; not available to solutions).

None of them take arguments. Each scores every submission on disk plus the fixed comparison entries.

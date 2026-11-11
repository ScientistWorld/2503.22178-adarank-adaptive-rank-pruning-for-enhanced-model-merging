# Evaluation Targets

All targets are measured in experiment `vit_b32_8task_merge` on held-out test images: 4,096 per task for SUN397, Stanford Cars, SVHN, GTSRB and MNIST, 3,996 for RESISC45, and 1,024 for EuroSAT and DTD (26,524 in total; each task counts equally). The same metrics on the 256-per-task public dev rows are available for development.

## Primary: multi-task accuracy of one merged encoder

- **Metric**: `avg_accuracy` (higher is better, coefficient 1.0). The mean over the eight tasks of top-1 accuracy, using the fixed zero-shot heads.
- **Reference points** (computed by the evaluator on the same images): `pretrained`, `task_arithmetic`, `fixed_topk_svd` and `individual`.

## Secondary: no task or class left behind

- **Metric**: `worst_task_accuracy` (higher is better, coefficient 0.35). The lowest per-task accuracy.
- **Metric**: `balanced_accuracy` (higher is better, coefficient 0.45). The per-task mean class recall, averaged over tasks.

## Constraint: single compact encoder, label-free construction

- **Deployment size** is enforced as a hard gate. The submission must be exactly one image encoder with the pretrained architecture: no extra tensors, routers, task-indexed weights or ensembles. Its parameter count therefore equals one model regardless of the number of tasks.
- **Validity** is also a hard gate. Non-finite, non-float, misshaped, missing or unreadable weights score the worst value of every metric. The `proposed` entry is always scored, so an absent submission is a failed run, not a tie.
- **Label-free construction** is a rule checked on the method, not a metric (see `CONSTRAINTS.md`). It covers held-out rows, dev labels and external labeled data.

## Ablation (fixed comparison entries)

- `fixed_topk_svd` against `task_arithmetic`: whether a fixed low-rank truncation alone changes merged accuracy.
- `individual`: the multi-model accuracy that merging tries to approach.

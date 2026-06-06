# Evaluation Targets

## Primary: synthetic_4task_merge

Tests whether a single merged model maintains strong average classification accuracy across multiple task-specific evaluation splits after merging independently fine-tuned models from one pretrained backbone.

- **Metric**: `avg_accuracy` (higher is better)

## Constraint: synthetic_4task_merge

Tests whether the merged solution keeps deployment compact rather than retaining one full model per task or adding a large trainable router.

- **Metric**: `merged_param_count` (lower is better)
- **Metric**: `param_multiplier_vs_single` (lower is better)
- **Metric**: `learnable_param_fraction` (lower is better)

## Ablation: synthetic_4task_merge

Compares the adaptive merged result against full task-vector merging and fixed low-rank truncation under the same generated tasks, labels, and evaluation split.

- **Metric**: `avg_accuracy` for `adarank`, `task_arithmetic`, and `fixed_topk_svd` (higher is better)

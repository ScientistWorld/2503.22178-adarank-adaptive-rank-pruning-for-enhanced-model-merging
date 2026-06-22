# Evaluation Targets

## Primary: synthetic_4task_merge

Tests whether a single merged model maintains strong average classification accuracy across multiple task-specific evaluation splits after merging independently fine-tuned models from one pretrained backbone.

- **Metric**: `avg_accuracy` (higher is better)
- **Metric**: `balanced_accuracy` (higher is better)
- **Metric**: `worst_task_accuracy` (higher is better)

## Constraint: synthetic_4task_merge

Tests whether the merged solution keeps deployment compact rather than retaining one full model per task or adding a large trainable router.

- **Metric**: `param_multiplier_vs_single` (lower is better)
- **Metric**: `learnable_param_fraction` (lower is better)

## Ablation: synthetic_4task_merge

Compares the adaptive compact merged result against non-adaptive compact baselines under the same generated tasks, labels, and evaluation split.

- **Metric**: `avg_accuracy` for the proposed method and compact baselines (higher is better)
- **Metric**: `balanced_accuracy` and `worst_task_accuracy` to catch improvements that come from neglecting minority classes or the hardest task.

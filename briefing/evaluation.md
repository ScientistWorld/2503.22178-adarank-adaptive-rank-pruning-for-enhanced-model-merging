# Evaluation

## Metrics

- **Average multi-task accuracy**: classification accuracy is computed separately for each task, then averaged across the tasks in the benchmark split. Higher is better.
- **Worst-task accuracy**: the lowest task accuracy among the benchmark tasks. Higher is better because it discourages solutions that improve the mean by sacrificing a difficult task.
- **Balanced accuracy**: macro recall computed per task and averaged across tasks. Higher is better because it detects class-imbalance failures.
- **Deployment parameter multiplier**: deployed parameter count as a multiple of one merged model. Lower is better when comparing resource use.
- **Learnable adaptation parameter fraction**: parameters optimized during any unlabeled post-training adaptation divided by the total task-specific model parameter pool. Lower is better as a constraint, provided accuracy remains competitive.

Wall-clock time and throughput are excluded from scoring because job hardware may vary. Resource constraints are measured with machine-independent parameter counts.

## Evaluation Protocol

A submitted method must write a saved artifact containing the deployed model state for each method it wants scored. The evaluator reloads that artifact, regenerates the fixed deterministic benchmark data from `data/`, recomputes predictions from the saved model states, and writes `scoring/scores.json`.

Labels from held-out evaluation examples are reserved for scoring. A method may use the benchmark's allowed training labels to construct task-specific source models and may use only unlabeled adaptation inputs for any post-training adaptation step.

The active experiment is `synthetic_4task_merge`: four deterministic heterogeneous classification tasks generated from tracked code, one fixed shared-backbone architecture, task-specific output heads, and held-out task evaluation splits. A valid compact submission deploys one shared backbone plus fixed task heads; the evaluator separately records any extra trainable adaptation capacity reported in the artifact metadata.

## Output Contract

`scripts/evaluate.sh` accepts an artifact path as its first argument and defaults to the canonical reproduction artifact path. It calls only code in `eval/` and `data/`, not any solution implementation. The script must be usable by any future method that produces the same artifact schema.

The evaluator writes the canonical result file `scoring/scores.json` with experiment key `synthetic_4task_merge` and one result entry per method found in the artifact.

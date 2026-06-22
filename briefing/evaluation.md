# Evaluation

## Metrics

- **Average multi-task accuracy**: classification accuracy is computed separately for each task, then averaged across the tasks in the benchmark split. Higher is better.
- **Worst-task accuracy**: the lowest task accuracy among the benchmark tasks. Higher is better because it discourages solutions that improve the mean by sacrificing a difficult task.
- **Balanced accuracy**: macro recall computed per task and averaged across tasks. Higher is better because it detects class-imbalance failures.
- **Deployment parameter multiplier**: deployed parameter count as a multiple of one merged model. Lower is better when comparing resource use.
- **Learnable adaptation parameters**: number of parameters optimized during the unlabeled adaptation stage. Lower is better as a constraint, provided accuracy remains competitive.
- **Learnable parameter percentage**: learnable adaptation parameters divided by the total parameter pool. Lower is better.

Wall-clock timing appears in the paper, but it should not be used as a scoring metric in this gym because job hardware may vary.

## Baselines and Targets

The paper reports vision model merging results for two CLIP vision backbones on benchmarks with 8, 14, and 20 image-classification tasks. Reference methods include unmerged individual models, compact merged models, and adaptive compact merged models.

The key paper targets used in `scoring/reference.json` are:

- Table 1: average accuracy for the main vision merging benchmarks.
- Table 4: 8-task ViT-B/32 and ViT-L/14 accuracy with total and learnable parameter counts.
- Ablations that compare fixed merge rules with adaptive compact merging while holding the deployed model size fixed.

## Evaluation Protocol

Evaluate a submitted method by loading its produced merged checkpoint and running classification on each task's held-out evaluation split. Report per-task accuracies and the unweighted average across the tasks in the corresponding benchmark. Any adaptation may use only unlabeled evaluation inputs under the same split and budget used for the compared methods; labels are reserved for final scoring.

For a scaled reproduction, the minimum useful setting is the 8-task ViT-B/32 benchmark or a clearly documented subset with the same problem structure: multiple fine-tuned checkpoints from one backbone, no labeled training data during merging, one merged model evaluated across all tasks, and resource metrics reported alongside accuracy.

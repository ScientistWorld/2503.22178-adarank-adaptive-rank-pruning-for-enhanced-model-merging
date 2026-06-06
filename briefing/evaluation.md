# Evaluation

## Metrics

- **Average multi-task accuracy**: classification accuracy is computed separately for each task, then averaged across the tasks in the benchmark split. Higher is better.
- **Total parameter pool**: number of task/backbone parameters involved in the merge or adaptation setup, reported in millions where available. Lower is better when comparing resource use.
- **Learnable adaptation parameters**: number of parameters optimized during the unlabeled adaptation stage. Lower is better as a constraint, provided accuracy remains competitive.
- **Learnable parameter percentage**: learnable adaptation parameters divided by the total parameter pool. Lower is better.

Wall-clock SVD and test-time adaptation times appear in the paper, but they should not be used as scoring metrics in this gym because job hardware may vary.

## Baselines and Targets

The paper reports vision model merging results for CLIP ViT-B/32 and ViT-L/14 backbones on 8, 14, and 20 image-classification tasks. Reference methods include the pretrained backbone, individual fine-tuned models, Weight Averaging, Task Arithmetic, Ties-Merging, Consensus Merging, TSV-M, CART, Iso-CTS, AdaMerging variants, and the proposed AdaRank variants.

The key paper targets used in `scoring/reference.json` are:

- Table 1: average accuracy for the main vision merging benchmarks.
- Table 4: 8-task ViT-B/32 and ViT-L/14 accuracy with total and learnable parameter counts.
- Table 5: ablation comparing fixed top-16% selection, restricted adaptation, and full-range adaptation.
- Table 6: ablation isolating coefficient adaptation and mask adaptation.

## Evaluation Protocol

Evaluate a submitted method by loading its produced merged checkpoint and running classification on each task's held-out evaluation split. Report per-task accuracies and the unweighted average across the tasks in the corresponding benchmark. Any adaptation may use only unlabeled evaluation inputs under the same split and budget used for the compared methods; labels are reserved for final scoring.

For a scaled reproduction, the minimum useful setting is the 8-task ViT-B/32 benchmark or a clearly documented subset with the same problem structure: multiple fine-tuned checkpoints from one backbone, no labeled training data during merging, one merged model evaluated across all tasks, and resource metrics reported alongside accuracy.

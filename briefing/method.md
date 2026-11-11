# Paper's Method

## Key Contribution

AdaRank replaces fixed top-k rank truncation in SVD-based model merging with adaptive binary selection of singular components. The method learns which singular components from each task vector and layer should be kept so that the merged model reduces inter-task interference while preserving task performance.

## Approach

The method starts from a pretrained backbone and task-specific fine-tuned checkpoints. For each task and layer, it forms a task vector, decomposes matrix-shaped task-vector weights with SVD, and represents the merged layer as the pretrained layer plus a weighted sum of selected singular components from all tasks.

Each singular component has a binary mask indicating whether it is preserved or pruned. AdaRank optimizes these masks at test time using unlabeled task data and an entropy-minimization objective. Continuous mask parameters are binarized in the forward pass and optimized with a straight-through estimator. The paper jointly adapts the layer/task merge coefficients and binary masks in its default setting, and applies this mechanism to several merging families including Task Arithmetic, TSV-M, CART, and Iso-CTS.

The reproduction is in `method/adarank_clip.py` and runs on the paper's own 8-task CLIP ViT-B/32 benchmark. It follows the official code (`method/official_adarank/`):
- Task vectors are taken relative to the pretrained encoder (TA). CART, with the average of the fine-tuned encoders as origin, is available via `--merge cart`.
- The q/k/v projections of a block are stacked into one 2304x768 matrix, as in open_clip. Each block therefore has 4 masked matrices of rank 768, giving 294,912 mask entries for 8 tasks.
- Non-matrix parameters are merged with coefficients only.
- Masks use a sigmoid straight-through estimator at temperature 10, initialised at +/-0.1. Coefficients are task- and layer-wise (8 x 151, initialised to 0.3).
- Both are adapted jointly with Adam (lr 1e-3, betas 0.9/0.999) on the summed prediction entropy of unlabeled inputs: 16 per task per step, 10,000 steps.

Departures from the paper:
- Adaptation uses a pool of unlabeled test-split rows (at most 2,048 per task) that is disjoint from the scored rows. The paper adapts on the full test set it reports on.
- The final adapted state is scored. The official code reports the best of periodic evaluations that use test labels.

Modes `mask_only`, `adamerging` and `ce_oracle` reproduce the paper's ablations and its supervised oracle.

## Main Claims

- **Primary claim**: adaptive singular-component selection improves average multi-task accuracy over fixed-rank or coefficient-only merging across vision and language model-merging benchmarks.
- **Efficiency claim**: AdaRank keeps a single merged model at inference and uses only a small fraction of the total parameter pool as learnable adaptation variables.
- **Ablation claim**: learning binary masks is independently beneficial, and allowing selection beyond the fixed top-k range improves over adapting only within the top range.
- **Generalization claim**: the same mechanism improves multiple base merging methods and multiple backbones.

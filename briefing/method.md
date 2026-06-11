# Paper's Method

## Key Contribution

AdaRank replaces fixed top-k rank truncation in SVD-based model merging with adaptive binary selection of singular components. The method learns which singular components from each task vector and layer should be kept so that the merged model reduces inter-task interference while preserving task performance.

## Approach

The method starts from a pretrained backbone and task-specific fine-tuned checkpoints. For each task and layer, it forms a task vector, decomposes matrix-shaped task-vector weights with SVD, and represents the merged layer as the pretrained layer plus a weighted sum of selected singular components from all tasks.

Each singular component has a binary mask indicating whether it is preserved or pruned. AdaRank optimizes these masks at test time using unlabeled task data and an entropy-minimization objective. Continuous mask parameters are binarized in the forward pass and optimized with a straight-through estimator. The paper jointly adapts the layer/task merge coefficients and binary masks in its default setting, and applies this mechanism to several merging families including Task Arithmetic, TSV-M, CART, and Iso-CTS.

The scaled reproduction implementation is in `method/adarank_synthetic.py`. The authors' official code release is cloned into `method/official_adarank/` as a reference for the full CLIP and GLUE experiments.

## Main Claims

- **Primary claim**: adaptive singular-component selection improves average multi-task accuracy over fixed-rank or coefficient-only merging across vision and language model-merging benchmarks.
- **Efficiency claim**: AdaRank keeps a single merged model at inference and uses only a small fraction of the total parameter pool as learnable adaptation variables.
- **Ablation claim**: learning binary masks is independently beneficial, and allowing selection beyond the fixed top-k range improves over adapting only within the top range.
- **Generalization claim**: the same mechanism improves multiple base merging methods and multiple backbones.

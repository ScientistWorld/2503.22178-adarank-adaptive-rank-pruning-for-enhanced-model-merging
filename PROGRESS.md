# Progress

## What Works

- Paper analysis and benchmark packaging are complete for a scaled reproduction.
- `method/adarank_synthetic.py` implements the AdaRank computational pipeline on a deterministic synthetic multi-task classification benchmark: fine-tune task models from a shared pretrained backbone, form task vectors, SVD matrix weights, adapt binary masks with STE and unlabeled entropy minimization, and evaluate a merged model.
- `eval/evaluate_outputs.py` scores saved predictions without importing from `method/`.
- `python validate.py --compare` passes structurally after job `0950827d-e38`.
- Job `0950827d-e38` executed the AdaRank pipeline end-to-end and wrote `outputs/adarank_synthetic/predictions.json`, `checkpoints/adarank_synthetic.pt`, and `scoring/scores.json`.

## Results

- First successful run, before the latest fix: Task Arithmetic `0.6660`, fixed top-k SVD `0.5967`, AdaRank `0.5303`, individual models `0.6992` average accuracy. This establishes `method_runs` but not `core_claim`.
- The failed AdaRank result came from initializing TA+AdaRank with top-k masks and saturated mask logits; the implementation is now corrected to initialize TA+AdaRank from full Task Arithmetic masks, use small signed STE logits, and learn layer coefficients with a conservative separate learning rate.

## Remaining

- Rerun the corrected AdaRank setup and compare AdaRank against Task Arithmetic and fixed top-k SVD on the scaled benchmark.
- If AdaRank does not outperform the fixed/full baselines, inspect the synthetic task difficulty and adaptation settings before claiming `core_claim`.

## Issues

- The original full paper benchmark depends on many large vision datasets and fine-tuned CLIP checkpoints. This turn packages a scaled benchmark first to establish the actual AdaRank method and a reusable scoring loop within the available budget.
- Job `0c730250-84c` failed before running the method because Python executed `method/adarank_synthetic.py` with `/home/user/method` as `sys.path[0]`, making sibling package `data/` unavailable. Fixed by exporting `PYTHONPATH=/home/user` in all shell entry points.

## Deviations from Paper

- Paper setting: CLIP ViT-B/32 and ViT-L/14 checkpoints fine-tuned on 8, 14, and 20 image-classification tasks.
- Reproduction setting for the first job: a small MLP backbone fine-tuned on four deterministic synthetic classification tasks.
- Reason: the full benchmark requires many large datasets and external fine-tuned checkpoints; the scaled benchmark preserves the paper's model-merging procedure and AdaRank computation while enabling a quick end-to-end validation job.

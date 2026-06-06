# Progress

## What Works

- Paper analysis and benchmark packaging are complete for a scaled reproduction.
- `method/adarank_synthetic.py` implements the AdaRank computational pipeline on a deterministic synthetic multi-task classification benchmark: fine-tune task models from a shared pretrained backbone, form task vectors, SVD matrix weights, adapt binary masks with STE and unlabeled entropy minimization, and evaluate a merged model.
- `eval/evaluate_outputs.py` scores saved predictions without importing from `method/`.
- `python validate.py --reference-only` passes after staging tracked source files.
- Continuation audit confirmed no completed `outputs/` or `scoring/scores.json` artifacts are present yet, so no milestone above `none` is claimed before a compute job finishes.

## Results

- No reproduced scores yet. The first submitted job will run `scripts/reproduce.sh` and write `scoring/scores.json`.

## Remaining

- Run the first compute job to verify that the method executes end-to-end.
- Inspect `python validate.py --compare` output from the job and fix any numeric or implementation issues before claiming `core_claim`.
- Compare AdaRank against Task Arithmetic and fixed top-k SVD on the scaled benchmark.
- If AdaRank does not outperform the fixed/full baselines, inspect the synthetic task difficulty and adaptation settings before claiming `core_claim`.

## Issues

- The original full paper benchmark depends on many large vision datasets and fine-tuned CLIP checkpoints. This turn packages a scaled benchmark first to establish the actual AdaRank method and a reusable scoring loop within the available budget.

## Deviations from Paper

- Paper setting: CLIP ViT-B/32 and ViT-L/14 checkpoints fine-tuned on 8, 14, and 20 image-classification tasks.
- Reproduction setting for the first job: a small MLP backbone fine-tuned on four deterministic synthetic classification tasks.
- Reason: the full benchmark requires many large datasets and external fine-tuned checkpoints; the scaled benchmark preserves the paper's model-merging procedure and AdaRank computation while enabling a quick end-to-end validation job.

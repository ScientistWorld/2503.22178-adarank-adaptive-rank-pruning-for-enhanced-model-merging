# Progress

## Current state

- **Benchmark:** the paper's own benchmark, merging eight CLIP ViT-B/32 image encoders fine-tuned on SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST and DTD (public tanganke checkpoints, standard zero-shot heads). Earlier iterations used a four-task synthetic stand-in; it was replaced because there the reproduced method lost to Task Arithmetic (see `REDTEAM.md`).
- **Method:** `method/adarank_clip.py`. It implements TA+AdaRank with the official defaults:
  - per-matrix SVD of the task vectors, with q/k/v stacked;
  - binary singular-component masks (STE, temperature 10);
  - task- and layer-wise coefficients;
  - joint Adam adaptation on the prediction entropy of unlabeled pool images, 10,000 steps.

  It also covers the CART base and the paper's ablations.
- **Evaluation:**
  - `eval/test/evaluate.py` is self-contained and scores `outputs/merged/<name>/model.safetensors` on 26,524 held-out rows;
  - `eval/train/evaluate.py` scores the 256-per-task dev rows;
  - the fixed baselines are computed by the evaluators.

## Results (held-out, avg / balanced / worst-task accuracy, %)

| Entry | Score |
|---|---|
| TA+AdaRank (reference) | 86.21 / 85.71 / 71.61 |
| Task Arithmetic | 67.80 / 65.89 / 48.63 |
| Fixed top-16% SVD | 67.54 / 65.61 / 49.32 |
| Pretrained | 47.98 / 46.70 / 31.08 |
| Individual experts (8 models) | 90.58 / 90.56 / 75.17 |

The paper reports 87.9 for TA+AdaRank and 69.2 for Task Arithmetic. Seed range of the reference is 0.36 points avg. A label-free KL-to-expert variant reaches 88.5, which demonstrates headroom. See `HARDENING.md` for variants, shortcuts and noise.

## Remaining gaps

- CART+AdaRank ties TA+AdaRank here (86.5); the paper reports +1.3.
- The 14- and 20-task settings, ViT-L/14, TSV-M/Iso-CTS bases and the language-model experiments are not reproduced.
- The labels of the public test splits are obtainable from the internet. Using them is forbidden by `scoring/CONSTRAINTS.md` and must be caught by the grader.

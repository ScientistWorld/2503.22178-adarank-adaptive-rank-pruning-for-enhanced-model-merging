# Scientist Constraints

## Model

- The deployed model is one CLIP ViT-B/32 image encoder with the pretrained architecture, in the submission format of `briefing/evaluation.md`.
- The visual projection and the per-task zero-shot heads are fixed by the evaluator. Submitted heads or extra tensors make the submission invalid.
- Larger or different backbones are out of scope.

## Sources

Build the merged encoder only from:

- the provided pretrained encoder;
- the eight provided fine-tuned encoders;
- the unlabeled adaptation pool (`data/downloads/prepared/adapt_<task>.pt`).

Re-training the task experts or training a new multi-task model from labeled data is out of scope.

## Labels

- No label may be used to build or select the merged encoder.
- The dev rows' labels (`dev_<task>.pt`) are for measuring progress only. Optimising weights, masks, coefficients or hyper-parameters against them is not allowed.
- Do not fetch the public datasets' labels (or any other labeled data) from outside the workspace.
- The held-out rows are not available, and adapting to them is not allowed.

## Inference

- One forward pass of one encoder per image, with no task-specific weights or routing.
- The task identity is used only to pick the fixed head, exactly as in the evaluator.

## Evaluation

- Submissions are scored only from `outputs/merged/<name>/model.safetensors`. Anything else a method writes (metadata, parameter counts, scores) is ignored.
- Hardware-dependent runtime is not a criterion.

## Compliance Checklist

- [ ] Exactly one image encoder with the pretrained tensor names and shapes; all values finite.
- [ ] Built only from the provided pretrained and fine-tuned encoders and the unlabeled adaptation pool.
- [ ] No labels (dev, held-out, or external) used for training, adaptation or model selection.
- [ ] No routing, ensembling or task-indexed parameters at inference.

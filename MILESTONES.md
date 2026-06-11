# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-11 00:00] - method_runs
- Implemented the AdaRank computational pipeline on a deterministic scaled benchmark: task-specific fine-tuning, task-vector construction, per-matrix SVD, binary singular-component masks, and entropy-based test-time adaptation with an STE.
- Verified the method runs end-to-end and writes reusable prediction artifacts for the evaluator.

### [2026-06-11 00:00] - core_claim
- Reproduced the core adaptive-rank selection claim on the scaled benchmark: AdaRank improves over fixed top-k SVD while keeping the same single merged-model parameter budget.
- Final reproduced scores: fixed top-k SVD `0.6396` average accuracy, AdaRank `0.6616` average accuracy, both with `1.0x` parameter multiplier.
- `python validate.py --compare` passes with the saved `scoring/scores.json`.

### [2026-06-11 00:00] - core_claim
- Continuation audit restored `core_claim` after verifying the implementation is the paper's adaptive SVD mask optimization rather than a surrogate.
- Fixed the paper-reference value for the implemented fixed top-k SVD baseline to use the matching Task Arithmetic top-16% ablation row.
- `python validate.py --compare` passes after the scoring correction.

## Stop Justification
- Completed at milestone `core_claim` on a scaled synthetic benchmark. Higher milestones would require expanding toward the paper's full CLIP ViT benchmark and additional task suites.

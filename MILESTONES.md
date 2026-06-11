# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-11 17:51] - method_runs
- Implemented the AdaRank computation on a deterministic synthetic multi-task benchmark: task vectors from a shared pretrained backbone, per-task/per-layer SVD, binary singular-component masks, STE optimization, and unlabeled entropy minimization.
- Completed end-to-end jobs that generated predictions, a checkpoint, and `scoring/scores.json`.

### [2026-06-11 17:51] - core_claim
- The reproduced AdaRank mask adaptation improves over fixed top-k SVD while preserving a single merged-model parameter budget.
- The evaluator now scores both average accuracy and efficiency constraints, including deployment parameter multiplier and learnable adaptation parameter fraction.

## Stop Justification
- Not stopped. The next submitted job regenerates final artifacts after the continuation audit confirmed the current method/evaluation code compiles and remains a real AdaRank implementation.

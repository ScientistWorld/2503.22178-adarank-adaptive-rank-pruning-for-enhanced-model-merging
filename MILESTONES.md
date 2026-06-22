# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-11 17:51] - method_runs
- Implemented the AdaRank computation on a deterministic synthetic multi-task benchmark: task vectors from a shared pretrained backbone, per-task/per-layer SVD, binary singular-component masks, STE optimization, and unlabeled entropy minimization.
- Completed end-to-end jobs that generated predictions, a checkpoint, and `scoring/scores.json`.

### [2026-06-11 17:51] - core_claim
- The reproduced AdaRank mask adaptation improves over fixed top-k SVD while preserving a single merged-model parameter budget.
- The evaluator now scores both average accuracy and efficiency constraints, including deployment parameter multiplier and learnable adaptation parameter fraction.

### [2026-06-11 17:55] - core_claim
- Job `39cba39c-61a` regenerated the canonical artifacts with the current official-aligned mask initialization and passed `python validate.py --compare`.
- Final refreshed scores: AdaRank `0.6636` average accuracy versus fixed top-k SVD `0.6396`, both at `1.0x` deployment parameter multiplier.

### [2026-06-22 22:20] - core_claim
- Audited the existing scaffold against the paper's computation and kept the current milestone at `core_claim`.
- Tightened the main evaluator to compute every declared benefit and constraint metric from saved predictions with independently regenerated deterministic labels.

## Stop Justification
- Completed at milestone `core_claim`.

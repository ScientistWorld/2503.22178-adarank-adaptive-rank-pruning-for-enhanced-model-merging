# Verification Report

## Initial Assessment

The workspace entered verification at `core_claim`. The paper method implementation appears to preserve the core computation in a scaled synthetic setting, and the latest committed evaluator is designed to score saved model artifacts rather than self-reported metrics. However, that artifact-based evaluator has not yet been exercised in a completed job: no `outputs/adarank_synthetic/artifact.pt` exists in the workspace, and the existing `scoring/scores*.json` files are stale outputs from the previous prediction-based evaluator.

## Issues Found

- Critical/pending: canonical scores are stale relative to the artifact-derived evaluator, and the required saved artifact is missing. A CPU test job must regenerate the artifact and scores before final `done`.
- Minor: strict method-agnostic documentation leaked paper-specific benchmark and method context in `briefing/evaluation.md` and `scoring/TARGETS.md`.
- Minor: `scoring/reference.json` lacked notes explaining which values are published paper references and which active metrics are gym-only additions.
- Minor: the artifact evaluator only accepted one adaptation metadata shape for learnable-parameter accounting, making it less reusable for future methods with a different auditable adaptation representation.
- Cosmetic/residual: the root contains pre-existing unrelated modified special files and deleted trajectory logs (`README.md`, `manifest.json`, `trajectory/...`). These were not part of verification changes.

## Fixes Applied

- Rewrote `briefing/evaluation.md` and tightened `scoring/TARGETS.md` so method-agnostic files describe the benchmark, metrics, and output contract without leaking the paper's method.
- Added an explanatory `notes` field to `scoring/reference.json` for paper-derived references and gym-only scored metrics.
- Extended `eval/artifact_scoring.py` to accept a direct nonnegative `learnable_params` metadata count as well as the existing structured adaptation metadata.
- Removed the unused stale prediction-probe evaluator.
- Kept the milestone at `core_claim` but will submit a CPU test job to regenerate artifact-derived canonical scores before final sign-off.

## Final Judgment

Pending compute verification. The workspace is not ready for final `done` until the artifact-based `scripts/reproduce.sh`, `scripts/evaluate.sh`, train split, and test split complete successfully and refresh the scoring files.

- Milestone: core_claim
- Ready for gym use: partial
- Confidence: medium
- Key limitation: artifact-derived canonical scores still need regeneration in the compute container.

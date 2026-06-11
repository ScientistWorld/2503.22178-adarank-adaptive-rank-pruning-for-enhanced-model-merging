# Train/Test Evaluation Design Notes

## Checked

- Read `eval/train/evaluate.py`, `eval/test/evaluate.py`, `scripts/evaluate_train.sh`, `scripts/evaluate_test.sh`, and `scoring/reference.json` side by side.
- Confirmed the shell scripts target the correct folders and output files: train writes `scoring/scores_train.json`, test writes `scoring/scores_test.json`.
- Confirmed there is no hardcoded method name in the train/test scripts; evaluators enumerate methods from the submitted prediction artifact.
- Confirmed `eval/train/` and `eval/test/` do not import each other.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` are absent.
- Ran `bash -n` on both split evaluation scripts.

## Changed

- Fixed a label-trust shortcut: both split evaluators now regenerate the fixed synthetic benchmark labels from `data.synthetic_tasks.make_tasks` and ignore labels embedded in the candidate artifact. This prevents a candidate from fabricating labels inside `predictions.json` to score perfectly.
- Replaced the visible even/held-out odd positional split with salted deterministic hash splits. Train and test use different salts so the visible split no longer reveals a trivial positional complement.
- Added artifact validation for unknown tasks, duplicate tasks, missing tasks, prediction length mismatches, and invalid class ids.
- Added `balanced_accuracy` to catch methods that boost average accuracy by neglecting minority classes.
- Added `worst_task_accuracy` to catch methods that sacrifice a hard task while preserving the overall mean.
- Added required metric descriptions in `scoring/reference.json` and kept the compactness constraints `param_multiplier_vs_single` and `learnable_param_fraction`.

## Runtime Concerns For Next Iteration

- The split evaluators now import `torch` indirectly through `data.synthetic_tasks`; this should be lightweight because it only regenerates small deterministic tensors, but the runner should verify the evaluation container has the same dependencies available for eval as for method generation.
- Because the benchmark uses a deterministic synthetic data generator that is visible in the workspace, a determined future agent can still reconstruct all benchmark labels outside the evaluator. The label-trust fix closes the easiest artifact-forgery path, but a stronger future design would need hidden generated labels or hidden perturbation seeds if the infrastructure can support scoring predictions on data not visible during development.
- The train/test scripts score saved prediction artifacts and do not rerun inference. This keeps held-out scoring cheap, but future methods must emit predictions for the fixed task names and test-set lengths expected by the evaluators.
- Balanced accuracy depends on each class appearing in the salted split. With the current 512 examples per task this is unlikely to be empty, and the evaluator raises an error rather than silently scoring an unrepresented class set.

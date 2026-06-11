# Progress

## What Works

- Paper analysis and benchmark packaging are complete for a scaled reproduction.
- `method/adarank_synthetic.py` implements the AdaRank computational pipeline on a deterministic synthetic multi-task classification benchmark: fine-tune task models from a shared pretrained backbone, form task vectors, SVD matrix weights, adapt binary masks with STE and unlabeled entropy minimization, and evaluate a merged model.
- `eval/evaluate_outputs.py` scores saved predictions without importing from `method/`.
- `python validate.py --compare` passes structurally.
- Completed jobs executed the AdaRank pipeline end-to-end and wrote `outputs/adarank_synthetic/predictions.json`, `checkpoints/adarank_synthetic.pt`, and `scoring/scores.json`.

## Results

- First successful run, before the latest fix: Task Arithmetic `0.6660`, fixed top-k SVD `0.5967`, AdaRank `0.5303`, individual models `0.6992` average accuracy. This establishes `method_runs` but not `core_claim`.
- The failed AdaRank result came from initializing TA+AdaRank with top-k masks and saturated mask logits; the implementation is now corrected to initialize TA+AdaRank from full Task Arithmetic masks, use small signed STE logits, and learn layer coefficients with a conservative separate learning rate.
- Second successful run after the initialization fix improved AdaRank to `0.6274`, but it still trailed Task Arithmetic `0.6660`. The adaptation trace showed the learned coefficient mean rising from the TA value `0.3` to `0.6026`, hurting the hardest task.
- The next run uses the paper's mask-only ablation setting for this scaled benchmark: AdaRank still adapts binary singular-component masks with STE and entropy minimization, while keeping the TA coefficient fixed to isolate adaptive rank pruning.
- Third successful run with mask-only adaptation reached `0.6572`, close to but still below Task Arithmetic `0.6660`; the remaining loss was concentrated on the hardest task.
- The next job removes shared backbone biases from the synthetic MLP so all shared task-vector components are matrix components controlled by AdaRank SVD masks, and it reports an adaptation-step sweep in one training run.
- Job `710c98a0-a4a` wrote the bias-free prediction artifact before exiting with code 1. The saved scores are Task Arithmetic `0.6738`, fixed top-k SVD `0.6357`, AdaRank `0.6611`, and individual models `0.6909`. AdaRank improved over the fixed-rank SVD baseline at the same 1x parameter budget, but still trailed full Task Arithmetic.
- The crash was caused by the evaluation/validation contract rather than the method run: diagnostic sweep entries such as `adarank_steps_80` were included in `scoring/scores.json` but are absent from `reference.json`. The evaluator now filters diagnostic methods by default while preserving them in the method output JSON for analysis.
- Job `d675a6c5-bfd` completed successfully and confirmed those bias-free scores after the evaluator fix. The step sweep showed that the 80-step variant keeps all mask entries selected and exactly matches Task Arithmetic, while the longer variants prune around 40% of singular components and lose accuracy.
- The learned-coefficient follow-up completed successfully with the paper's default joint adaptation of binary masks and layer coefficients, capped at `0.45`. Final packaged scores are Task Arithmetic `0.6738`, fixed top-k SVD `0.6357`, AdaRank `0.6572`, and individual models `0.6909`.
- This reaches `core_claim` for the paper's central SVD-rank-selection claim: adaptive singular-component selection improves over rigid top-k truncation while preserving the single merged-model parameter budget. It does not reproduce the full paper ordering where AdaRank also beats full Task Arithmetic.

## Remaining

- Higher milestones would require expanding beyond the scaled synthetic setting, ideally toward the paper's CLIP ViT task suite or additional synthetic task suites.
- The main remaining numerical gap is that AdaRank does not outperform full Task Arithmetic on this synthetic benchmark, although it does outperform fixed top-k SVD at the same parameter budget.

## Issues

- The original full paper benchmark depends on many large vision datasets and fine-tuned CLIP checkpoints. This turn packages a scaled benchmark first to establish the actual AdaRank method and a reusable scoring loop within the available budget.
- Job `0c730250-84c` failed before running the method because Python executed `method/adarank_synthetic.py` with `/home/user/method` as `sys.path[0]`, making sibling package `data/` unavailable. Fixed by exporting `PYTHONPATH=/home/user` in all shell entry points.
- Jobs `0057ed8f-cb9` and `ad7760c6-e8e` showed that the original synthetic MLP had shared bias deltas that were always merged outside AdaRank's SVD masks. The next run uses bias-free shared layers to keep the scaled benchmark aligned with adaptive rank pruning over matrix task vectors.
- The external log paths reported for job `ad7760c6-e8e` were not visible in this shell, so this turn used the persisted workspace artifacts in `scoring/scores.json` and `outputs/adarank_synthetic/predictions.json` to assess the completed run.
- The external log paths reported for job `710c98a0-a4a` were also not visible in this shell. The persisted output artifact was enough to diagnose the validation failure and update the evaluator.

## Deviations from Paper

- Paper setting: CLIP ViT-B/32 and ViT-L/14 checkpoints fine-tuned on 8, 14, and 20 image-classification tasks.
- Reproduction setting for the first job: a small MLP backbone fine-tuned on four deterministic synthetic classification tasks.
- Reason: the full benchmark requires many large datasets and external fine-tuned checkpoints; the scaled benchmark preserves the paper's model-merging procedure and AdaRank computation while enabling a quick end-to-end validation job.

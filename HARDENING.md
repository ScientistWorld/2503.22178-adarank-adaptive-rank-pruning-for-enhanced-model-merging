# Hardening report

This is the final hardening pass on this gym. `REDTEAM.md` audited an earlier version: four synthetic vector-classification tasks with a small MLP. In that version the reproduced AdaRank lost to plain Task Arithmetic (65.35% vs 66.77%), and seven shortcuts paid.

The previous hardening pass replaced the synthetic version with the paper's own benchmark: CLIP ViT-B/32 merged across eight image tasks. It also wrote a new, self-contained hidden evaluator, but it did not write this report.

This pass then did four things:
- enlarged the held-out test from 8,192 to 26,524 rows;
- made submission names collision-safe;
- added a label-free variant that clearly beats the reference, to demonstrate headroom;
- reran the reference from the final code, and pushed the reference, the variants and all shortcuts through `scripts/evaluate_test.sh` in one job (`c8bdd57b-94b`, exit 0, `validate.py` passes).

All scores below come from that job unless marked otherwise. The raw evidence is saved:
- `redteam/hardening/results/<case>/scores_test.json` for every case;
- `redteam/hardening/summary.json` for normalised ranks and paired bootstraps;
- `redteam/hardening/results_1024/` and `summary_1024.json` for the same checks on the earlier 1,024-row-per-task test.

Accuracies are percentages. `R` is the rank change against the reference: the sum over metrics of |coefficient| × improvement ÷ the metric's spread across the honest attempts below (coefficients 1 / 0.45 / 0.35 for avg / balanced / worst-task accuracy).

## 1. Do better results score higher?

Yes. "Before" is the synthetic gym audited in `REDTEAM.md`, then the CLIP gym on the earlier 8,192-row held-out set. "After" is the final 26,524-row held-out set. Paired differences use the same rows for every method, with a paired bootstrap over rows.

| Method (all one ViT-B/32 encoder unless noted) | Synthetic gym (REDTEAM) | CLIP, 8,192 rows: avg / bal / worst | **CLIP, 26,524 rows: avg / bal / worst** | Δavg vs ref (paired SE, z) | Δworst (SE) | R |
|---|---|---|---|---|---|---|
| **Reference: TA+AdaRank, paper defaults** | 65.35 (lost to TA) | 86.21 / 85.63 / 71.68 | **86.21 / 85.71 / 71.61** | — | — | 0 |
| Better: AdaRank parameterisation, KL to each expert on the unlabeled pool (`distill`) | — | — | 88.47 / 88.26 / 73.73 | +2.26 (0.19, z 11.6) | +2.12 (0.88) | +0.89 |
| same, seed 1 (`distill_seed1`) | — | — | 88.53 / 88.40 / 73.56 | +2.33 (0.20, z 11.8) | +1.95 (0.88) | +0.91 |
| same on the CART base (`cart_distill`) | — | — | 88.50 / 88.35 / 73.24 | +2.30 (0.21, z 10.9) | +1.64 (0.90) | +0.87 |
| Weight average of three reference seeds (`seed_soup`) | — | 86.58 / 86.02 / 71.78 | 86.56 / 86.05 / 71.12 | +0.35 (0.11, z 3.3) | −0.49 (0.59) | +0.08 |
| Info-max objective (`adarank_im`) | — | 86.88 / 86.77 / 68.07 | 87.12 / 87.01 / 68.63 | +0.91 (0.17, z 5.3) | −2.98 (0.88) | +0.12 |
| CART+AdaRank (paper's stronger base) | — | 86.28 / 85.78 / 71.00 | 86.53 / 86.19 / 70.85 | +0.33 (0.20, z 1.6) | −0.76 (0.88) | +0.07 |
| Weaker: CART, no adaptation | — | 84.46 / 83.97 / 69.53 | 84.86 / 84.16 / 69.70 | −1.35 | −1.90 | −0.58 |
| Weaker: AdaMerging (coefficients only) | — | 84.80 / 84.22 / 69.04 | 84.74 / 84.08 / 69.04 | −1.47 (z −7.1) | −2.56 | −0.66 |
| Weaker: masks only (paper Table 6 ablation) | — | 78.70 / 77.68 / 58.79 | 79.04 / 77.61 / 58.79 | −7.17 | −12.82 | −3.24 |
| Baseline: Task Arithmetic (λ=0.3) | 66.77 | 67.66 / 65.89 / 48.63 | 67.80 / 65.89 / 48.63 | −18.40 | −22.97 | — |
| Baseline: fixed top-16% SVD | 63.42 | 67.40 / 65.72 / 49.32 | 67.54 / 65.61 / 49.32 | −18.67 | −22.29 | — |
| Baseline: pretrained | — | 47.00 / 46.14 / 30.47 | 47.98 / 46.70 / 31.08 | −38.23 | −40.53 | — |
| Individual experts (8 encoders, upper reference) | 70.02 | 90.70 / 90.61 / 75.49 | 90.58 / 90.56 / 75.17 | +4.37 | +3.56 | — |

**Seed noise.** Four reference runs on the 26,524-row test:
- seed 0 (run twice), seed 1 and seed 2;
- the seed-0 rerun from the final code is bitwise identical to the earlier seed-0 run;
- avg range **0.36** points, balanced 0.39, worst-task 1.66;
- paired SE between seeds is 0.13–0.14 points (avg) and 0.8 (worst-task).

**Results:**
- The paper's effect (AdaRank vs Task Arithmetic, +18.4 points) is about 50× the seed range.
- The KL-to-expert variant gains +2.3 points avg, +2.6 balanced and +2.0 worst-task. That is more than 6× the seed range on avg, above the worst-task seed range, and holds across two seeds and two merge bases.
- Every weaker variant ranks below the reference.
- The 8,192 → 26,524-row enlargement cut the paired SE of avg-accuracy differences by about 35% (e.g. seed soup 0.17 → 0.11, z 2.2 → 3.3).

## 2. Reproduction against the paper

Measured on the 26,524-row held-out set, against the paper's Table 1 for ViT-B/32 with 8 tasks:
- **Core claim reproduced:** TA+AdaRank **86.2** vs Task Arithmetic **67.8** (paper 87.9 vs 69.2).
- Other matches: fixed top-16% 67.5 (paper 68.8), pretrained 48.0 (48.0), individual 90.6 (90.5), CART 84.9 (84.7), mask-only 79.0 (79.9).

What is **not** reproduced:
- **CART+AdaRank** reaches 86.5 here, which ties TA+AdaRank (z 1.6). The paper reports 89.2.
- **AdaMerging** reaches 84.7 here, against the paper's 80.1. The gym's version runs 10,000 unclamped steps, so the gap between AdaRank and AdaMerging is 1.5 points (z 7.1) instead of the paper's 7.8.

Departures from the paper, all documented in `scoring/reference.json`:
- The method adapts on a pool of at most 2,048 unlabeled test-split images per task that is disjoint from the scored rows. The paper adapts on the full test set it scores.
- The final adapted state is scored. The official code reports the best of periodic evaluations that use test labels.
- The public tanganke fine-tuned checkpoints replace the authors' checkpoints.

Mask/coefficient parameterisation, STE (temperature 10, ±0.1 init), λ init 0.3, Adam lr 1e-3, 16 images per task per step and 10,000 steps all follow the official config `method/official_adarank/vision/configs/adarank_B_T8/adarank_TA.yaml`.

## 3. Major shortcuts: fix and score before → after

"Before" is the synthetic gym's hidden test from `REDTEAM.md`. "After" is the final evaluator; each case was staged on disk and scored by a fresh no-argument `bash scripts/evaluate_test.sh`.

| Shortcut | Before | Fix | After (proposed entry, avg acc) |
|---|---|---|---|
| Regenerate test labels from the visible generator and train on them | 100 / 100 / 100, R +3.43 | Real data; held-out rows, labels and per-image hashes live only in `eval/test/`. The raw parquet files are deleted after the splits are built, and no code outside `eval/test/` can rebuild the held-out labels. | Not reproducible from the workspace. Still possible through the public internet: see §4. |
| Retrain the "fixed" task heads | 77.4, R +1.24 | Heads and projection are owned by `eval/test/assets/`. A submission must be exactly the 199 encoder tensors. | `extra_head_tensors`: 0.0 on every metric (failed run) |
| Under-report adaptation cost; negative parameter count | R +0.0000014 / +0.348 | Nothing a method writes is read. Deployment size is a hard format gate. Adaptation cost is not scored (not recomputable from a submission). | `metadata_underreport`: identical model, identical score (R 0); nothing gained |
| Empty method set | tie | `proposed` is always written. Absent or invalid = worst value on every metric. | `empty_submissions`: 0.0 / 0.0 / 0.0 |
| Train-score allowlist hides methods | tie | Every `outputs/merged/<name>/` is scored; no allowlist exists. | `filter_omission`: bad `proposed` scored 4.07; unlisted entry scored too |
| Edit the data generator / labels | R +31.3 | `eval/test/evaluate.py` imports nothing from `data/` and verifies hidden images and labels against hashes. | `data_tamper` (dev labels → 0, heads zeroed, `data/clip_benchmark.py` edited): 4.07, the bad model's true score |
| Per-task models under one entry (routing) | accepted, soft penalty | Hard gate: exactly one encoder's tensor names and shapes. | `individual_routing`: 0.0 on every metric |
| NaN weights | accepted | Non-finite, non-float or misshaped tensors fail the run. | `nan_weights`: 0.0 on every metric |
| Prewritten scores / metadata | did not pay | Scores are always recomputed from `model.safetensors`. | `forged_scores`: 4.07 |
| Invalid count / empty payload crashes the evaluator | no score file | Unreadable file → worst value, no crash. | `empty_payload`: 0.0 on every metric |
| Submission named like a baseline, or like its renamed key (new this pass) | could overwrite another submission's scores | A submission key never replaces an existing entry; it gets more `submission_` prefixes (both evaluators). | `name_collision_chain`: baselines intact; both submissions scored under distinct keys |
| A single expert submitted as the "merge" (new check) | — | (format-valid; scored honestly) | `expert_copy`: 45.28 |

Brute force:
- **Coefficient grid search** on the visible dev labels (`ta_lambda_grid_dev`) scores 69.7, far below the reference.
- **Exhaustive mask search** is infeasible: 2^294,912 binary masks.
- **Supervised oracle** (`ce_oracle_dev`): masks and coefficients trained with cross-entropy on the 2,048 visible dev labels score 86.31 / 86.03 / 70.85. That is +0.10 avg (z 0.5) and −0.76 worst-task, R +0.0005: a tie within noise, so it does not pay. It is also a rule violation.

## 4. Shortcuts left open (for the separate grader)

- **Public test labels.** The held-out rows are a subset of public Hugging Face test splits. A method could download those labels, or ones from another copy of the datasets, and train on them; `data/fetch.py --force-raw` fetches the files. The evaluator cannot tell such a model apart. `CONSTRAINTS.md` forbids it. This is the largest remaining shortcut and needs a provenance check by the grader.
- **Transductive adaptation on more public test images.** A method could adapt without labels on all public test images, which include the held-out rows. This is out of bounds by `CONSTRAINTS.md` and probably worth a fraction of a point. It cannot be detected from the weights.
- **External multi-task checkpoints.** An encoder trained elsewhere with labels (e.g. a published multi-task CLIP) is a format-valid single encoder. Out of bounds; the grader must check what the merge was built from.
- **Dev labels.** Training or selecting on the 256 labeled dev rows per task does not pay on the held-out test (ce_oracle_dev ties), but it is a rule violation.
- **Train-side tampering.** `scores_train.json` is computed from editable dev files in `data/`, so a method can fake its own dev score. The ranked test side does not depend on these files (`data_tamper` case).

## 5. Headroom

- **Bounds.** No scored reference metric is at its bound. Reference 86.2 / 85.7 / 71.6 vs the bound of 100 on each; the individual experts reach 90.6 / 90.6 / 75.2.
- **Better variant.** A label-free variant beats the reference on every metric, far beyond paired and seed noise: KL to each task's own fine-tuned encoder on the unlabeled pool, inside the AdaRank parameterisation, scores 88.5 / 88.3 / 73.7.
- **Not saturated.** That variant still sits 2.1 points below the individual experts, so the test is not saturated by a simple variant.
- **Baselines.** Task Arithmetic and fixed top-k sit about 18 points below the reference.
- **No new experiment.** No extra scored experiment was added; it was not needed. If a future version needs more headroom, the paper's 14- and 20-task settings are where TA+AdaRank falls furthest short: 82.1 / 81.4 vs individual 90.3 / 90.5, with CART/TSV-M+AdaRank about 5 points better.

## 6. Reference score before → after

- **Synthetic gym (REDTEAM):** 65.35 / 56.16 / 55.29, below Task Arithmetic's 66.77, so the paper's direction was not reproduced.
- **CLIP gym, 8,192 rows:** 86.21 / 85.63 / 71.68.
- **CLIP gym, final (26,524 rows):** **86.21 / 85.71 / 71.61**, with Task Arithmetic at 67.80.

The rerun from the final code reproduced the earlier weights bit for bit, so only the larger test changed the numbers. `scoring/scores_test.json`, `scores_train.json` and `scores.json` are regenerated from this reference alone.

## 7. Weights

Unchanged: one experiment, weight 1. Coefficients are 1.0 for avg accuracy (the paper's metric), 0.45 for balanced accuracy and 0.35 for worst-task accuracy, all higher-is-better. The two secondary metrics are gym additions: they stop a method from raising the average by giving up a task or minority classes. The `adarank_im` row (+0.9 avg, −3.0 worst) shows them working, since it ranks only marginally above the reference.

The test enlargement does not change the weights. Every task still counts equally, whatever its row count.

## 8. Changes made in this pass

- `eval/test/assets/hidden_rows.json`: held-out rows enlarged to 4,096 per task where the public test split allows (RESISC45 3,996; EuroSAT and DTD 1,024, since their splits are used up by the adaptation pool and dev rows). The original 1,024 rows per task are kept.
- `eval/test/prepare_hidden.py`: re-indexes a task whose row list changed.
- `eval/test/assets/hidden_index.json` and `baseline_predictions.json` rebuilt on the new rows; raw parquet scrubbed again.
- `eval/test/evaluate.py` and `eval/train/evaluate.py`: collision-safe submission keys.
- `method/adarank_clip.py`: optional `--objective distill` used only for the headroom check. The default is unchanged, which the bitwise-identical reference rerun confirms.
- `redteam/hardening/run_checks.py`:
  - two new shortcut cases (`name_collision_chain`, `expert_copy`);
  - the stashed reference is refreshed on every run;
  - R is normalised by the honest variants' spread;
  - paired bootstrap now covers both avg and worst-task accuracy.
- Docs: `briefing/evaluation.md`, `scoring/TARGETS.md` and the `scoring/reference.json` notes give the new held-out size. Paper numbers are unchanged.

## 9. What I could not fix

- **CART+AdaRank gap.** Its paper gain over TA+AdaRank (+1.3) does not appear here (+0.3, not significant). The likely causes are the smaller adaptation pool, scoring the final state instead of the best checkpoint, and different expert checkpoints. This is a secondary claim; the core claim holds.
- **Public data.** Label leakage and transductive adaptation through the public datasets cannot be closed by the evaluator; see §4.
- **Worst-task noise.** Worst-task accuracy is still noisier than the average (seed range 1.7 points), because the hardest tasks include DTD, which has only 1,024 held-out rows.

## Reproduce

- Reference: `bash scripts/run.sh`, which runs `scripts/reproduce.sh` and the validator.
- Hardening checks: `python redteam/hardening/run_checks.py` after the variant runs listed in the git history of `scripts/run.sh` (commit "Hardening: enlarge held-out split…").
- Re-summarise: `python redteam/hardening/run_checks.py --summarize-only`.

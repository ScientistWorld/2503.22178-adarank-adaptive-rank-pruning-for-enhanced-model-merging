> **Historical (pre-hardening).** Everything in this folder except `hardening/` audits the earlier synthetic 4-task MLP gym. That gym was replaced in the hardening iteration by the paper's CLIP ViT-B/32 8-task benchmark. `audit.py` and the case scripts import modules that no longer exist and are kept only as the record of the "before" scores in `REDTEAM.md`. The current checks are in `redteam/hardening/` (see `HARDENING.md`).

This audit uses the visible generator, model definition, visible scorer, and
reproduction implementation (improve mode). It never changes `eval/`, test
scripts, or the Python environment. The paper is consulted only for the
reproduction comparison, never to construct shortcuts.

Run all experiments in the gym compute container:

```bash
PYTHONPATH=/home/user python redteam/audit.py --suite
```

Each named folder has a `run.sh` which stages its artifact, runs the visible
evaluation, then invokes `bash scripts/evaluate_test.sh` **with no arguments**.
The hidden scorer is a separate process reading only saved files. No training
process, monkey patch, or background process survives into it. State is restored
after each experiment. Scores and small factual summaries ship with the audit;
tensor artifacts, caches, and logs are ignored and can be regenerated.

Reference seed reruns vary initialization and source-model training randomness
while keeping the benchmark data seed at 7. Merely setting artifact config.seed
to another value is rejected by the evaluator and would change the data, so it
would not be a valid seed-spread measurement.

All honest candidates retain the original baseline entries, isolating changes
to `adarank`. Extra source training is explicitly a quality-control probe, not
an allowed contribution under the fixed-source model-merging constraint.
The rank grid searches only visible-slice scores. Neither adaptation nor grid
selection consults the hidden slice. Label-leak shortcuts explicitly violate
that separation by regenerating labels using the supplied data code.

`summary.json` reports raw metrics, per-metric seed ranges, and an illustrative
rank calculated over this audit's population using the required directions,
absolute coefficients, experiment weight, and empirical standard deviations.
Future attempts change those deviations, so these scalar ranks are not fixed
gym constants. Missing metrics in a successful test contribute zero, as
specified by the audit.
Rejected tests without a fresh score file have a null rank; no external failure
penalty is assumed. Recompute this summary without PyTorch using
`python3 redteam/summarize.py`.
The original on-disk hidden scores, rather than the paper's unrelated vision
accuracy values in `reference.json`, define zero improvement.

`shortcut_data_labels` edits an ordinary data generator and saves constant
model states. It changes the task rather than improving a method. The
comparison remains on identical data for every method, and all outputs are
standard tensors. `shortcut_underreport` and `shortcut_negative_count` change
only saved metadata. `shortcut_filter_omission` changes only the saved method
dictionary and train-score method allowlist. These are benchmark scoring
shortcuts, not evaluator-process tampering.

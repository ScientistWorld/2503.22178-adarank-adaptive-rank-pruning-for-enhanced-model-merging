# Experiment Overview

## vit_b32_8task_merge

Establishes whether a single image encoder can serve eight heterogeneous classification tasks nearly as well as eight separately fine-tuned encoders, when it is built only from those fine-tuned encoders, their shared pretrained starting point, and unlabeled task images.

The tasks are scenes, car models, aerial scenes, satellite land use, street-view digits, traffic signs, handwritten digits and textures. All share one CLIP ViT-B/32 architecture and fixed zero-shot readouts.

The experiment compares the submitted encoder against fixed reference points that the evaluator computes itself on the same held-out images:

- the unmerged pretrained encoder (lower reference);
- plain weight-update addition, with and without a fixed low-rank truncation (simple merging baselines);
- the individual fine-tuned encoders (multi-model upper reference).

Average accuracy measures overall multi-task quality. Worst-task and balanced accuracy check that the gain is not bought by abandoning a task or the rarer classes of a task. The single-encoder submission format enforces the deployment constraint (no growth with the number of tasks) as a hard requirement.

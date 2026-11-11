# Research Direction

## Research Problem

Multi-task model merging: combine several encoders fine-tuned from one pretrained model into a single encoder of the same size that performs well on all of their tasks. The central difficulty is interference between the fine-tuning updates, which makes simple addition lose much of the experts' accuracy.

## Core Contribution Area

Better ways to select, transform, weight or combine the task-specific weight updates into one set of weights. This includes ways to use the unlabeled adaptation images for that purpose.

## Approach Scope

In-bounds:

- new decompositions or parameterisations of the task updates;
- selecting or re-weighting update components per task and layer;
- interference-reducing transformations such as orthogonalisation or re-centering;
- better initialisation, objectives or schedules for unlabeled adaptation of merge parameters;
- regularisation;
- data-efficient use of the adaptation pool.

The output must remain one encoder of the fixed architecture.

## Out of Bounds

- Using any labels (dev rows, held-out rows, external data) to build or select the merged encoder.
- Keeping several encoders, routers, task-indexed weights, ensembles or mixtures of experts at inference.
- Re-training experts or a multi-task model from labeled data; other or larger backbones.
- Changing the fixed heads, projection, preprocessing or evaluation code.

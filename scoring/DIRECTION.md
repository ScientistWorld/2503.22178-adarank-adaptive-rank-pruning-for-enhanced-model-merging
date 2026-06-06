# Research Direction

## Research Problem

This benchmark studies multi-task model merging: combining several independently fine-tuned models that share one pretrained backbone into a single model that performs well across all of their tasks. The central question is how to reduce task interference during merging while preserving the practical benefit of a compact deployed model.

## Core Contribution Area

A scientist should work on improving how task-specific weight updates are selected, transformed, weighted, or combined before producing the final merged checkpoint. The contribution must remain within model merging for homologous checkpoints rather than changing the problem into routing, ensembling, multitask retraining, or architecture scaling.

## Approach Scope

In-bounds work includes new ways to decompose task updates, select or weight update components, regularize interference, adapt merge parameters with unlabeled data, calibrate the merged model after merging, or improve coefficient search. The scientist may modify the merge objective, parameterization, initialization, or adaptation schedule as long as the final artifact is still one merged model for the fixed backbone and task set.

## Out of Bounds

- Replacing the benchmark backbone with a larger or different pretrained architecture.
- Keeping all task-specific models and using a router, task index, ensemble, or mixture of experts at inference.
- Training a new multi-task model from the original labeled training sets instead of merging fine-tuned checkpoints.
- Adding external labeled data or teacher predictions from outside the provided checkpoint and dataset set.
- Optimizing for hardware-specific runtime instead of model-merging quality and resource constraints.

# Problem Definition

## Research Question

How can several task-specialized models that share the same pretrained backbone be combined into one multi-task model that performs well across all tasks? The benchmark focuses on model merging: given a pretrained model and multiple independently fine-tuned checkpoints, produce a single set of model weights that can be evaluated on every task without training on the original labeled training sets.

## Why It Matters

Maintaining one full model per task scales poorly in storage, memory, and deployment complexity. A strong merging procedure would let practitioners reuse public fine-tuned models while serving a compact multi-task model with much of the accuracy of the individual experts.

## Success Criteria

A successful solution should:

- Achieve high average accuracy across all target tasks after merging.
- Preserve performance as the number and diversity of tasks grows.
- Use the same base architecture and evaluation data as the compared methods.
- Avoid retaining a growing collection of task-specific parameters or routers at inference time.
- Keep any adaptation-time learnable parameters small relative to the total parameter pool.

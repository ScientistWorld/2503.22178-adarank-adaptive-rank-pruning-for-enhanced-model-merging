# Experiment Overview

## synthetic_4task_merge

Tests the paper's central model-merging question in a scaled, deterministic setting: whether one merged model can perform well across several heterogeneous classification tasks while avoiding the resource cost of storing all individual fine-tuned models. The experiment also reports adaptation-parameter overhead and includes fixed and full-vector merge baselines so future agents can tell whether improvements come from better merging rather than from a larger parameter budget.

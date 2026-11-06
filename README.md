# AdaRank: Adaptive Rank Pruning for Enhanced Model Merging

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2503.22178"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2503.22178-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2503.22178](https://arxiv.org/abs/2503.22178) |
| **Domain** | Model Merging / Multi-Task Learning |
| **TL;DR** | AdaRank improves multi-task model merging by learning which SVD components of task vectors to keep, rather than using a fixed top-k rank rule. |

## Summary

The paper studies how to merge multiple independently fine-tuned models into a single model that performs well across all tasks. It argues that fixed top-k SVD truncation is suboptimal because high singular-value components can help one task while interfering with others, and because useful rank varies across tasks and layers. AdaRank addresses this by learning binary masks over singular components with unlabeled test-time entropy minimization. The reported results show higher average accuracy than static and adaptive merging baselines while keeping deployment to a single merged model.

## Problem

How can several task-specialized models that share the same pretrained backbone be combined into one multi-task model that performs well across all tasks? The benchmark focuses on model merging: given a pretrained model and multiple independently fine-tuned checkpoints, produce a single set of model weights that can be evaluated on every task without training on the original labeled training sets.

## The paper's method

AdaRank replaces fixed top-k rank truncation in SVD-based model merging with adaptive binary selection of singular components. The method learns which singular components from each task vector and layer should be kept so that the merged model reduces inter-task interference while preserving task performance.

## Key results of the paper

- On the 8-task ViT-B/32 benchmark, Task Arithmetic improves from 69.2% to 87.9% when augmented with AdaRank, while CART improves from 84.7% to 89.2%.
- On ViT-B/32 across 8, 14, and 20 tasks, TSV-M+AdaRank reports 88.9%, 86.9%, and 86.9% average accuracy.
- Table 4 reports AdaRank learnable mask overhead of 294,912 parameters for ViT-B/32 and 786,432 for ViT-L/14, both 0.032% of the corresponding total parameter pool.

## Run it

```bash
pip install sciworld
sw research 2503.22178 --mode innovate --agent claude-code
sw evaluate 2503.22178
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).


# Overview

- **Paper ID:** 2503.22178
- **Title:** AdaRank: Adaptive Rank Pruning for Enhanced Model Merging
- **Domain:** Model Merging / Multi-Task Learning
- **TL;DR:** AdaRank improves multi-task model merging by learning which SVD components of task vectors to keep, rather than using a fixed top-k rank rule.

## Short Summary

The paper studies how to merge multiple independently fine-tuned models into a single model that performs well across all tasks. It argues that fixed top-k SVD truncation is suboptimal because high singular-value components can help one task while interfering with others, and because useful rank varies across tasks and layers. AdaRank addresses this by learning binary masks over singular components with unlabeled test-time entropy minimization. The reported results show higher average accuracy than static and adaptive merging baselines while keeping deployment to a single merged model.

## Key Results

- On the 8-task ViT-B/32 benchmark, Task Arithmetic improves from 69.2% to 87.9% when augmented with AdaRank, while CART improves from 84.7% to 89.2%.
- On ViT-B/32 across 8, 14, and 20 tasks, TSV-M+AdaRank reports 88.9%, 86.9%, and 86.9% average accuracy.
- Table 4 reports AdaRank learnable mask overhead of 294,912 parameters for ViT-B/32 and 786,432 for ViT-L/14, both 0.032% of the corresponding total parameter pool.

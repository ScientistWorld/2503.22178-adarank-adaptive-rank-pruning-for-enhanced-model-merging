# Problem Definition

## Research Question

Several image encoders were fine-tuned, each on a different classification task, starting from the same pretrained encoder. How can they be combined into **one** encoder that performs well on all of the tasks at once, without labels and without retraining on the original training sets?

## Setting

- **Backbone:** CLIP ViT-B/32 image encoder (pretrained `openai/clip-vit-base-patch32`).
- **Fine-tuned encoders (given, fixed):** one per task. All eight start from the same pretrained encoder:
  - SUN397 (scenes, 397 classes)
  - Stanford Cars (196)
  - RESISC45 (remote sensing, 45)
  - EuroSAT (satellite land use, 10)
  - SVHN (house-number digits, 10)
  - GTSRB (traffic signs, 43)
  - MNIST (handwritten digits, 10)
  - DTD (textures, 47)
- **Classification:** each task is read out by a fixed zero-shot head built from the pretrained CLIP text encoder with the task's class names. The head is applied to the L2-normalised projected image embedding with logit scale 100. The heads and the visual projection are part of the problem, not of the solution.
- **Data available to a solution:**
  - the pretrained and fine-tuned weights;
  - an unlabeled adaptation pool of test-split images per task (`data/downloads/prepared/adapt_<task>.pt`; 600–2,048 images per task);
  - a small labeled dev slice (256 per task) that exists only for development scoring.

  Utilities are in `data/clip_benchmark.py`.

## Why It Matters

Keeping one model per task multiplies storage and serving cost by the number of tasks. Naively adding the fine-tuning updates loses much of the experts' accuracy because the updates interfere with each other. A strong merging procedure recovers most of the experts' accuracy in a single model of the original size.

## Success Criteria

A good solution produces a single encoder of the original architecture and size that:

- reaches high average accuracy over the eight tasks on held-out images, approaching the individually fine-tuned encoders;
- does not sacrifice any task (worst-task accuracy) or any class within a task (balanced accuracy);
- uses no labels for building the merged encoder.

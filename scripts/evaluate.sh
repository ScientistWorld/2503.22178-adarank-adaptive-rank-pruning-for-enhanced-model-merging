#!/bin/bash
# Standard evaluation: scores every submission under outputs/merged/<name>/ and the fixed
# baselines on the public labeled dev rows; writes scoring/scores.json.
#
# Submission format: outputs/merged/<name>/model.safetensors holding exactly the tensors of
# one CLIP ViT-B/32 image encoder (transformers CLIPVisionModel names/shapes, finite floats).
# The fixed zero-shot heads and visual projection are attached by the evaluator.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p scoring
python eval/train/evaluate.py --output scoring/scores.json

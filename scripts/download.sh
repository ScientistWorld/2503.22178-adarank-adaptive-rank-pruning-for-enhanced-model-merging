#!/bin/bash
# Download all data and models needed for this environment (run on a node with internet).
#
# Estimated total download size: ~12 GB
#   - pretrained CLIP ViT-B/32 (openai/clip-vit-base-patch32, 0.6 GB)
#   - 8 task-specific fine-tuned CLIP ViT-B/32 image encoders (tanganke/clip-vit-base-patch32_<task>, 2.8 GB)
#   - public test-split parquet files of SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN,
#     GTSRB, MNIST and DTD (8.6 GB; only fetched while the prepared splits are missing)
# Estimated disk usage after preparation: ~4.5 GB (the raw parquet files are deleted by
#   `python data/prepare_data.py --scrub-raw` once the splits are built)
#
# All files come from pinned Hugging Face revisions and are checked against the sha256
# digests recorded in data/tasks.json. Idempotent: verified files are skipped.
#
# Building the splits needs pyarrow (installed in the container), so it runs at the
# start of the first job (scripts/run.sh / scripts/reproduce.sh):
#   python data/prepare_data.py            # public adaptation + dev splits
#   python eval/test/prepare_hidden.py     # held-out split (evaluation side only)
#   python data/prepare_data.py --scrub-raw

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/downloads
python3 data/fetch.py "$@"

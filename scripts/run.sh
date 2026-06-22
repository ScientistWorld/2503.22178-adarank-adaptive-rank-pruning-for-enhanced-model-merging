#!/bin/bash
# Run the train/test scoring split against the frozen final prediction artifact.

set -e

cd /home/user

bash scripts/evaluate_train.sh outputs/adarank_synthetic/predictions.json
bash scripts/evaluate_test.sh outputs/adarank_synthetic/predictions.json

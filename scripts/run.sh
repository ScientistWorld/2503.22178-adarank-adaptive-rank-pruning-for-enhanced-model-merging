#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

bash scripts/reproduce.sh --learn-lambda --lambda-lr 0.0002 --max-lambda 0.45 --sweep-adapt-steps 80,160,320,500
python validate.py --compare

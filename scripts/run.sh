#!/bin/bash
# Final standardized reproduction job.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

bash scripts/reproduce.sh
python validate.py --compare

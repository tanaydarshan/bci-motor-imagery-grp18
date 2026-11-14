#!/bin/bash
# Resume the checkpointed 12- and 6-class LOSO reruns after a container restart.
cd "$(dirname "$0")/.."
PY=/tmp/claude-0/venv/bin/python
export FEATURE_CACHE=feature_cache_fresh
$PY -u loso/loso_models.py $1 --tag final --jobs 2 >> loso/results/log_final_$1.txt 2>&1

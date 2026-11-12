#!/bin/bash
# Body-part (6-class) fresh LOSO run, started after the GroupKFold verification so the 4 cores are not oversubscribed.
cd "$(dirname "$0")/.."
PY=/tmp/claude-0/venv/bin/python
export FEATURE_CACHE=feature_cache_fresh
until grep -q VERIFY_DONE loso/results/log_paper_check.txt 2>/dev/null; do sleep 120; done
$PY -u loso/loso_models.py 6 --tag final --jobs 2 >> loso/results/log_final_6.txt 2>&1
echo C_DONE >> loso/results/log_final_runner.txt

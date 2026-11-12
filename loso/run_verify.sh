#!/bin/bash
cd "$(dirname "$0")/.."
PY=/tmp/claude-0/venv/bin/python
export FEATURE_CACHE=feature_cache_fresh
until [ "$(ls feature_cache_fresh/*.npy 2>/dev/null | wc -l)" -ge 4 ]; do sleep 60; done
sleep 120
for p in main families onset; do nice -n 5 $PY -u loso/verify_paper_numbers.py $p >> loso/results/log_paper_check.txt 2>&1; done
echo VERIFY_DONE >> loso/results/log_paper_check.txt

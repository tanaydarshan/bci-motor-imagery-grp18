#!/bin/bash
cd "$(dirname "$0")/.."
PY=/tmp/claude-0/venv/bin/python
$PY -u loso/loso_eval.py LR --perm 1 --jobs 4 >> loso/results/log_LR_perm.txt 2>&1
$PY -u loso/loso_eval.py bin --perm 1 --jobs 4 >> loso/results/log_bin_perm.txt 2>&1

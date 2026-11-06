#!/bin/bash
cd "$(dirname "$0")/.."
PY=/tmp/claude-0/venv/bin/python
L=loso/results
$PY -u loso/loso_models.py LR --extra --tag extra --jobs 1 >> $L/log_extra_LR.txt 2>&1
$PY -u loso/loso_models.py bin --extra --tag extra --jobs 1 >> $L/log_extra_bin.txt 2>&1
echo EXTRA_DONE >> $L/log_runner.txt

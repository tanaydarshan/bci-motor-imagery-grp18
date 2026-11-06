#!/bin/bash
cd "$(dirname "$0")/.."
PY=/tmp/claude-0/venv/bin/python
while pgrep -f run_models.sh > /dev/null; do sleep 60; done
$PY -u loso/learning_curve.py bin >> loso/results/log_learning_curve.txt 2>&1
$PY -u loso/loso_models.py 6 --tag run1 >> loso/results/log_models_6.txt 2>&1
$PY -u loso/loso_models.py 12 --tag run1 >> loso/results/log_models_12.txt 2>&1
echo STAGE2_DONE >> loso/results/log_models_runner.txt

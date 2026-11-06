#!/bin/bash
cd "$(dirname "$0")/.."
PY=/tmp/claude-0/venv/bin/python
L=loso/results
$PY -u loso/loso_models.py LR --tag run1 >> $L/log_models_LR.txt 2>&1
$PY -u loso/loso_models.py bin --tag run1 >> $L/log_models_bin.txt 2>&1
$PY -u loso/loso_models.py bin --tag run2 >> $L/log_models_bin_run2.txt 2>&1
$PY -u loso/learning_curve.py bin >> $L/log_learning_curve.txt 2>&1
$PY -u loso/loso_models.py 6 --tag run1 >> $L/log_models_6.txt 2>&1
$PY -u loso/loso_models.py 12 --tag run1 >> $L/log_models_12.txt 2>&1
echo MAIN_DONE >> $L/log_runner.txt

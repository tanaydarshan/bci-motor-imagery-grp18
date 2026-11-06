#!/bin/bash
# Sequential model-by-model LOSO runs (each resumes from its checkpoint).
cd "$(dirname "$0")/.."
PY=/tmp/claude-0/venv/bin/python
$PY -u loso/loso_models.py LR --extra --tag run1 >> loso/results/log_models_LR.txt 2>&1
$PY -u loso/loso_models.py bin --tag run1 >> loso/results/log_models_bin.txt 2>&1
$PY -u loso/loso_models.py bin --tag run2 >> loso/results/log_models_bin_run2.txt 2>&1
echo STAGE1_DONE >> loso/results/log_models_runner.txt

#!/bin/bash
# Runs the LOSO jobs one after another (each resumes from its checkpoint if restarted).
cd "$(dirname "$0")/.."
PY=/tmp/claude-0/venv/bin/python
while pgrep -f "loso_eval.py (bin|LR)$" > /dev/null; do sleep 60; done
for t in 6 12; do [ -f loso/results/loso_$t.json ] || $PY -u loso/loso_eval.py $t --jobs 4 >> loso/results/log_$t.txt 2>&1; done
for t in bin LR 6 12; do [ -f loso/results/loso_${t}_perm.json ] || $PY -u loso/loso_eval.py $t --perm 1 --jobs 4 >> loso/results/log_${t}_perm.txt 2>&1; done
echo ALL_DONE >> loso/results/log_runner.txt

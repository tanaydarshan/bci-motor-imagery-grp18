#!/bin/bash
# Fresh final LOSO run: features recomputed from raw EDF into an empty cache, every task, checkpointed.
cd "$(dirname "$0")/.."
PY=/tmp/claude-0/venv/bin/python
L=loso/results
export FEATURE_CACHE=feature_cache_fresh
if [ "$1" = "A" ]; then
  for t in bin LR 12; do $PY -u loso/loso_models.py $t --tag final --jobs 2 >> $L/log_final_$t.txt 2>&1; done
  echo A_DONE >> $L/log_final_runner.txt
else
  until [ "$(ls feature_cache_fresh/*.npy 2>/dev/null | wc -l)" -ge 4 ]; do sleep 60; done
  sleep 120
  $PY -u loso/loso_models.py 6 --tag final --jobs 2 >> $L/log_final_6.txt 2>&1
  echo B_DONE >> $L/log_final_runner.txt
fi

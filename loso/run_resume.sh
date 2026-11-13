#!/bin/bash
# Resume after a container restart: 12-class LOSO (checkpointed) alongside the remaining GroupKFold checks,
# then the 6-class LOSO once the checks are done.
cd "$(dirname "$0")/.."
PY=/tmp/claude-0/venv/bin/python
export FEATURE_CACHE=feature_cache_fresh
L=loso/results
if [ "$1" = "A" ]; then
  $PY -u loso/loso_models.py 12 --tag final --jobs 2 >> $L/log_final_12.txt 2>&1
  echo A12_DONE >> $L/log_final_runner.txt
else
  grep -q "DONE families" $L/log_paper_check.txt || nice -n 5 $PY -u loso/verify_paper_numbers.py families >> $L/log_paper_check.txt 2>&1
  nice -n 5 $PY -u loso/verify_paper_numbers.py onset >> $L/log_paper_check.txt 2>&1
  echo VERIFY_DONE >> $L/log_paper_check.txt
  $PY -u loso/loso_models.py 6 --tag final --jobs 2 >> $L/log_final_6.txt 2>&1
  echo B6_DONE >> $L/log_final_runner.txt
fi

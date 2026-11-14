"""Compare the fresh LOSO rerun (tag 'final', features recomputed from raw EDF) with the first run (tag 'run1').
For each finished task it reports: per-model accuracy, trial-by-trial agreement with run1, the train/test gap,
and whether test accuracy drifts with fold order (Spearman correlation of per-person accuracy vs fold index).
Usage: python loso/compare_final.py
"""
import json, os
import numpy as np
from scipy.stats import spearmanr, binomtest

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
chance = {'bin': 0.5, '6': 1 / 6, '12': 1 / 12, 'LR': 0.5}
out = {}
for t in ['bin', 'LR', '6', '12']:
    f = os.path.join(R, f'models_{t}_final.json')
    if not os.path.exists(f): continue
    A = json.load(open(f)); B = json.load(open(os.path.join(R, f'models_{t}_run1.json')))
    people = sorted(A, key=int)
    print(f'=== {t}: {len(people)}/47 people done ===')
    out[t] = {'people_done': len(people)}
    for m in A[people[0]]['pred']:
        y = np.concatenate([[c for _, c in A[p]['trials']] for p in people])
        pa = np.concatenate([A[p]['pred'][m] for p in people])
        pb = np.concatenate([B[p]['pred'][m] for p in people if p in B])
        same = float((pa == pb).mean()) if len(pa) == len(pb) else float('nan')
        acc = float((pa == y).mean())
        per = [np.mean(np.array(A[p]['pred'][m]) == np.array([c for _, c in A[p]['trials']])) for p in people]
        rho, prho = spearmanr(np.arange(len(per)), per)
        tr = np.mean([A[p]['train_trial_acc'][m] for p in people])
        pv = binomtest(int((pa == y).sum()), len(y), chance[t], alternative='greater').pvalue
        out[t][m] = dict(correct=int((pa == y).sum()), n=len(y), acc=acc, same_as_run1=same, train_acc=float(tr),
                         order_rho=float(rho), order_p=float(prho), p_vs_chance=float(pv))
        print(f'  {m:<11} {int((pa == y).sum()):4d}/{len(y)} = {100 * acc:5.2f}%  same as run1: {100 * same:5.1f}%  '
              f'train {100 * tr:5.1f}%  order rho = {rho:+.2f} (p = {prho:.2f})  p(chance) = {pv:.1e}')
json.dump(out, open(os.path.join(R, 'compare_final.json'), 'w'), indent=1)

"""Summarise the model-by-model LOSO results: accuracy, balanced accuracy, macro F1, Cohen's kappa,
binomial p, per-person mean ± SD, window accuracy, training accuracy (overfitting gap), and run-to-run
reproducibility (run1 vs run2)."""
import json, os, sys
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
summary = {}
for task, chance in [('bin', 1 / 2), ('6', 1 / 6), ('12', 1 / 12), ('LR', 1 / 2)]:
    f = os.path.join(OUT, f'models_{task}_run1.json')
    if not os.path.exists(f): continue
    R = json.load(open(f))
    fx = os.path.join(OUT, f'models_{task}_extra.json')
    if os.path.exists(fx):   # comparison models run separately; merge people present in both
        RX = json.load(open(fx)); R = {p: r for p, r in R.items() if p in RX}
        for p in R:
            assert R[p]['trials'] == RX[p]['trials']
            for k in ['pred', 'win_test', 'train_trial_acc', 'train_win_acc']: R[p][k].update(RX[p][k])
    models = list(next(iter(R.values()))['pred'])
    print(f'\n=== Task {task}: LOSO over {len(R)} people (chance {100 * chance:.2f}%) ===')
    print(f'{"Model":<13}{"Correct":>9}{"Acc":>8}{"BalAcc":>8}{"MacroF1":>8}{"Kappa":>7}{"p":>9}'
          f'{"Person mean±SD":>17}{"Window":>8}{"TrainTrial":>11}{"TrainWin":>9}')
    summary[task] = {}
    for m in models:
        yt = np.concatenate([[a for _, a in r['trials']] for r in R.values()])
        yp = np.concatenate([r['pred'][m] for r in R.values()])
        per = np.array([np.mean(np.array(r['pred'][m]) == np.array([a for _, a in r['trials']])) for r in R.values()])
        c = int((yt == yp).sum()); n = len(yt)
        win = sum(r['win_test'][m][0] for r in R.values()) / sum(r['win_test'][m][1] for r in R.values())
        trt = np.mean([r['train_trial_acc'][m] for r in R.values()]); trw = np.mean([r['train_win_acc'][m] for r in R.values()])
        s = dict(correct=c, n=n, acc=c / n, balanced_acc=balanced_accuracy_score(yt, yp),
                 macro_precision=precision_score(yt, yp, average='macro', zero_division=0),
                 macro_recall=recall_score(yt, yp, average='macro', zero_division=0),
                 macro_f1=f1_score(yt, yp, average='macro'), kappa=cohen_kappa_score(yt, yp),
                 p=binomtest(c, n, chance, alternative='greater').pvalue, person_mean=per.mean(), person_sd=per.std(),
                 window_acc=win, train_trial_acc=trt, train_window_acc=trw)
        summary[task][m] = s
        print(f'{m:<13}{c:>5}/{n:<3}{100 * s["acc"]:>8.2f}{100 * s["balanced_acc"]:>8.2f}{100 * s["macro_f1"]:>8.2f}'
              f'{s["kappa"]:>7.3f}{s["p"]:>9.1e}{100 * per.mean():>10.2f} ± {100 * per.std():<5.2f}{100 * win:>8.2f}'
              f'{100 * trt:>11.2f}{100 * trw:>9.2f}')
    f2 = os.path.join(OUT, f'models_{task}_run2.json')
    if os.path.exists(f2):
        R2 = json.load(open(f2)); common = sorted(set(R) & set(R2))
        same = all(R[p]['pred'][m] == R2[p]['pred'][m] for p in common for m in models)
        c1 = sum(sum(np.array(R[p]['pred']['Ensemble']) == np.array([a for _, a in R[p]['trials']])) for p in common)
        c2 = sum(sum(np.array(R2[p]['pred']['Ensemble']) == np.array([a for _, a in R2[p]['trials']])) for p in common)
        print(f'Rerun check ({len(common)} people): ensemble {c1} vs {c2} correct; all per-trial predictions identical: {same}')
        summary[task]['rerun_identical'] = bool(same)
json.dump(summary, open(os.path.join(OUT, 'summary_models.json'), 'w'), indent=1, default=float)

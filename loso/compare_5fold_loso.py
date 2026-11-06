"""Trial-by-trial comparison of the ensemble under 5-fold (3-fold for 6/12-class) and LOSO, for all four tasks."""
import io, contextlib, json, os
import numpy as np
from scipy.stats import binomtest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LNS = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(REPO, 'v11_results', 'loader.py')).read(), LNS)
G = LNS['G']; sw, tw = G['sw'], G['trial_of_win']
from sklearn.model_selection import GroupKFold

P5 = json.load(open(os.path.join(REPO, 'v10_results', 'predictions.json')))
key5 = {'bin': 'bin', '6': '6', '12': '12', 'LR': 'lr'}
report = {}
for task in ['bin', '6', '12', 'LR']:
    T = G['TASKS'][task]; idx = np.where(T['mask'])[0]; y = T['y']
    order, fold_of = [], {}
    for f, (tr_l, te_l) in enumerate(GroupKFold(n_splits=T['k']).split(idx, y[idx], groups=sw[idx])):
        for t in np.unique(tw[idx[te_l]]): order.append(int(t)); fold_of[int(t)] = f
    p5 = dict(zip(order, P5[key5[task]]['y_pred']))
    truth = {int(t): int(y[tw == t][0]) for t in order}
    assert all(truth[t] == a for t, a in zip(order, P5[key5[task]]['y_true']))
    L = json.load(open(os.path.join(REPO, 'loso', 'results', f'models_{task}_run1.json')))
    pL = {t: v for r in L.values() for (t, _), v in zip(r['trials'], r['pred']['Ensemble'])}
    ts = sorted(truth)
    c5 = np.array([p5[t] == truth[t] for t in ts]); cL = np.array([pL[t] == truth[t] for t in ts])
    same = np.array([p5[t] == pL[t] for t in ts])
    b, c = int((c5 & ~cL).sum()), int((~c5 & cL).sum())
    mcn = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    # people whose result changed
    person = {t: int(sw[tw == t][0]) for t in ts}
    changed = {}
    for t in ts:
        if c5[ts.index(t)] != cL[ts.index(t)]:
            changed.setdefault(person[t], [0, 0])[0 if c5[ts.index(t)] else 1] += 1
    print(f'\n=== {task} ({len(ts)} trials) ===')
    print(f'  5-fold correct {c5.sum()} ({100 * c5.mean():.2f}%)   LOSO correct {cL.sum()} ({100 * cL.mean():.2f}%)   difference {int(cL.sum() - c5.sum()):+d}')
    print(f'  same predicted label on {same.sum()}/{len(ts)} trials ({100 * same.mean():.1f}%); different on {int((~same).sum())}')
    print(f'  right only in 5-fold: {b}   right only in LOSO: {c}   (McNemar p = {mcn:.2f})')
    print(f'  people whose score changed: {len(changed)} of 47; '
          f'per person [lost in LOSO, gained in LOSO]: {dict(sorted(changed.items()))}')
    report[task] = dict(correct_5fold=int(c5.sum()), correct_loso=int(cL.sum()), same=int(same.sum()), n=len(ts),
                        only_5fold=b, only_loso=c, mcnemar_p=mcn, people_changed=len(changed))
json.dump(report, open(os.path.join(REPO, 'loso', 'results', 'compare_5fold_loso.json'), 'w'), indent=1)

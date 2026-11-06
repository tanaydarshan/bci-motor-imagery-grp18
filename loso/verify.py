"""Independent checks of the evaluation code (no model training).

1. Rebuilds the 5-fold person-wise split, attaches trial ids to the saved 5-fold predictions and compares them,
   trial by trial, with the LOSO predictions (are equal totals a coincidence or the same predictions?).
2. Audits every fold of both schemes for overlap of people, recordings and trials.
3. Checks the mirror label maps and the label-free normalisation steps.
"""
import io, contextlib, json, os
import numpy as np
from scipy.stats import binomtest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LNS = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(REPO, 'v11_results', 'loader.py')).read(), LNS)
G = LNS['G']
sw, sess_w, tw = G['sw'], G['sess_w'], G['trial_of_win']
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

report = {}
P5 = json.load(open(os.path.join(REPO, 'v10_results', 'predictions.json')))
key5 = {'bin': 'bin', '6': '6', '12': '12', 'LR': 'lr'}

print('=== 1. Fold audit (people / recordings / trials never on both sides) ===')
for task, T in G['TASKS'].items():
    idx = np.where(T['mask'])[0]; y = T['y']
    for name, splitter in [('5/3-fold', GroupKFold(n_splits=T['k'])), ('LOSO', LeaveOneGroupOut())]:
        n_f = 0; tested = []
        for tr_l, te_l in splitter.split(idx, y[idx], groups=sw[idx]):
            tr, te = idx[tr_l], idx[te_l]; n_f += 1
            assert not set(sw[tr]) & set(sw[te]), 'person overlap'
            assert not set(sess_w[tr]) & set(sess_w[te]), 'recording overlap'
            assert not set(tw[tr]) & set(tw[te]), 'trial overlap'
            tested += list(np.unique(tw[te]))
        assert len(tested) == len(set(tested)) == len(np.unique(tw[idx])), 'every trial tested exactly once'
        print(f'  {task:>3} {name:<8}: {n_f:2d} folds, {len(tested)} trials each tested once, no overlap')

print('\n=== 2. Mirror label maps (training-only augmentation) ===')
names6 = ['RA', 'LA', 'RL', 'LL', 'RW', 'LW']
mb = G['mirror_label']['body'](np.arange(6)); print('  body :', {names6[i]: names6[j] for i, j in enumerate(mb)})
mm = G['mirror_label']['multi'](np.arange(12))
print('  12   :', {f'{names6[i // 2]}-{"AI"[i % 2]}': f'{names6[j // 2]}-{"AI"[j % 2]}' for i, j in enumerate(mm)})
print('  bin  :', dict(zip([0, 1], G['mirror_label']['bin'](np.arange(2)))), ' side:', dict(zip([0, 1], G['mirror_label']['side'](np.arange(2)))))
assert list(mb) == [1, 0, 3, 2, 5, 4] and list(G['mirror_label']['bin'](np.arange(2))) == [0, 1]

print('\n=== 3. Label-free normalisation: per-recording feature means are ~0 whatever the labels ===')
Xf = G['X_final']
dev = max(np.abs(Xf[sess_w == r, :1666].mean(0)).max() for r in np.unique(sess_w))
print(f'  max |mean| of aligned features within any recording: {dev:.2e} (alignment uses recording id only)')

print('\n=== 4. LOSO vs 5-fold: same predictions or a coincidence? ===')
for task in ['bin', '6', 'LR', '12']:
    fl = os.path.join(REPO, 'loso', 'results', f'loso_{task}.json')
    if not os.path.exists(fl): continue
    T = G['TASKS'][task]; idx = np.where(T['mask'])[0]; y = T['y']
    order = []
    for tr_l, te_l in GroupKFold(n_splits=T['k']).split(idx, y[idx], groups=sw[idx]):
        order += list(np.unique(tw[idx[te_l]]))
    p5 = P5[key5[task]]
    assert len(order) == len(p5['y_true'])
    ytrue5 = {t: a for t, a in zip(order, p5['y_true'])}
    assert all(ytrue5[t] == y[tw == t][0] for t in order), 'trial id mapping check failed'
    pred5 = {t: b for t, b in zip(order, p5['y_pred'])}
    L = json.load(open(fl))
    predL = {t: p for t, _, _, p in L['trials']}; trueL = {t: a for t, _, a, _ in L['trials']}
    ts = sorted(predL)
    c5 = np.array([pred5[t] == trueL[t] for t in ts]); cL = np.array([predL[t] == trueL[t] for t in ts])
    same_pred = np.mean([pred5[t] == predL[t] for t in ts])
    b, c = int((c5 & ~cL).sum()), int((~c5 & cL).sum())
    p_mcn = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    print(f'  {task:>3}: 5-fold {c5.sum()}/{len(ts)}, LOSO {cL.sum()}/{len(ts)} | identical prediction on {100 * same_pred:.1f}% of trials | '
          f'correct only in 5-fold: {b}, only in LOSO: {c} (McNemar p = {p_mcn:.2f})')
    report[task] = dict(fivefold=int(c5.sum()), loso=int(cL.sum()), same_pred=float(same_pred), only5=b, onlyLOSO=c, mcnemar_p=p_mcn)
json.dump(report, open(os.path.join(REPO, 'loso', 'results', 'verify_loso_vs_5fold.json'), 'w'), indent=1)
print('DONE')

"""Learning curve: does test accuracy grow plausibly with the number of training people, and how far is
training accuracy above test accuracy (overfitting)?

The test people are fixed (person-wise 5-fold split). For each fold, the model is trained on a random subset
of n of the remaining people (n = 5, 10, 20, 30, all), with the fixed v10 ensemble and mirror augmentation.
Usage: python loso/learning_curve.py [task] [--reps 2]
"""
import argparse, io, contextlib, json, os, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('task', nargs='?', default='bin'); ap.add_argument('--reps', type=int, default=2)
args = ap.parse_args()
LNS = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(REPO, 'v11_results', 'loader.py')).read(), LNS)
G = LNS['G']; sw, tw = G['sw'], G['trial_of_win']; T = G['TASKS'][args.task]
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

MODELS = [lambda: LogisticRegression(C=0.01, max_iter=3000),
          lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
          lambda: ExtraTreesClassifier(n_estimators=500, max_features='sqrt', n_jobs=4, random_state=42),
          lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=42)]
X, Xm, ymap, y = T['X'], T['Xm'], T['ymap'], T['y']
idx = np.where(T['mask'])[0]


def trial_acc(Pr, cls, rows):
    t = tw[rows]; ut = np.unique(t)
    yp = cls[np.array([Pr[t == u].mean(0) for u in ut]).argmax(1)]
    return int((yp == np.array([y[tw == u][0] for u in ut])).sum()), len(ut)


rng = np.random.default_rng(0)
out = {}
for n_people in [5, 10, 20, 30, 'all']:
    reps = 1 if n_people == 'all' else args.reps
    c_te = n_te = 0; tr_accs = []
    t0 = time.time()
    for rep in range(reps):
        for tr_l, te_l in GroupKFold(n_splits=5).split(idx, y[idx], groups=sw[idx]):
            tr_all, te = idx[tr_l], idx[te_l]
            people = np.unique(sw[tr_all])
            chosen = people if n_people == 'all' else rng.choice(people, n_people, replace=False)
            tr = tr_all[np.isin(sw[tr_all], chosen)]
            assert not set(sw[tr]) & set(sw[te])
            A = np.vstack([X[tr], Xm[tr]]); yA = np.concatenate([y[tr], ymap(y[tr])])
            sc = StandardScaler().fit(A); As, Bs, Rs = sc.transform(A), sc.transform(X[te]), sc.transform(X[tr])
            fits = [m().fit(As, yA) for m in MODELS]; cls = fits[0].classes_
            Pt = np.mean([f.predict_proba(Bs) for f in fits], 0); Pr = np.mean([f.predict_proba(Rs) for f in fits], 0)
            c, n = trial_acc(Pt, cls, te); c_te += c; n_te += n
            ct, nt = trial_acc(Pr, cls, tr); tr_accs.append(ct / nt)
    out[str(n_people)] = {'test_acc': c_te / n_te, 'train_acc': float(np.mean(tr_accs)), 'reps': reps}
    print(f'train people {n_people!s:>4}: test {100 * c_te / n_te:.2f}%  train {100 * np.mean(tr_accs):.2f}%  ({time.time() - t0:.0f}s)', flush=True)
    json.dump(out, open(os.path.join(REPO, 'loso', 'results', f'learning_curve_{args.task}.json'), 'w'), indent=1)
print('DONE')

"""Null distribution of the full pipeline under shuffled labels (person-wise 5-fold, ensemble + mirror + trial averaging).

Labels are shuffled across trials within each recording (all windows of a trial move together), as in loso_eval.py.
Usage: python loso/perm_null.py <task> [--n 20]
"""
import argparse, io, contextlib, json, os, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('task'); ap.add_argument('--n', type=int, default=20)
args = ap.parse_args()
LNS = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(REPO, 'v11_results', 'loader.py')).read(), LNS)
G = LNS['G']; sw, tw, sess_w = G['sw'], G['trial_of_win'], G['sess_w']; T = G['TASKS'][args.task]
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

MODELS = [lambda: LogisticRegression(C=0.01, max_iter=3000),
          lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
          lambda: ExtraTreesClassifier(n_estimators=500, max_features='sqrt', n_jobs=4, random_state=42),
          lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=42)]
X, Xm, ymap = T['X'], T['Xm'], T['ymap']
idx = np.where(T['mask'])[0]


def permuted(y, rng):
    yp = y.copy()
    lab = {t: y[tw == t][0] for t in np.unique(tw[idx])}
    for r in np.unique(sess_w[idx]):
        ts = np.unique(tw[idx][sess_w[idx] == r]); l = np.array([lab[t] for t in ts]); rng.shuffle(l)
        for t, v in zip(ts, l): yp[tw == t] = v
    return yp


def cv(y):
    c = n = 0
    for tr_l, te_l in GroupKFold(n_splits=T['k']).split(idx, y[idx], groups=sw[idx]):
        tr, te = idx[tr_l], idx[te_l]
        A = np.vstack([X[tr], Xm[tr]]); yA = np.concatenate([y[tr], ymap(y[tr])])
        sc = StandardScaler().fit(A); As, Bs = sc.transform(A), sc.transform(X[te])
        fits = [m().fit(As, yA) for m in MODELS]; cls = fits[0].classes_
        P = np.mean([f.predict_proba(Bs) for f in fits], 0)
        t = tw[te]; ut = np.unique(t)
        yp = cls[np.array([P[t == u].mean(0) for u in ut]).argmax(1)]
        c += int((yp == np.array([y[tw == u][0] for u in ut])).sum()); n += len(ut)
    return c / n


out_f = os.path.join(REPO, 'loso', 'results', f'perm_null_{args.task}.json')
accs = json.load(open(out_f))['shuffled_acc'] if os.path.exists(out_f) else []
rng = np.random.default_rng(123)
for i in range(args.n):
    yp = permuted(T['y'], rng)
    if i < len(accs): continue          # resume: same draws, skip runs already done
    t0 = time.time(); accs.append(cv(yp))
    json.dump({'task': args.task, 'shuffled_acc': accs}, open(out_f, 'w'), indent=1)
    print(f'run {i + 1}/{args.n}: {100 * accs[-1]:.2f}%  ({time.time() - t0:.0f}s)', flush=True)
a = np.array(accs)
print(f'Null: mean {100 * a.mean():.2f}%, SD {100 * a.std():.2f}%, min {100 * a.min():.2f}%, max {100 * a.max():.2f}%')
print('DONE')

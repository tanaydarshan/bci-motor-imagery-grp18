"""Leave-one-subject-out (LOSO) evaluation of the final v10 pipeline.

Each of the 47 participants is held out once; the model is trained on the other 46.
Leakage controls:
  * groups = person id (not recording), so people with two sessions are never split;
  * only label-free steps (filtering, rejection, per-recording z-scoring, feature alignment,
    covariance re-centring) are computed before the split, exactly as in the notebook;
  * scaler, classifiers and mirror augmentation use the training people only;
  * the configuration is the fixed v10 one; nothing is tuned on the LOSO results.

Usage:  python loso/loso_eval.py <task> [--perm N]      task in {bin, 6, 12, LR}
"""
import argparse, io, contextlib, json, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, 'loso', 'results')

ap = argparse.ArgumentParser()
ap.add_argument('task', choices=['bin', '6', '12', 'LR'])
ap.add_argument('--perm', type=int, default=0, help='number of shuffled-label runs (0 = real labels)')
ap.add_argument('--jobs', type=int, default=2, help='CPU cores for Extra Trees')
args = ap.parse_args()

# Load the exact notebook pipeline (data, onsets, cleaning, features, alignment, mirror) without running models.
LNS = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(REPO, 'v11_results', 'loader.py')).read(), LNS)
G = LNS['G']
np_, sw, trial_of_win, sess_w = np, G['sw'], G['trial_of_win'], G['sess_w']
T = G['TASKS'][args.task]

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from scipy.stats import binomtest

RS = 42
MODELS = {   # identical to the notebook's ENS_MODELS
    'LogReg': lambda: LogisticRegression(C=0.01, max_iter=3000),
    'LDA': lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    'ExtraTrees': lambda: ExtraTreesClassifier(n_estimators=500, max_features='sqrt', n_jobs=args.jobs, random_state=RS),
    'HistGB': lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=RS),
}


def permuted(y, mask, rng):
    """Shuffle trial labels across trials within each recording (all windows of a trial move together)."""
    yp = y.copy()
    trials = np.unique(trial_of_win[mask])
    lab = {t: y[trial_of_win == t][0] for t in trials}
    for r in np.unique(sess_w[mask]):
        ts = np.unique(trial_of_win[(sess_w == r) & mask]); l = np.array([lab[t] for t in ts]); rng.shuffle(l)
        for t, v in zip(ts, l): yp[trial_of_win == t] = v
    return yp


def run(y, ckpt):
    """LOSO over all people; per-person results are saved to `ckpt` after every fold so a restart resumes."""
    X, Xm, ymap = T['X'], T['Xm'], T['ymap']
    idx = np.where(T['mask'])[0]
    names = list(MODELS) + ['Ensemble']
    done = json.load(open(ckpt)) if os.path.exists(ckpt) else {}
    folds = list(LeaveOneGroupOut().split(idx, y[idx], groups=sw[idx]))
    assert len(folds) == len(np.unique(sw[idx]))
    for k, (tr_l, te_l) in enumerate(folds):
        tr, te = idx[tr_l], idx[te_l]
        # leakage checks: one test person, never in training; whole trials on one side only
        assert len(np.unique(sw[te])) == 1 and not set(sw[tr]) & set(sw[te])
        assert not set(trial_of_win[tr]) & set(trial_of_win[te])
        person = int(sw[te][0])
        if str(person) in done: continue
        t0 = time.time()
        A = np.vstack([X[tr], Xm[tr]]); yA = np.concatenate([y[tr], ymap(y[tr])])   # mirror: training only
        sc = StandardScaler().fit(A); A, B = sc.transform(A), sc.transform(X[te])
        P = {}
        for name, mk in MODELS.items():
            clf = mk().fit(A, yA); P[name] = clf.predict_proba(B); cls = clf.classes_
        P['Ensemble'] = np.mean([P[m] for m in MODELS], 0)
        tt = trial_of_win[te]; ut = np.unique(tt)
        yt = np.array([y[te][tt == t][0] for t in ut])
        rec = {}
        for name, Pr in P.items():
            yp = cls[np.array([Pr[tt == t].mean(0) for t in ut]).argmax(1)]
            rec[name] = [int((yp == yt).sum()), len(ut), int((cls[Pr.argmax(1)] == y[te]).sum()), len(te)]
            if name == 'Ensemble':
                rec['trials'] = [[int(t), person, int(a), int(b)] for t, a, b in zip(ut, yt, yp)]
        done[str(person)] = rec
        json.dump(done, open(ckpt, 'w'))
        c = sum(r['Ensemble'][0] for r in done.values()); n = sum(r['Ensemble'][1] for r in done.values())
        print(f'  person {len(done):2d}/{len(folds)}: {rec["Ensemble"][0]}/{len(ut)} correct '
              f'(running {c}/{n} = {100 * c / n:.2f}%) {time.time() - t0:.0f}s', flush=True)
    out = {m: {'correct': sum(r[m][0] for r in done.values()), 'n': sum(r[m][1] for r in done.values()),
               'win_correct': sum(r[m][2] for r in done.values()), 'win_n': sum(r[m][3] for r in done.values())} for m in names}
    per_person = {int(p): r['Ensemble'][:2] for p, r in done.items()}
    trials_out = sorted(t for r in done.values() for t in r['trials'])
    return out, per_person, trials_out


os.makedirs(OUT_DIR, exist_ok=True)
y = T['y']
if args.perm == 0:
    print(f'=== LOSO, task {args.task}, real labels ===', flush=True)
    out, per_person, trials = run(y, os.path.join(OUT_DIR, f'ckpt_{args.task}.json'))
    n_classes = len(np.unique(y[T['mask']]))
    res = {'task': args.task, 'n_people': len(per_person), 'chance': 1 / n_classes, 'models': {}, 'per_person': per_person,
           'trials': trials, 'trial_columns': ['trial', 'person', 'y_true', 'y_pred']}
    for m, r in out.items():
        res['models'][m] = {'correct': r['correct'], 'n': r['n'], 'acc': r['correct'] / r['n'],
                            'window_acc': r['win_correct'] / r['win_n'],
                            'p_binomial': binomtest(r['correct'], r['n'], 1 / n_classes, alternative='greater').pvalue}
    accs = np.array([c / n for c, n in per_person.values()])
    res['per_person_mean'], res['per_person_sd'] = float(accs.mean()), float(accs.std())
    json.dump(res, open(os.path.join(OUT_DIR, f'loso_{args.task}.json'), 'w'), indent=1)
    for m, r in res['models'].items():
        print(f'{m:>11}: {r["correct"]}/{r["n"]} = {100 * r["acc"]:.2f}%  window {100 * r["window_acc"]:.2f}%  p = {r["p_binomial"]:.2g}')
    print(f'Per-person accuracy: {100 * accs.mean():.2f} ± {100 * accs.std():.2f}%')
else:
    rng = np.random.default_rng(7)
    accs = []
    for i in range(args.perm):
        print(f'=== LOSO, task {args.task}, shuffled labels run {i + 1} ===', flush=True)
        yp = permuted(y, T['mask'], rng)   # drawn in order, so a resumed run reuses the same shuffles
        out, _, _ = run(yp, os.path.join(OUT_DIR, f'ckpt_{args.task}_perm{i + 1}.json'))
        accs.append(out['Ensemble']['correct'] / out['Ensemble']['n'])
        json.dump({'task': args.task, 'shuffled_acc': accs}, open(os.path.join(OUT_DIR, f'loso_{args.task}_perm.json'), 'w'), indent=1)
    print('Shuffled-label accuracy:', [round(100 * a, 2) for a in accs])
print('DONE', flush=True)

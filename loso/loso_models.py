"""Model-by-model LOSO evaluation with full metrics, training accuracy and per-trial predictions.

Same leakage controls as loso_eval.py (person-wise LeaveOneGroupOut, scaler/models/mirroring fitted on
the 46 training people only, fixed v10 settings). For every model it stores the trial-level prediction of
every test trial and the accuracy on the training people, so overfitting (train >> test) is visible.
Results are checkpointed after every person.

Usage: python loso/loso_models.py <task> [--tag run1] [--extra] [--jobs 4]
"""
import argparse, io, contextlib, json, os, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'loso', 'results')
ap = argparse.ArgumentParser()
ap.add_argument('task', choices=['bin', '6', '12', 'LR'])
ap.add_argument('--tag', default='run1')
ap.add_argument('--extra', action='store_true', help='run only the comparison models SVM, Random Forest, AdaBoost and Naive Bayes')
ap.add_argument('--jobs', type=int, default=4)
args = ap.parse_args()

LNS = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(REPO, 'v11_results', 'loader.py')).read(), LNS)
G = LNS['G']; sw, tw = G['sw'], G['trial_of_win']; T = G['TASKS'][args.task]

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

RS = 42
MODELS = {   # the four ensemble members, exactly as in the notebook
    'LogReg': lambda: LogisticRegression(C=0.01, max_iter=3000),
    'LDA': lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    'ExtraTrees': lambda: ExtraTreesClassifier(n_estimators=500, max_features='sqrt', n_jobs=args.jobs, random_state=RS),
    'HistGB': lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=RS),
}
ENS = list(MODELS)
if args.extra:   # comparison models only, same settings as the notebook's comparison section
    ENS = []
    MODELS = ({
        'SVM': lambda: SVC(C=10, kernel='rbf', gamma='scale', random_state=RS),
        'RandomForest': lambda: RandomForestClassifier(n_estimators=300, max_depth=15, n_jobs=args.jobs, random_state=RS),
        'AdaBoost': lambda: AdaBoostClassifier(n_estimators=100, random_state=RS),
        'NaiveBayes': lambda: GaussianNB(),
    })


def scores(clf, B):
    if hasattr(clf, 'predict_proba'): return clf.predict_proba(B)
    d = clf.decision_function(B)
    return np.c_[-d, d] if d.ndim == 1 else d


def trial_pred(Pr, cls, rows):
    t = tw[rows]; ut = np.unique(t)
    return ut, cls[np.array([Pr[t == u].mean(0) for u in ut]).argmax(1)]


X, Xm, ymap, y = T['X'], T['Xm'], T['ymap'], T['y']
idx = np.where(T['mask'])[0]
ck = os.path.join(OUT, f'models_{args.task}_{args.tag}.json')
done = json.load(open(ck)) if os.path.exists(ck) else {}
folds = list(LeaveOneGroupOut().split(idx, y[idx], groups=sw[idx]))
for k, (tr_l, te_l) in enumerate(folds):
    tr, te = idx[tr_l], idx[te_l]
    assert len(np.unique(sw[te])) == 1 and not set(sw[tr]) & set(sw[te]) and not set(tw[tr]) & set(tw[te])
    person = str(int(sw[te][0]))
    if person in done: continue
    t0 = time.time()
    A = np.vstack([X[tr], Xm[tr]]); yA = np.concatenate([y[tr], ymap(y[tr])])
    sc = StandardScaler().fit(A)
    A_s, B_s, R_s = sc.transform(A), sc.transform(X[te]), sc.transform(X[tr])   # R = real (unmirrored) training windows
    P, PR, cls_ref = {}, {}, None
    for name, mk in MODELS.items():
        clf = mk().fit(A_s, yA)
        cls = clf.classes_
        if cls_ref is None: cls_ref = cls
        assert np.array_equal(cls, cls_ref), 'class order differs between models'
        P[name], PR[name] = scores(clf, B_s), scores(clf, R_s)
    if ENS: P['Ensemble'] = np.mean([P[m] for m in ENS], 0); PR['Ensemble'] = np.mean([PR[m] for m in ENS], 0)
    rec = {'trials': None, 'pred': {}, 'win_test': {}, 'train_trial_acc': {}, 'train_win_acc': {}}
    for name in P:
        ut, yp = trial_pred(P[name], cls_ref, te)
        rec['trials'] = [[int(u), int(y[tw == u][0])] for u in ut]
        rec['pred'][name] = [int(v) for v in yp]
        rec['win_test'][name] = [int((cls_ref[P[name].argmax(1)] == y[te]).sum()), len(te)]
        utr, yptr = trial_pred(PR[name], cls_ref, tr)
        ytr = np.array([y[tw == u][0] for u in utr])
        rec['train_trial_acc'][name] = float((yptr == ytr).mean())
        rec['train_win_acc'][name] = float((cls_ref[PR[name].argmax(1)] == y[tr]).mean())
    done[person] = rec
    json.dump(done, open(ck, 'w'))
    key = 'Ensemble' if ENS else 'SVM'
    e = sum(sum(np.array(r['pred'][key]) == np.array([a for _, a in r['trials']])) for r in done.values())
    n = sum(len(r['trials']) for r in done.values())
    print(f'  person {len(done):2d}/{len(folds)} done in {time.time() - t0:.0f}s, {key} running {e}/{n} = {100 * e / n:.2f}%', flush=True)
print('DONE', flush=True)

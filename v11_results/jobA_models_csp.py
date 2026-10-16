# Job A: every classifier, with and without filter-bank CSP, on all four tasks, inside the final v10 pipeline.
exec(open('/tmp/claude-0/exp/v11/loader.py').read())
exec('''
import json, time
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
RS = 42
MODELS = {
  "LogReg": lambda: LogisticRegression(C=0.01, max_iter=3000),
  "LDA": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
  "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=500, max_features="sqrt", n_jobs=2, random_state=RS),
  "HistGB": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=RS),
  "SVM": lambda: SVC(C=10, kernel="rbf", gamma="scale"),
  "RandomForest": lambda: RandomForestClassifier(n_estimators=300, max_depth=15, n_jobs=2, random_state=RS),
  "NaiveBayes": lambda: GaussianNB(),
  "AdaBoost": lambda: AdaBoostClassifier(n_estimators=100, random_state=RS),
}
ENS = ["LogReg", "LDA", "ExtraTrees", "HistGB"]
bands = list(X_bands.values())

def rec_align(F, idx):
    F = F.copy()
    for r in np.unique(sess_w[idx]):
        m = sess_w[idx] == r
        F[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-8)
    return F

def fbcsp(tr, te, y_tr, ymap):
    ftr, ftrm, fte = [], [], []
    for Xb in bands:
        A = np.concatenate([Xb[tr], Xb[tr][:, ch_perm]]).astype(np.float64)
        ya = np.concatenate([y_tr, ymap(y_tr)])
        csp = CSP(n_components=4, reg="ledoit_wolf", log=True).fit(A, ya)
        ftr.append(csp.transform(Xb[tr].astype(np.float64))); ftrm.append(csp.transform(Xb[tr][:, ch_perm].astype(np.float64)))
        fte.append(csp.transform(Xb[te].astype(np.float64)))
    return rec_align(np.hstack(ftr), tr), rec_align(np.hstack(ftrm), tr), rec_align(np.hstack(fte), te)

def scores(clf, B):
    if hasattr(clf, "predict_proba"): return clf.predict_proba(B)
    d = clf.decision_function(B)
    return np.c_[-d, d] if d.ndim == 1 else d

results = {}
for task in ["LR", "bin", "6", "12"]:
    T = TASKS[task]
    idx = np.where(T["mask"])[0]; y = T["y"]
    for fold, (tr_l, te_l) in enumerate(GroupKFold(n_splits=T["k"]).split(idx, y[idx], groups=sw[idx])):
        tr, te = idx[tr_l], idx[te_l]; t0 = time.time()
        Ctr, Ctrm, Cte = fbcsp(tr, te, y[tr], T["ymap"])
        tt = trial_of_win[te]; ut = np.unique(tt); yt = np.array([y[te][tt == t][0] for t in ut])
        for use_csp in (False, True):
            A = np.vstack([T["X"][tr], T["Xm"][tr]]); B = T["X"][te]
            if use_csp: A = np.hstack([A, np.vstack([Ctr, Ctrm])]); B = np.hstack([B, Cte])
            yA = np.concatenate([y[tr], T["ymap"](y[tr])])
            sc = StandardScaler().fit(A); A, B = sc.transform(A), sc.transform(B)
            P = {}
            for name, mk in MODELS.items():
                clf = mk().fit(A, yA); P[name] = scores(clf, B); cls = clf.classes_
            P["Ensemble"] = np.mean([P[m] for m in ENS], 0)
            for name, Pr in P.items():
                key = f"{task}|{name}|{'CSP' if use_csp else 'base'}"
                r = results.setdefault(key, dict(correct=0, n=0, folds=[], window=[]))
                yp = cls[np.array([Pr[tt == t].mean(0) for t in ut]).argmax(1)]
                c = int((yp == yt).sum()); r["correct"] += c; r["n"] += len(ut); r["folds"].append(c / len(ut))
                r["window"].append(float(np.mean(cls[Pr.argmax(1)] == y[te])))
        print(f"{task} fold {fold+1} done in {time.time()-t0:.0f}s", flush=True)
        json.dump(results, open("/tmp/claude-0/exp/v11/jobA.json", "w"))
print("JOB A DONE", flush=True)
''', G)

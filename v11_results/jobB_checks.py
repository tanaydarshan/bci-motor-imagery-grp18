# Job B: reproducibility, label-permutation on all tasks, and nested cross-validation of every design choice.
exec(open('/tmp/claude-0/exp/v11/loader.py').read())
exec('''
import json, time
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
OUT = {}
def save(): json.dump(OUT, open("/tmp/claude-0/exp/v11/jobB.json", "w"))
def members(rs):
    return {"LogReg": lambda: LogisticRegression(C=0.01, max_iter=3000),
            "LDA": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=500, max_features="sqrt", n_jobs=2, random_state=rs),
            "HistGB": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=rs)}

def cv(idx, y, ymap, X, Xm, k, rs=42, mirror=True):
    """Person-wise CV on the people in idx; returns pooled trial-correct counts for the 4 members and the ensemble."""
    corr = {m: 0 for m in ["LogReg", "LDA", "ExtraTrees", "HistGB", "Ensemble"]}; n = 0
    for tr_l, te_l in GroupKFold(n_splits=k).split(idx, y[idx], groups=sw[idx]):
        tr, te = idx[tr_l], idx[te_l]
        A, yA = X[tr], y[tr]
        if mirror: A = np.vstack([A, Xm[tr]]); yA = np.concatenate([yA, ymap(y[tr])])
        sc = StandardScaler().fit(A); A, B = sc.transform(A), sc.transform(X[te])
        P = {}
        for m, mk in members(rs).items():
            c = mk().fit(A, yA); P[m] = c.predict_proba(B); cls = c.classes_
        P["Ensemble"] = np.mean(list(P.values()), 0)
        tt = trial_of_win[te]; ut = np.unique(tt); yt = np.array([y[te][tt == t][0] for t in ut])
        for m, Pr in P.items():
            corr[m] += int((cls[np.array([Pr[tt == t].mean(0) for t in ut]).argmax(1)] == yt).sum())
        n += len(ut)
    return corr, n

# ---- 1. Reproducibility: same seed twice, then three other seeds (binary, final v10 setup)
T = TASKS["bin"]; idx = np.where(T["mask"])[0]
rep = []
for rs in [42, 42, 1, 2, 3]:
    c, n = cv(idx, T["y"], T["ymap"], T["X"], T["Xm"], T["k"], rs=rs); rep.append([rs, c["Ensemble"], n])
    print("repro seed", rs, c["Ensemble"], "/", n, flush=True)
OUT["repro_bin"] = rep; save()

# ---- 2. Label permutation on all four tasks (labels shuffled across trials within each recording)
rng = np.random.default_rng(7)
def permuted(y, mask):
    yp = y.copy()
    lab = {t: y[trial_of_win == t][0] for t in np.unique(trial_of_win[mask])}
    for r in np.unique(sess_w[mask]):
        ts = np.unique(trial_of_win[(sess_w == r) & mask]); l = np.array([lab[t] for t in ts]); rng.shuffle(l)
        for t, v in zip(ts, l): yp[trial_of_win == t] = v
    return yp
OUT["perm"] = {}
for task, T in TASKS.items():
    idx = np.where(T["mask"])[0]; accs = []
    for rep_i in range(3):
        c, n = cv(idx, permuted(T["y"], T["mask"]), T["ymap"], T["X"], T["Xm"], T["k"]); accs.append(c["Ensemble"] / n)
    OUT["perm"][task] = accs; save(); print("perm", task, [round(100*a, 2) for a in accs], flush=True)

# ---- 3. Nested cross-validation: every development choice is re-made using ONLY the training people of each outer fold
CONFIGS = {
  "bin": {"no-align": (X_noalign, None, False), "align": (X_v8, X_v10_mir, False), "align+mirror": (X_v8, X_v10_mir, True)},
  "6":   {"no-align": (X_noalign, None, False), "align": (X_v8, X_v10_mir, False), "align+mirror": (X_v8, X_v10_mir, True)},
  "12":  {"no-align": (X_noalign, None, False), "align": (X_v8, X_v10_mir, False), "align+mirror": (X_v8, X_v10_mir, True)},
  "LR":  {"std-cal": (X_v8, X_v10_mir, False), "std-cal+mirror": (X_v8, X_v10_mir, True),
          "img-cal": (X_img, X_img_mir, False), "img-cal+mirror": (X_img, X_img_mir, True)},
}
OUT["nested"] = {}
for task, T in TASKS.items():
    idx = np.where(T["mask"])[0]; y = T["y"]; k = T["k"]; tot_c = tot_n = 0; picks = []
    for fo, (tr_l, te_l) in enumerate(GroupKFold(n_splits=k).split(idx, y[idx], groups=sw[idx])):
        outer_tr, outer_te = idx[tr_l], idx[te_l]; t0 = time.time()
        best = (-1, None, None)
        for cname, (X, Xm, mir) in CONFIGS[task].items():
            c, n = cv(outer_tr, y, T["ymap"], X, Xm, 4 if k == 5 else 3, mirror=mir)   # inner CV on training people only
            for m, v in c.items():
                if v / n > best[0]: best = (v / n, cname, m)
        _, cname, m = best; X, Xm, mir = CONFIGS[task][cname]
        A, yA = X[outer_tr], y[outer_tr]
        if mir: A = np.vstack([A, Xm[outer_tr]]); yA = np.concatenate([yA, T["ymap"](y[outer_tr])])
        sc = StandardScaler().fit(A); A, B = sc.transform(A), sc.transform(X[outer_te])
        mods = members(42)
        if m == "Ensemble":
            Ps = []
            for mk in mods.values():
                cl = mk().fit(A, yA); Ps.append(cl.predict_proba(B)); cls = cl.classes_
            Pr = np.mean(Ps, 0)
        else:
            cl = mods[m]().fit(A, yA); Pr = cl.predict_proba(B); cls = cl.classes_
        tt = trial_of_win[outer_te]; ut = np.unique(tt); yt = np.array([y[outer_te][tt == t][0] for t in ut])
        cc = int((cls[np.array([Pr[tt == t].mean(0) for t in ut]).argmax(1)] == yt).sum())
        tot_c += cc; tot_n += len(ut); picks.append([cname, m, round(best[0], 4), cc, len(ut)])
        print(f"nested {task} outer {fo+1}: picked {cname}/{m} (inner {best[0]:.3f}) -> outer {cc}/{len(ut)} ({time.time()-t0:.0f}s)", flush=True)
    OUT["nested"][task] = dict(correct=tot_c, n=tot_n, picks=picks); save()
    print(f"NESTED {task}: {tot_c}/{tot_n} = {100*tot_c/tot_n:.2f}%", flush=True)
print("JOB B DONE", flush=True)
''', G)

"""Recompute, from the fresh feature cache, every GroupKFold number in the paper that is not covered by the LOSO files:
main 5-fold results and confusion statistics, ablation, feature families, and onset sensitivity (+-2 s).
Usage: FEATURE_CACHE=feature_cache_fresh python loso/verify_paper_numbers.py [main|families|onset]
"""
import io, contextlib, json, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'loso', 'results', 'paper_check.json')
part = sys.argv[1] if len(sys.argv) > 1 else 'main'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def load(shift=0.0):
    """Load the notebook pipeline; optionally shift every detected onset by `shift` seconds."""
    nb = json.load(open(os.path.join(REPO, 'EEG_preprocessing_v10.ipynb'))); C = nb['cells']
    cache = os.environ.get('FEATURE_CACHE', 'feature_cache') + ('' if shift == 0 else f'_shift{shift:+.0f}')
    G = {}
    os.chdir(REPO)
    for i, c in enumerate(C):
        if c['cell_type'] != 'code' or i < 3: continue
        code = ''.join(c['source']).replace("CACHE_DIR = 'feature_cache'", f"CACHE_DIR = '{cache}'")
        if 'import tensorflow' in code or 'Input(' in code or 'run_dl(' in code: continue
        if ('run_ml(' in code and 'def run_ml' not in code) or 'gkf3 = GroupKFold' in code or 'abl = {}' in code \
                or 'plt.' in code or 'scores_anova' in code: continue
        if 'def run_ensemble' in code: code = code.split('ens_bin = run_ensemble')[0]
        if 'MIRROR = {' in code: code = code.split('final_bin = run_mirror')[0]
        with contextlib.redirect_stdout(io.StringIO()): exec(code, G)
        if shift and 'onsets = {}' in code:
            G['onsets'] = {k: v + shift for k, v in G['onsets'].items()}
        if 'MIRROR = {' in code: break
    return G


from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def cv(G, X, y, k, Xm=None, ymap=None, mask=None, models=None):
    """Person-wise GroupKFold; returns pooled trial-level correct/n, fold means, window mean, and trial predictions."""
    sw, tw = G['sw'], G['trial_of_win']
    idx = np.arange(len(y)) if mask is None else np.where(mask)[0]
    models = models or G['ENS_MODELS']
    out = {m: dict(correct=0, n=0, trial=[], window=[]) for m in list(models) + ['Ensemble']}
    preds = {}
    for tr_l, te_l in GroupKFold(n_splits=k).split(idx, y[idx], groups=sw[idx]):
        tr, te = idx[tr_l], idx[te_l]
        assert not set(sw[tr]) & set(sw[te])
        A, yA = X[tr], y[tr]
        if Xm is not None: A = np.vstack([A, Xm[tr]]); yA = np.concatenate([yA, ymap(y[tr])])
        sc = StandardScaler().fit(A); A, B = sc.transform(A), sc.transform(X[te])
        P = {}
        for m, mk in models.items():
            clf = mk().fit(A, yA); P[m] = clf.predict_proba(B); cls = clf.classes_
        P['Ensemble'] = np.mean([P[m] for m in models], 0)
        tt = tw[te]; ut = np.unique(tt); yt = np.array([y[te][tt == t][0] for t in ut])
        for m, Pr in P.items():
            yp = cls[np.array([Pr[tt == t].mean(0) for t in ut]).argmax(1)]
            c = int((yp == yt).sum()); out[m]['correct'] += c; out[m]['n'] += len(ut)
            out[m]['trial'].append(c / len(ut)); out[m]['window'].append(float((cls[Pr.argmax(1)] == y[te]).mean()))
            if m == 'Ensemble': preds.update({int(t): (int(a), int(b)) for t, a, b in zip(ut, yt, yp)})
    summ = {m: dict(correct=v['correct'], n=v['n'], pooled=100 * v['correct'] / v['n'],
                    fold_mean=100 * np.mean(v['trial']), fold_sd=100 * np.std(v['trial']),
                    window_mean=100 * np.mean(v['window']), window_sd=100 * np.std(v['window'])) for m, v in out.items()}
    return summ, preds


def save():
    json.dump(res, open(OUT, 'w'), indent=1, default=float)


if part == 'main':
    G = load()
    TK = {'bin': (G['yw_bin'], 5), '6': (G['yw_body'], 3), '12': (G['yw_multi'], 3)}
    ml = G['mirror_label']
    from pyriemann.tangentspace import TangentSpace
    X_noalign = np.hstack([G['X_all'], TangentSpace(metric='riemann').fit_transform(G['covs_all'])])
    for t, (y, k) in TK.items():
        t0 = time.time()
        res[f'mirror_{t}'], p = cv(G, G['X_final'], y, k, G['X_mirror'], ml[{'bin': 'bin', '6': 'body', '12': 'multi'}[t]])
        res[f'mirror_{t}_preds'] = p
        res[f'nomirror_{t}'], _ = cv(G, G['X_final'], y, k)
        res[f'noalign_{t}'], _ = cv(G, X_noalign, y, k)
        save(); print(t, 'mirror', res[f'mirror_{t}']['Ensemble'], f'{time.time() - t0:.0f}s', flush=True)
    print('DONE main', flush=True)

if part == 'families':
    G = load()
    from pyriemann.tangentspace import TangentSpace
    n = [G['X_feat'].shape[1], G['X_spectro'].shape[1], G['X_scalo'].shape[1], G['X_conn'].shape[1]]
    o = np.cumsum([0] + n)
    Xa = G['X_final']
    fam = {'STFT': Xa[:, o[1]:o[2]], 'CWT': Xa[:, o[2]:o[3]], 'STFT+CWT': Xa[:, o[1]:o[3]],
           'Handcrafted': Xa[:, o[0]:o[1]], 'All': Xa}
    for t, (y, k) in {'bin': (G['yw_bin'], 5), '6': (G['yw_body'], 3), '12': (G['yw_multi'], 3)}.items():
        for f, X in fam.items():
            if f'family_{f}_{t}' in res: continue  # already computed before a restart
            r, _ = cv(G, X, y, k); res[f'family_{f}_{t}'] = r['Ensemble']; save()
            print('family', f, t, r['Ensemble']['correct'], flush=True)
    print('DONE families', flush=True)

if part == 'onset':
    from pyriemann.estimation import Covariances
    for shift in (-2.0, 2.0):
        if all(f'onset_{shift:+.0f}_{t}' in res for t in ('bin', '6')): continue  # done before a restart
        G = load(shift)
        # fast families as in the original check: spectrogram, scalogram, connectivity, covariance (aligned), no mirror
        n = [G['X_feat'].shape[1], G['X_spectro'].shape[1], G['X_scalo'].shape[1], G['X_conn'].shape[1]]
        X = G['X_final'][:, n[0]:]
        for t, (y, k) in {'bin': (G['yw_bin'], 5), '6': (G['yw_body'], 3)}.items():
            if f'onset_{shift:+.0f}_{t}' in res: continue
            r, _ = cv(G, X, y, k); res[f'onset_{shift:+.0f}_{t}'] = r['Ensemble']; save()
            print('onset', shift, t, r['Ensemble']['correct'], '/', r['Ensemble']['n'], flush=True)
    G = load(0.0)
    n = [G['X_feat'].shape[1], G['X_spectro'].shape[1], G['X_scalo'].shape[1], G['X_conn'].shape[1]]
    X = G['X_final'][:, n[0]:]
    for t, (y, k) in {'bin': (G['yw_bin'], 5), '6': (G['yw_body'], 3)}.items():
        r, _ = cv(G, X, y, k); res[f'onset_+0_{t}'] = r['Ensemble']; save()
        print('onset 0', t, r['Ensemble']['correct'], '/', r['Ensemble']['n'], flush=True)
    print('DONE onset', flush=True)

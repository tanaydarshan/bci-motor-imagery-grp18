# Loads the exact v10 notebook pipeline (data, timing, cleaning, features, alignment, mirror) without running models.
import json, os, io, contextlib, numpy as np
os.chdir('/home/user/bci-motor-imagery-grp18')
nb = json.load(open('EEG_preprocessing_v10.ipynb')); C = nb['cells']; s = lambda i: ''.join(C[i]['source'])
G = {}
for i, c in enumerate(C):
    if c['cell_type'] != 'code' or i < 3: continue
    code = s(i)
    if 'import tensorflow' in code or 'Input(' in code or 'run_dl(' in code: continue
    if ('run_ml(' in code and 'def run_ml' not in code) or 'gkf3 = GroupKFold' in code or 'abl = {}' in code or 'plt.' in code or 'scores_anova' in code: continue
    if 'def run_v8' in code: code = code.split('v8_bin = run_v8')[0]
    if 'MIRROR = {' in code: code = code.split('v10_bin = run_v10')[0]
    with contextlib.redirect_stdout(io.StringIO()): exec(code, G)
    if 'MIRROR = {' in code: break
exec('''
side_w = yw_body % 2; img = yw_bin == 1
def imagery_align(Xin):
    Xo = Xin.copy()
    for r in np.unique(sess_w):
        m = (sess_w == r) & img
        if m.sum() > 1: Xo[m] = (Xo[m] - Xo[m].mean(0)) / (Xo[m].std(0) + 1e-8)
    return Xo
X_img, X_img_mir = imagery_align(X_v8), imagery_align(X_v10_mir)
from pyriemann.tangentspace import TangentSpace as _TS
X_noalign = np.hstack([X_all, _TS(metric="riemann").fit_transform(covs_all)])
TASKS = {
  "bin": dict(mask=np.ones(len(yw_bin), bool), y=yw_bin, ymap=mirror_label["bin"], k=5, X=X_v8, Xm=X_v10_mir),
  "6":   dict(mask=np.ones(len(yw_bin), bool), y=yw_body, ymap=mirror_label["body"], k=3, X=X_v8, Xm=X_v10_mir),
  "12":  dict(mask=np.ones(len(yw_bin), bool), y=yw_multi, ymap=mirror_label["multi"], k=3, X=X_v8, Xm=X_v10_mir),
  "LR":  dict(mask=img, y=side_w, ymap=mirror_label["side"], k=5, X=X_img, Xm=X_img_mir),
}
''', G)

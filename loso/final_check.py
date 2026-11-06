"""Final completeness and correctness check of every LOSO result file.

For each task and model: all people present, every trial predicted exactly once, stored true labels equal the
labels recomputed from the data, accuracy recomputed from the raw predictions.
"""
import io, contextlib, json, os
import numpy as np
from scipy.stats import binomtest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(REPO, 'loso', 'results')
LNS = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(REPO, 'v11_results', 'loader.py')).read(), LNS)
G = LNS['G']; sw, tw = G['sw'], G['trial_of_win']
ok = True
out = {}
for task, chance in [('bin', .5), ('6', 1 / 6), ('12', 1 / 12), ('LR', .5)]:
    T = G['TASKS'][task]; idx = np.where(T['mask'])[0]; y = T['y']
    trials = np.unique(tw[idx]); people = np.unique(sw[idx])
    truth = {int(t): int(y[tw == t][0]) for t in trials}
    owner = {int(t): int(sw[tw == t][0]) for t in trials}
    D = json.load(open(os.path.join(R, f'models_{task}_run1.json')))
    fx = os.path.join(R, f'models_{task}_extra.json')
    X = json.load(open(fx)) if os.path.exists(fx) else None
    print(f'\n=== {task}: {len(people)} people, {len(trials)} trials, chance {100 * chance:.2f}% ===')
    checks = [len(D) == len(people), set(map(int, D)) == set(map(int, people))]
    seen = [t for r in D.values() for t, _ in r['trials']]
    checks += [len(seen) == len(set(seen)) == len(trials), set(seen) == set(map(int, trials))]
    checks += [all(truth[t] == a for r in D.values() for t, a in r['trials'])]
    checks += [all(owner[t] == int(p) for p, r in D.items() for t, _ in r['trials'])]
    print('  people complete / trials complete & unique / labels match data / trials belong to their test person:',
          all(checks), checks)
    ok &= all(checks)
    models = list(next(iter(D.values()))['pred'])
    sources = [(m, D) for m in models] + ([(m, X) for m in next(iter(X.values()))['pred']] if X else [])
    if X: assert len(X) == len(people) and all(X[p]['trials'] == D[p]['trials'] for p in D)
    out[task] = {}
    for m, S in sources:
        c = sum(int(v == truth[t]) for r in S.values() for (t, _), v in zip(r['trials'], r['pred'][m]))
        n = sum(len(r['pred'][m]) for r in S.values())
        assert n == len(trials)
        p = binomtest(c, n, chance, alternative='greater').pvalue
        out[task][m] = [c, n]
        print(f'  {m:<13} {c:>3}/{n} = {100 * c / n:6.2f}%   p = {p:.2g}')
print('\nALL CHECKS PASSED' if ok else '\nCHECK FAILED')
json.dump(out, open(os.path.join(R, 'final_check.json'), 'w'), indent=1)

"""Colour figures for Section 4 (results) from the verified LOSO result files. Run from the repository root."""
import json, os, shutil
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'paper_final', 'figures')
R = os.path.join(REPO, 'loso', 'results')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
                     'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'stix'})

# existing colour figures produced from the final notebook
for src, dst in [('fig6_families.png', 'fig11_families.png'), ('fig7_confusion.png', 'fig13_confusion.png'),
                 ('fig8_importance.png', 'fig12_importance.png')]:
    shutil.copy(os.path.join(REPO, 'paper', 'figures_color', src), os.path.join(OUT, dst))

# LOSO accuracy of every model on every task
S = json.load(open(os.path.join(R, 'summary_models.json')))
tasks = [('bin', 'Action vs. imagery', 50), ('6', 'Body part (6 classes)', 100 / 6),
         ('12', 'Twelve tasks', 100 / 12), ('LR', 'Imagery vs. imagery (L/R)', 50)]
models = ['LogReg', 'LDA', 'ExtraTrees', 'HistGB', 'Ensemble']
names = {'LogReg': 'LR', 'LDA': 'LDA', 'ExtraTrees': 'Extra Trees', 'HistGB': 'HistGB', 'Ensemble': 'Ensemble'}
cols = ['#5B8FD0', '#7FB77E', '#E6A157', '#B07CC6', '#1F4E8C']
fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
for ax, (t, title, chance) in zip(axes, tasks):
    vals = [100 * S[t][m]['acc'] for m in models]
    bars = ax.bar(range(5), vals, color=cols, edgecolor='#333333', lw=0.6)
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f'{v:.1f}', ha='center', fontsize=8.5)
    ax.axhline(chance, color='#C0392B', ls='--', lw=1.2)
    ax.set_xticks(range(5)); ax.set_xticklabels([names[m] for m in models], rotation=35, ha='right', fontsize=9)
    ax.set_title(title, fontsize=11); ax.set_ylim(0, max(vals) * 1.15)
axes[0].set_ylabel('LOSO trial accuracy (%)')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig14_loso_models.png'), dpi=250, bbox_inches='tight'); plt.close(fig)

# learning curve
L = json.load(open(os.path.join(R, 'learning_curve_bin.json')))
x = [5, 10, 20, 30, 38, 46]
y = [100 * L[k]['test_acc'] for k in ['5', '10', '20', '30', 'all']] + [100 * S['bin']['Ensemble']['acc']]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(x, y, 'o-', color='#1F4E8C', lw=2, ms=6)
for xi, yi in zip(x, y): ax.text(xi, yi + 0.8, f'{yi:.1f}', ha='center', fontsize=9)
ax.axhline(50, color='#C0392B', ls='--', lw=1.2); ax.text(46, 51.2, 'chance', ha='right', fontsize=9, color='#C0392B')
ax.annotate('LOSO', xy=(46, y[-1]), xytext=(41, y[-1] - 9), fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.8))
ax.set_xlabel('Number of training participants'); ax.set_ylabel('Test accuracy (%)')
ax.set_ylim(45, 88); ax.set_xticks(x)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig15_learning_curve.png'), dpi=250, bbox_inches='tight'); plt.close(fig)
print('done')

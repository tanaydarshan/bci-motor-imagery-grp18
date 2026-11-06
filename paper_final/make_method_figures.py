"""Colour figures for Section 3 (methodology). Run from the repository root:
    python paper_final/make_method_figures.py
"""
import io, contextlib, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'paper_final', 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
                     'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'stix'})

ACT, IMG = '#2F6DB5', '#E07B39'           # action / imagery
BLUE_L, GREEN_L, ORANGE_L, PURPLE_L, GREY_L, ROSE_L = '#DCE8F5', '#DDEFD9', '#FBE3CF', '#E8E0F2', '#EEEEEE', '#F6DADA'
EDGE = '#3A3A3A'


def box(ax, x, y, w, h, title, sub=None, fc=BLUE_L, fs=11, sfs=9.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.012,rounding_size=0.025',
                                fc=fc, ec=EDGE, lw=1.1))
    if sub:
        ax.text(x, y + h * 0.16, title, ha='center', va='center', fontsize=fs, weight='bold')
        ax.text(x, y - h * 0.2, sub, ha='center', va='center', fontsize=sfs, style='italic', color='#333333')
    else:
        ax.text(x, y, title, ha='center', va='center', fontsize=fs, weight='bold')


def arrow(ax, x0, y0, x1, y1, color=EDGE):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=13, lw=1.2, color=color,
                                 shrinkA=0, shrinkB=0))


def blank(figsize, xlim=(0, 1), ylim=(0, 1)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis('off')
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', name)


# ---------------------------------------------------------------- 1. pipeline
fig, ax = blank((9.2, 6.4))
ax.text(0.25, 0.97, '(a) Signal preparation', ha='center', fontsize=12)
ax.text(0.75, 0.97, '(b) Representation and classification', ha='center', fontsize=12)
left = [('Raw EEG and motion recordings', 'Emotiv EPOC X, 14 channels, 128 Hz', GREY_L),
        ('Trial onset detection', 'motion sensor, action vs. imagery contrast', ORANGE_L),
        ('Band-pass filtering (4–45 Hz)', 'zero-phase FIR', BLUE_L),
        ('Trial extraction and rejection', '12 trials per recording; 636 → 476', BLUE_L),
        ('Per-recording normalisation', 'channel-wise z-score', BLUE_L)]
right = [('Sliding windows', '2 s length, 0.5 s stride; 8,092 windows', GREEN_L),
         ('Feature extraction', '1,771 spectral, time–frequency, spatial features', GREEN_L),
         ('Subject-wise alignment', 'per-recording standardisation, re-centring', GREEN_L),
         ('Ensemble classification', 'LR, LDA, Extra Trees, HistGB; mirror-augmented', PURPLE_L),
         ('Trial-level decision', 'mean of 17 window probabilities', ROSE_L)]
ys = np.linspace(0.85, 0.12, 5)
for col, items in [(0.25, left), (0.75, right)]:
    for i, (t, s, c) in enumerate(items):
        box(ax, col, ys[i], 0.42, 0.11, t, s, fc=c)
        if i: arrow(ax, col, ys[i - 1] - 0.055, col, ys[i] + 0.055)
ax.plot([0.46, 0.5, 0.5], [ys[-1], ys[-1], ys[0]], color=EDGE, lw=1.2)
arrow(ax, 0.5, ys[0], 0.54, ys[0])
save(fig, 'fig1_pipeline.png')

# ---------------------------------------------------------------- 2. protocol
names = ['RA', 'LA', 'RL', 'LL', 'RW', 'LW']
fig, axes = plt.subplots(2, 1, figsize=(10, 3.6), sharex=True)
for ax, start, base, title in [(axes[0], 25, False, '(a) Standard session'), (axes[1], 120, True, '(b) Session with resting baseline')]:
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    if base:
        ax.add_patch(Rectangle((0, 0.15), 50, 0.7, fc='#CFE8D5', ec=EDGE, lw=0.8))
        ax.text(25, 0.5, 'eyes open', ha='center', va='center', fontsize=9)
        ax.add_patch(Rectangle((55, 0.15), 50, 0.7, fc='#B9D1E8', ec=EDGE, lw=0.8))
        ax.text(80, 0.5, 'eyes closed', ha='center', va='center', fontsize=9)
    for k in range(12):
        t0 = start + 15 * k
        c = ACT if k % 2 == 0 else IMG
        ax.add_patch(Rectangle((t0, 0.15), 10, 0.7, fc=c, ec=EDGE, lw=0.6, alpha=0.9))
        ax.text(t0 + 5, 0.5, f'{names[k // 2]}-{"AI"[k % 2]}', ha='center', va='center', fontsize=7.5, rotation=90, color='white')
    ax.set_title(title, loc='left', fontsize=10.5)
axes[1].set_xlim(0, 300); axes[1].set_xlabel('Time from start of recording (s)')
axes[0].legend(handles=[Rectangle((0, 0), 1, 1, fc=ACT), Rectangle((0, 0), 1, 1, fc=IMG)],
               labels=['Action (10 s)', 'Imagery (10 s)'], loc='upper right', ncol=2, frameon=False, fontsize=9)
fig.tight_layout(); save(fig, 'fig2_protocol.png')

# ---------------------------------------------------------------- 3. onset detection (real recording)
G = {}
nb = json.load(open(os.path.join(REPO, 'EEG_preprocessing_v10.ipynb')))
with contextlib.redirect_stdout(io.StringIO()):
    os.chdir(REPO)
    for i in [3, 6, 9]:
        exec(''.join(nb['cells'][i]['source']), G)
tab = G['onset_table']
cand = tab[(tab.onset_s == 26.0) & (tab.status.str.startswith('motion (confident)'))].sort_values('t_stat', ascending=False)
rec = cand.iloc[0]
path = os.path.join(G['BASE_PATH'], rec.recording)
z = G['motion_energy'](path[:-4] + '.md.edf', 0.5); z = (z - np.median(z)) / (z.std() + 1e-9)
step = 0.5; t0s = np.arange(0, rec.duration_s - G['PROTOCOL_SEC'] + step, step); tcurve = []
for t0 in t0s:
    d = []
    for k in range(0, 12, 2):
        a = z[int((t0 + 15 * k + 1) / step):int((t0 + 15 * k + 10) / step)]
        b = z[int((t0 + 15 * (k + 1) + 1) / step):int((t0 + 15 * (k + 1) + 10) / step)]
        d.append(a.mean() - b.mean())
    d = np.array(d); tcurve.append(d.mean() / (d.std() / np.sqrt(6) + 1e-9))
fig, axes = plt.subplots(2, 1, figsize=(10, 6.2))
tt = np.arange(len(z)) * step
ax = axes[0]
for k in range(12):
    s = rec.onset_s + 15 * k
    ax.axvspan(s, s + 10, color=ACT if k % 2 == 0 else IMG, alpha=0.22, lw=0)
ax.plot(tt, z, color='#222222', lw=0.8)
ax.axvline(15, color='#888888', ls='--', lw=1.2); ax.axvline(rec.onset_s, color='#1B8A3A', lw=1.8)

ax.set_xlim(0, tt[-1]); ax.set_ylabel('Motion energy (z)'); ax.set_xlabel('Time (s)'); ax.set_title('(a)', loc='left')
from matplotlib.lines import Line2D
ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=ACT, alpha=0.35), Rectangle((0, 0), 1, 1, fc=IMG, alpha=0.35),
                   Line2D([], [], color='#888888', ls='--'), Line2D([], [], color='#1B8A3A', lw=1.8)],
          labels=['action block', 'imagery block', '15 s (old fixed onset)', f'{rec.onset_s:.1f} s (detected onset)'],
          loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=9)
ax = axes[1]
ax.plot(t0s, tcurve, color='#1B8A3A', lw=1.8)
ax.axhline(4, color='#888888', ls=':', lw=1); ax.text(0.3, 4.8, 't = 4 (confidence threshold)', fontsize=9, color='#555555')
ax.axvline(15, color='#888888', ls='--', lw=1.2); ax.axvline(rec.onset_s, color='#1B8A3A', lw=1.2, alpha=0.6)
ax.set_xlim(t0s[0], t0s[-1]); ax.set_xlabel('Candidate onset $t_0$ (s)'); ax.set_ylabel('Paired $t$'); ax.set_title('(b)', loc='left')
fig.tight_layout(); save(fig, 'fig3_onset.png')
print('onset example:', rec.recording.split('/')[0], rec.onset_s, round(rec.t_stat, 1))

# ---------------------------------------------------------------- 4. cleaning
fig, ax = blank((10, 4.2))
top = [(0.1, 'Raw EEG', '53 recordings', GREY_L), (0.3, 'Band-pass', '4–45 Hz FIR', BLUE_L), (0.5, 'Epoching', '636 trials', BLUE_L)]
for x, t, s, c in top: box(ax, x, 0.78, 0.15, 0.2, t, s, fc=c)
for i in range(2): arrow(ax, top[i][0] + 0.075, 0.78, top[i + 1][0] - 0.075, 0.78)
ax.add_patch(FancyBboxPatch((0.64, 0.6), 0.33, 0.36, boxstyle='round,pad=0.012,rounding_size=0.025', fc=ROSE_L, ec=EDGE, lw=1.1))
ax.text(0.805, 0.9, 'Trial rejection (any condition)', ha='center', fontsize=11, weight='bold')
for j, s in enumerate(['(i) flat signal, $\\sigma = 0$', '(ii) max$|x| \\geq$ 150 $\\mu$V   (158 trials)', '(iii) mean |kurtosis| $\\geq$ 10   (29 trials)']):
    ax.text(0.66, 0.82 - 0.075 * j, s, fontsize=10)
arrow(ax, 0.575, 0.78, 0.64, 0.78)
ax.text(0.805, 0.53, '160 trials rejected (27 by both rules)', ha='center', fontsize=10, style='italic', color='#9B2C2C')
arrow(ax, 0.805, 0.5, 0.805, 0.33)
bot = [(0.805, 'Clean trials', '476 trials, 47 participants', GREEN_L), (0.5, 'Per-recording z-score', r'$\hat{x} = (x-\mu_{r,c})/\sigma_{r,c}$', BLUE_L),
       (0.17, 'Normalised data', r'476 $\times$ 14 $\times$ 1280', GREY_L)]
for x, t, s, c in bot: box(ax, x, 0.2, 0.25, 0.22, t, s, fc=c)
arrow(ax, 0.68, 0.2, 0.625, 0.2); arrow(ax, 0.375, 0.2, 0.295, 0.2)
save(fig, 'fig4_cleaning.png')

# ---------------------------------------------------------------- 5. windowing
fig, ax = plt.subplots(figsize=(10, 3.9))
cm = plt.get_cmap('viridis')
for i in range(17):
    s = 0.5 * i
    ax.add_patch(Rectangle((s, 1.2 + 0.28 * i), 2, 0.2, fc=cm(i / 16), ec='none', alpha=0.9))
ax.text(-0.15, 1.3, '$w_1$', ha='right', va='center'); ax.text(8.0 - 0.15, 1.2 + 0.28 * 16 + 0.1, '$w_{17}$', ha='right', va='center')
ax.add_patch(Rectangle((0, 0.2), 10, 0.7, fc=BLUE_L, ec=EDGE, lw=1))
ax.text(5, 0.55, 'one trial: 10 s, 1280 samples, 14 channels', ha='center', va='center', fontsize=10.5)
ax.text(0.1, 5.3, 'window length 2 s (256 samples)\nstride 0.5 s (75% overlap)\n17 windows per trial', fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#999999'))
ax.set_xlim(-0.6, 10.2); ax.set_ylim(0, 6.1); ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xlabel('Time within trial (s)'); ax.set_xticks(range(11))
save(fig, 'fig5_windowing.png')

# ---------------------------------------------------------------- 6. feature families
fig, ax = blank((10.5, 5.2))
box(ax, 0.5, 0.9, 0.3, 0.12, 'EEG window', '14 channels × 256 samples', fc=GREY_L)
fams = [('Handcrafted', 'Welch PSD, statistics,\nHjorth, fractal, DWT', '336', BLUE_L),
        ('Spectrogram', 'STFT, 0.5 s Hann\nwindow, 50% overlap', '560', ORANGE_L),
        ('Scalogram', 'Morlet CWT,\n10 frequencies', '714', GREEN_L),
        ('Connectivity', 'coherence and\nasymmetry, 7 pairs', '56', PURPLE_L),
        ('Covariance', 'Ledoit–Wolf, 14 × 14,\ntangent space', '105', ROSE_L)]
xs = np.linspace(0.1, 0.9, 5)
for x, (t, s, n, c) in zip(xs, fams):
    ax.add_patch(FancyBboxPatch((x - 0.085, 0.38), 0.17, 0.3, boxstyle='round,pad=0.01,rounding_size=0.02', fc=c, ec=EDGE, lw=1.1))
    ax.text(x, 0.63, t, ha='center', fontsize=11, weight='bold')
    ax.text(x, 0.52, s, ha='center', va='center', fontsize=8.8, style='italic', color='#333333')
    ax.text(x, 0.42, f'{n} features', ha='center', fontsize=10)
    arrow(ax, 0.5, 0.84, x, 0.685); arrow(ax, x, 0.375, 0.5 + (x - 0.5) * 0.4, 0.19)
box(ax, 0.5, 0.12, 0.62, 0.13, 'Per-recording standardisation  →  1,771 features per window',
    'covariance matrices re-centred per recording before the tangent-space projection', fc='#FFF4D6')
save(fig, 'fig6_features.png')

# ---------------------------------------------------------------- 7. mirror augmentation
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={'width_ratios': [1, 1.15]})
pos = {'AF3': (-0.28, 0.82), 'AF4': (0.28, 0.82), 'F7': (-0.72, 0.5), 'F8': (0.72, 0.5), 'F3': (-0.38, 0.55), 'F4': (0.38, 0.55),
       'FC5': (-0.62, 0.25), 'FC6': (0.62, 0.25), 'T7': (-0.9, -0.02), 'T8': (0.9, -0.02), 'P7': (-0.72, -0.5), 'P8': (0.72, -0.5),
       'O1': (-0.3, -0.85), 'O2': (0.3, -0.85)}
ax = axes[0]; ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.2, 1.25)
ax.add_patch(Circle((0, 0), 1.05, fc='#FAFAFA', ec=EDGE, lw=1.2)); ax.plot([-0.1, 0, 0.1], [1.04, 1.17, 1.04], color=EDGE, lw=1.2)
ax.axvline(0, ymin=0.1, ymax=0.9, color='#BBBBBB', ls='--', lw=0.8)
pal = plt.get_cmap('tab10')
for j, (a, b) in enumerate([('AF3', 'AF4'), ('F7', 'F8'), ('F3', 'F4'), ('FC5', 'FC6'), ('T7', 'T8'), ('P7', 'P8'), ('O1', 'O2')]):
    (xa, ya), (xb, yb) = pos[a], pos[b]
    ax.annotate('', xy=(xb - 0.08, yb), xytext=(xa + 0.08, ya), arrowprops=dict(arrowstyle='<->', color=pal(j), lw=1.3, alpha=0.8))
    for nm, (x, y) in [(a, (xa, ya)), (b, (xb, yb))]:
        ax.add_patch(Circle((x, y), 0.075, fc=pal(j), ec='white', lw=0.8))
        ax.text(x, y - 0.15, nm, ha='center', fontsize=8)
ax.text(-0.8, 1.12, 'left', fontsize=9, color='#666666'); ax.text(0.65, 1.12, 'right', fontsize=9, color='#666666')
ax.set_title('(a) Swapped electrode pairs', fontsize=11)
ax = axes[1]; ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title('(b) Mirrored training copy', fontsize=11)
rows = [('Per-channel features', 'swap columns of partner electrodes'), ('Coherence (pair)', 'unchanged'),
        ('Asymmetry log(P$_L$/P$_R$)', 'sign flipped'), ('Covariance $\\tilde{C}$', '$P\\tilde{C}P^{\\top}$ before tangent space'),
        ('Label', 'RA↔LA, RL↔LL, RW↔LW')]
for i, (a, b) in enumerate(rows):
    y = 0.86 - 0.17 * i
    box(ax, 0.24, y, 0.44, 0.12, a, fc=BLUE_L, fs=9.5)
    arrow(ax, 0.465, y, 0.535, y)
    box(ax, 0.76, y, 0.44, 0.12, b, fc=ORANGE_L, fs=9)
ax.text(0.5, 0.0, 'Action/imagery label unchanged. Added to training folds only; test windows are never mirrored.', ha='center', fontsize=9.5, style='italic')
fig.tight_layout(); save(fig, 'fig9_mirror.png')

# ---------------------------------------------------------------- 8. evaluation
fig = plt.figure(figsize=(11, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.25)
for k, (title, nf, npeople) in enumerate([('(a) Person-wise 5-fold', 5, 5), ('(b) Leave-one-subject-out', 6, 12)]):
    ax = fig.add_subplot(gs[k]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(title, fontsize=11)
    for f in range(nf):
        y = 0.88 - f * 0.12
        lab = f'Fold {f + 1}' if (k == 0 or f < nf - 1) else 'Fold 47'
        if k == 1 and f == nf - 2:
            ax.plot([0.59] * 3, [y + 0.035, y, y - 0.035], 'o', color='#555555', ms=2.5); continue
        ax.text(0.0, y, lab, va='center', fontsize=9)
        w = 0.78 / npeople
        for p in range(npeople):
            test = (p == f) if k == 0 else (p == (f if f < nf - 1 else npeople - 1))
            ax.add_patch(Rectangle((0.2 + p * w, y - 0.04), w * 0.92, 0.08, fc=IMG if test else BLUE_L, ec=EDGE, lw=0.5))
    ax.text(0.58, 0.14, '9–10 test people per fold' if k == 0 else '1 test person, 46 for training', ha='center', fontsize=9, style='italic')
ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=IMG, ec=EDGE), Rectangle((0, 0), 1, 1, fc=BLUE_L, ec=EDGE)],
          labels=['test people', 'training people'], loc='lower center', bbox_to_anchor=(-0.1, -0.12), ncol=2, frameon=False, fontsize=9)
ax = fig.add_subplot(gs[2]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title('(c) Trial-level decision', fontsize=11)
for i in range(17):
    ax.add_patch(Rectangle((0.08 + i * 0.03, 0.9 - i * 0.008), 0.25, 0.006, fc=plt.get_cmap('viridis')(i / 16)))
ax.text(0.78, 0.86, '17 windows of\none test trial', ha='center', va='center', fontsize=9)
arrow(ax, 0.32, 0.74, 0.32, 0.68)
box(ax, 0.42, 0.58, 0.72, 0.13, r'Ensemble: $p_w = \frac{1}{4}\sum_m p_w^{(m)}$', fc=PURPLE_L, fs=10.5)
arrow(ax, 0.42, 0.515, 0.42, 0.445)
box(ax, 0.42, 0.38, 0.72, 0.12, r'Trial: $\bar{p} = \frac{1}{17}\sum_w p_w$', fc=GREEN_L, fs=10.5)
arrow(ax, 0.42, 0.32, 0.42, 0.25)
box(ax, 0.42, 0.18, 0.72, 0.12, r'Label $= \arg\max_c \bar{p}_c$', fc=ROSE_L, fs=10.5)
save(fig, 'fig10_evaluation.png')

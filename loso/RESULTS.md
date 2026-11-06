# Leave-one-subject-out (LOSO) evaluation — results and checks

All numbers below were produced by the scripts in this folder from the final v10 pipeline.

## How LOSO is done (no leakage)

* 47 folds: each participant is held out once, and the model is trained on the other 46 (`LeaveOneGroupOut`, groups = person, not recording).
* Steps that use no labels (filtering, trial rejection, per-recording z-scoring, feature extraction, per-recording feature alignment, covariance re-centring) are run once, as in the notebook.
* Everything that learns from labels is fitted inside each fold, on the 46 training people only: the scaler, every classifier, and mirror augmentation.
* The settings are the fixed v10 ones. Nothing was tuned on LOSO results, and the ensemble remains the reported model even when a single model scores higher.
* Built-in asserts check every fold: exactly one test person, and no person or trial in both training and test.

## Verification (`verify.py`, `log_verify.txt`)

| Check | Result |
|---|---|
| People / recordings / trials on both sides of a split | None, in all 4 tasks, for both 5-fold and LOSO |
| Every trial tested exactly once | Yes (476 trials; 241 for left vs right) |
| Mirror label map | RA↔LA, RL↔LL, RW↔LW; action/imagery unchanged |
| Feature alignment is label-free | Per-recording feature means are 0 (to 10⁻¹²), whatever the labels |
| LOSO vs 5-fold, action vs imagery | Same total (382) by coincidence: 7 trials right only in 5-fold and 7 only in LOSO; 97.1% identical predictions |
| LOSO vs 5-fold, body part / left vs right | 123 vs 127 and 141 vs 140; no significant difference (McNemar p = 0.73 and 1.00) |

## Reproducibility: does accuracy rise when rerun?

No. Independent reruns gave **identical per-trial predictions**:

* action vs imagery: 3 LOSO runs, 476/476 identical
* body part: 2 runs, 476/476 identical
* left vs right: 2 runs, 241/241 identical
* 12 classes: 172/172 identical on the 15 people both runs covered

## Learning curve, action vs imagery (`learning_curve.py`)

The same people are always tested (5-fold by person); only the number of training people changes.

| Training people | 5 | 10 | 20 | 30 | all (~38) | 46 (LOSO) |
|---|---|---|---|---|---|---|
| Test accuracy (%) | 70.90 | 75.11 | 77.00 | 78.68 | 80.25 | 80.25 |

Accuracy rises smoothly with diminishing gains and levels off. That is normal learning, not leakage or overfitting.

## Model-by-model LOSO results (trial level)

The table columns are:

* **Acc**: pooled trial accuracy.
* **BalAcc**: balanced accuracy.
* **F1**: macro F1.
* **κ**: Cohen's kappa.
* **p**: one-sided binomial test against chance.
* **Person**: per-person mean ± SD.
* **Window**: window-level accuracy.
* **Train**: trial accuracy on the training people.

### Action vs imagery (476 trials, chance 50%)

| Model | Correct | Acc | BalAcc | F1 | κ | p | Person | Window | Train |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 382 | 80.25 | 80.23 | 80.24 | 0.61 | 2e-42 | 80.5 ± 18.7 | 68.23 | 93.2 |
| LDA | 385 | 80.88 | 80.88 | 80.88 | 0.62 | 2e-44 | 80.6 ± 19.5 | 67.44 | 94.0 |
| Extra Trees | 371 | 77.94 | 77.92 | 77.92 | 0.56 | 4e-36 | 78.5 ± 18.9 | 68.33 | 100 |
| HistGB | 383 | 80.46 | 80.42 | 80.42 | 0.61 | 4e-43 | 80.7 ± 18.6 | 69.56 | 97.0 |
| **Ensemble** | **382** | **80.25** | 80.23 | 80.24 | 0.61 | 2e-42 | 81.3 ± 17.8 | 69.76 | 100 |

### Body part (476 trials, chance 16.67%)

| Model | Correct | Acc | BalAcc | F1 | κ | p | Person | Window | Train |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 121 | 25.42 | 25.09 | 24.95 | 0.11 | 8e-7 | 24.6 ± 12.4 | 21.58 | 91.8 |
| LDA | 117 | 24.58 | 24.34 | 24.29 | 0.10 | 7e-6 | 23.5 ± 11.2 | 21.18 | 91.2 |
| Extra Trees | 99 | 20.80 | 20.49 | 19.82 | 0.05 | 0.01 | 21.0 ± 11.3 | 19.61 | 100 |
| HistGB | 131 | 27.52 | 27.34 | 27.36 | 0.13 | 2e-9 | 27.9 ± 14.9 | 21.94 | 100 |
| **Ensemble** | **123** | **25.84** | 25.57 | 25.42 | 0.11 | 3e-7 | 25.3 ± 11.5 | 22.08 | 100 |

### Twelve tasks (476 trials, chance 8.33%)

| Model | Correct | Acc | BalAcc | F1 | κ | p | Person | Window | Train |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 101 | 21.22 | 20.87 | 20.53 | 0.14 | 3e-18 | 20.5 ± 13.1 | 14.74 | 97.9 |
| LDA | 97 | 20.38 | 20.01 | 19.76 | 0.13 | 2e-16 | 20.2 ± 14.4 | 14.21 | 95.9 |
| Extra Trees | 91 | 19.12 | 18.74 | 17.45 | 0.12 | 1e-13 | 18.9 ± 15.8 | 13.58 | 100 |
| HistGB | 90 | 18.91 | 18.70 | 18.19 | 0.12 | 3e-13 | 18.8 ± 13.0 | 15.09 | 100 |
| **Ensemble** | **106** | **22.27** | 21.88 | 21.63 | 0.15 | 1e-20 | 22.0 ± 14.7 | 14.90 | 100 |

### Left vs right imagery (241 trials, chance 50%)

| Model | Correct | Acc | BalAcc | F1 | κ | p | Person | Window | Train |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 147 | 61.00 | 61.10 | 61.00 | 0.22 | 4e-4 | 61.1 ± 23.5 | 53.82 | 95.0 |
| LDA | 137 | 56.85 | 56.91 | 56.85 | 0.14 | 0.02 | 56.4 ± 25.9 | 53.58 | 95.7 |
| Extra Trees | 131 | 54.36 | 54.11 | 54.01 | 0.08 | 0.10 | 54.5 ± 24.6 | 51.33 | 100 |
| HistGB | 122 | 50.62 | 50.44 | 50.40 | 0.01 | 0.45 | 51.8 ± 24.0 | 51.62 | 100 |
| **Ensemble** | **141** | **58.51** | 58.57 | 58.51 | 0.17 | 0.005 | 58.5 ± 24.6 | 53.48 | 100 |
| SVM (RBF) | 130 | 53.94 | 53.77 | 53.74 | 0.08 | 0.12 | 53.2 ± 23.3 | 53.09 | 100 |
| Random Forest | 125 | 51.87 | 51.86 | 51.85 | 0.04 | 0.30 | 50.5 ± 22.8 | 51.82 | 100 |
| AdaBoost | 128 | 53.11 | 53.12 | 53.10 | 0.06 | 0.18 | 51.5 ± 23.5 | 52.60 | 85.4 |
| Naive Bayes | 125 | 51.87 | 51.99 | 51.86 | 0.04 | 0.30 | 49.8 ± 25.7 | 51.72 | 69.2 |

Left vs right uses the imagery-only calibration, which was chosen after seeing results. Nested cross-validation, where that choice is made inside the training data, gives 52.28% (p = 0.26). Treat that as the honest estimate for this task.

## LOSO compared with 5-fold (ensemble, trial level)

| Task | 5-fold (3-fold for 6/12) | LOSO | Nested CV |
|---|---|---|---|
| Action vs imagery | 80.25% (382) | 80.25% (382) | 79.41% (378) |
| Body part | 26.68% (127) | 25.84% (123) | – |
| Twelve tasks | 21.43% (102) | 22.27% (106) | – |
| Left vs right imagery | 58.09% (140) | 58.51% (141) | 52.28% (126) |

Training on 46 instead of about 31–38 people changes accuracy by less than one point on every task. The results do not depend on the choice of cross-validation scheme.

## Overfitting

Training accuracy is far above test accuracy for every model: 91–100% on training people against 20–81% on new people. Tree models reach 100% on their training data by construction. This gap means the models fit individual people well and only part of what they learn transfers to a new person. It is not leakage: leakage raises test accuracy, and the shuffled-label tests (chance on all tasks) and the learning curve rule it out.

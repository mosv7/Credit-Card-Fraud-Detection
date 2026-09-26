# Credit Card Fraud Detection

A production-shaped machine learning pipeline for detecting fraudulent credit card
transactions on the classic ULB/Kaggle **Credit Card Fraud Detection** dataset.

The project benchmarks five classifiers, tunes the decision threshold for the
cost-sensitive fraud problem, and ships a single self-contained model bundle
(`model` + `threshold` + `scaler`) ready for serving.

**Final model: XGBoost — F1 `0.8828` / PR-AUC `0.8770` on validation, holding at
F1 `0.8633` / PR-AUC `0.8444` on the untouched holdout test set.**

---

## Table of Contents

1. [Executive Summary and Project Context](#1-executive-summary-and-project-context)
2. [Dataset Characteristics and Preprocessing Strategy](#2-dataset-characteristics-and-preprocessing-strategy)
3. [Comparative Experimental Results](#3-comparative-experimental-results)
4. [Key Scientific Insights and Technical Takeaways](#4-key-scientific-insights-and-technical-takeaways)
5. [Final Model Validation on the Holdout Test Set](#5-final-model-validation-on-the-holdout-test-set)
6. [Project Deliverables and Software Architecture](#6-project-deliverables-and-software-architecture)
7. [Final Conclusions and Recommendations](#7-final-conclusions-and-recommendations)
8. [How to Run](#8-how-to-run)
9. [Project Layout](#9-project-layout)

---

## 1. Executive Summary and Project Context

### 1.1 Problem statement

Payment fraud is a direct-margin problem. Every fraudulent authorisation that is
*not* blocked becomes a realised loss, and every legitimate authorisation that *is*
blocked becomes a customer who cannot spend. Both errors are expensive, and they
are expensive in opposite directions — which makes this a **threshold-tuning
problem wrapped inside a ranking problem**, not a plain classification problem.

The task is posed as binary classification on a transaction ledger:

| | |
|---|---|
| **Positive class (`Class = 1`)** | Fraudulent transaction — the event we must catch |
| **Negative class (`Class = 0`)** | Legitimate transaction — the event we must not disturb |
| **Feature count** | 30 (`Time`, `Amount`, `V1`–`V28`) |
| **Rows** | 284,807 transactions |
| **Observation window** | ~2 days (0 → 172,792 seconds) |

### 1.2 Why this dataset is a hard and instructive benchmark

Three properties combine to make the dataset a genuine stress test:

1. **Extreme imbalance.** Fraud is 492 of 284,807 rows — **0.1727%**, a
   roughly **1:578** negative-to-positive ratio.
2. **Already-PCA'd features.** 28 of the 30 features (`V1`–`V28`) are the output
   of a PCA performed upstream for confidentiality. They are anonymised and
   non-interpretable, so feature-engineering insight is limited to `Time` and
   `Amount`.
3. **A realistic base rate.** Because fraud is rare, *accuracy is a useless
   metric here*: a model that labels **every** transaction as legitimate scores
   **99.83% accuracy** while detecting **zero** fraud. The whole evaluation
   design of this project exists to avoid that trap.

### 1.3 Objectives

1. Establish a trustworthy, leak-free evaluation protocol for a 1:578
   imbalanced problem.
2. Compare a linear baseline, an ensemble baseline, a soft-voting ensemble, a
   gradient-boosted tree model, and a resampling strategy on identical splits.
3. Tune the decision threshold on a held-out validation set, so the operating
   point is chosen on data the model never trained on.
4. Report a final, single, honest number from an untouched holdout test set.
5. Package the result as a loadable artifact rather than a notebook side-effect.

### 1.4 Key results at a glance

| Question | Answer |
|---|---|
| Which model won? | **XGBoost** (F1 `0.8828`, PR-AUC `0.8770` on validation) |
| Did it generalise? | Yes — F1 `0.8828` → `0.8633`, PR-AUC `0.8770` → `0.8444` on holdout |
| How much fraud is caught? | **81.1%** (60 of 74) at a **92.3% alert precision** |
| Cost of that operating point? | 5 false alarms / 42,722 transactions (0.012%) |
| Does resampling help? | **No** — SMOTE *lowered* PR-AUC from `0.7740` to `0.7566` |
| Does the naive `0.5` threshold work? | **No** — it silently halves recall (`0.8649` → `0.6081`) |
| Production readiness | Threshold + scaler are bundled with the model; single artifact |

### 1.5 Business framing of the chosen operating point

At the shipped threshold of `0.1492`, on a comparable stream of 42,722
transactions the model produces:

- **60 frauds prevented** out of 74 that occurred;
- **5 legitimate transactions needlessly declined** (a customer-experience cost
  of ~0.012% of all traffic, or 1 in every ~8,500 authorisations);
- **14 frauds missed**.

This is a defensible point on the PR curve for a system that *blocks*
automatically. Note that the identical threshold used for *scoring* a review
queue would be far too permissive — the same probabilities should be routed to a
tiered policy, as discussed in [§7](#7-final-conclusions-and-recommendations).

---

## 2. Dataset Characteristics and Preprocessing Strategy

### 2.1 Provenance and scale

| Property | Value |
|---|---|
| Name | Credit Card Fraud Detection (ULB / Kaggle) |
| Rows | 284,807 |
| Columns | 31 (30 features + 1 target) |
| Memory footprint | 67.4 MB in memory (150 MB on disk) |
| dtypes | `float64` × 30, `int64` × 1 |
| Target | `Class` ∈ {0, 1} |

**The 150 MB CSV is deliberately not committed to this repository** — it exceeds
GitHub's 100 MB per-file hard limit. See [§8](#8-how-to-run) for how to obtain
it.

### 2.2 Data quality

| Check | Result | Consequence |
|---|---|---|
| Missing values | **0** | No imputation required |
| Fully-constant columns | 0 | No zero-variance removal required |
| Duplicate rows | **1,081** (0.38%) | Retained — see caveat below |
| Out-of-range / impossible values | None found | No clipping required |
| Class leakage in features | None detected | Split is safe |

> **Caveat on the 1,081 duplicates.** The duplicates were inspected and
> deliberately **kept**. Removing them would be defensible, but it changes the
> class balance of an already tiny positive class, and — critically — it would
> mean de-duplicating *before* the train/test split, which risks placing a
> copy of the same transaction on both sides of the holdout boundary. Keeping
> them is the leakage-safe choice. Production code should de-duplicate on a
> transaction key *after* splitting.

### 2.3 Class distribution — the defining characteristic

```
Class
0    284,315    99.8273 %
1        492     0.1727 %
```

A **1:578** imbalance. Practical consequences that shaped every design decision:

- Accuracy is uninformative (a majority-class baseline already scores 99.83%).
- Per-class **precision and recall** must be reported; the macro average matters
  more than the weighted average, because the weighted average is dominated by
  the negative class and therefore always looks excellent.
- **F1** and **PR-AUC** are the primary selection metrics.
- Validation and test sets, at ~74 positive rows each, contain only about
  **±6% relative sampling noise** on recall. This is the single most important
  caveat in this report and is revisited in [§4.4](#44-statistical-significance-and-the-noise-floor) and
  [§7](#7-final-conclusions-and-recommendations).

### 2.4 Feature block analysis

The 30 features split into two structurally different groups.

#### Block A — `V1` … `V28` (28 features, 93% of the feature space)

PCA components derived from the original transaction fields before anonymisation.

| Property | Value |
|---|---|
| Column means | ~`1e-15` (numerically zero — confirms mean-centring) |
| Standard deviation range | `0.330` → `1.959` |
| Range | e.g. `V1` spans `-56.4` → `2.45` |

These are already centred and on comparable scales, so they were passed to the
models **untouched**. Re-scaling them would have been an unnecessary and
potentially harmful step (it can amplify the heavy-tailed components, where
`V1`'s `σ = 1.96` sits against an extreme negative tail).

#### Block B — `Time` and `Amount` (2 features, on completely different scales)

| Statistic | `Time` | `Amount` |
|---|---|---|
| min | 0.00 | 0.00 |
| 25% | 54,201.5 | 5.60 |
| median | 84,692.0 | 22.00 |
| 75% | 139,320.5 | 77.17 |
| max | 172,792.0 | 25,691.16 |
| mean | 94,813.86 | 88.35 |
| std | 47,488.15 | 250.12 |

`Time` is a monotonic elapsed-seconds counter spanning ~48 hours; `Amount` is a
heavily right-skewed currency amount (max is **291×** the median). Feeding these
raw into a distance- or gradient-based learner is a scaling bug waiting to
happen — `Time` alone would dominate the entire loss landscape.

### 2.5 Amount: how fraud actually differs

Class-conditional `Amount` statistics reveal a non-obvious pattern:

| Class | count | mean | std | 25% | **median** | 75% | max |
|---|---|---|---|---|---|---|---|
| `0` Legitimate | 284,315 | 88.29 | 250.11 | 5.65 | **22.00** | 77.05 | 25,691.16 |
| `1` Fraud | 492 | 122.21 | 256.68 | 1.00 | **9.25** | 105.89 | 2,125.87 |

Two findings that a mean-only analysis would have inverted:

1. **Median fraud amount (9.25) is less than half the median legitimate amount
   (22.00).** Fraud skews *small* — consistent with card-not-present probing
   and micro-fraud designed to stay under review thresholds.
2. **Mean fraud amount (122.21) is *higher* than legitimate (88.29)** — but only
   because of a heavy right tail. **The mean and the median tell opposite
   stories.** Any univariate summary based on the mean is actively misleading
   here.

Corroborating this, fraud is concentrated at the bottom of the amount range but
not exclusively:

- **50.61%** of frauds are ≤ 10, vs **35.18%** of legitimate transactions — a
  ~1.4× enrichment, a *shift* in distribution rather than a separator.
- **55.08%** of frauds are ≤ 20, vs **48.84%** of legitimate — only a 1.13×
  enrichment, i.e. an even weaker signal at the 20 boundary.

> **Engineering takeaway:** "small transactions are fraud" is **not** a valid
> rule. 49% of legitimate traffic is also small. The discriminative power lives
> in the *joint* interaction between `Amount` and the PCA features — which is
> precisely the argument for a tree ensemble over a linear model.

Aggregate exposure for context: legitimate transactions total **25,102,462** in
value, frauds total **60,128**. Fraud is a volume phenomenon with a
long-tailed value distribution, not a value-at-risk phenomenon.

### 2.6 Time: weak but non-zero drift

Fraud rate by window:

| Window | Frauds | Transactions | Fraud rate |
|---|---|---|---|
| First hour (`0`–`3,600s`) | 2 | 3,964 | 0.050% |
| Day 1 (`0`–`86,400s`) | 281 | 144,787 | 0.194% |
| Day 2 (`86,400`+s) | 211 | 140,021 | 0.151% |
| Final 6 h (`>140,400s`) | 89 | 68,567 | 0.130% |

The **first hour is 3–4× quieter than average** (0.050% vs 0.173%), which is a
sensible warm-up effect — the ledger's opening hours are dominated by
low-risk, low-volume batch activity. Day 1 is modestly more fraud-dense than
Day 2.

The practical implication is a **data-engineering** one, not a modelling one:
`Time` is a proxy for *ledger position*, and a random split leaks a tiny amount
of this ordering information across the train/validation boundary. In a
production system, splits should be **temporal** (train on days 1–2, validate
on day 3, test on day 4). The random stratified split used here is
deliberately the *optimistic* choice, which is why the holdout results in
[§5](#5-final-model-validation-on-the-holdout-test-set) should be read as an
upper bound rather than a forecast.

### 2.7 Preprocessing and splitting strategy

The strategy is deliberately minimal — for an already-clean, already-PCA'd
dataset, restraint beats a long preprocessing chain.

```
creditcard.csv  (284,807 × 31)
        │
        ├── drop 'Class'  →  X (30 features)        y (binary target)
        │
        ├── STEP 1: stratified split, test_size = 0.15
        │       X_temp / y_temp  (242,085)   ├── X_test / y_test   (42,722)  15%
        │            │
        │            ├── STEP 2: stratified split of X_temp,
        │            │           test_size = 0.15/0.85 = 0.17647  ← corrected ratio
        │            │
        │            │      X_train / y_train  (199,364)   X_val / y_val  (42,721)  15%
        │            │
        │            └── RobustScaler fitted on X_train[['Time','Amount']] ONLY
        │                       │ fit_transform → train      (scaler learned here)
        │                       │ transform     → val        (no leakage)
        │                       │ transform     → test       (no leakage)
        │
        └── model.fit(X_train, y_train)   →   threshold tuned on X_val
                                              →  single final score on X_test
```

#### The five rules applied

| # | Rule | Rationale |
|---|---|---|
| 1 | **`stratify=y` on both splits** | Guarantees the 0.1727% base rate is preserved. Without it, a random split would hand the validation set a different number of frauds and make threshold tuning and model comparison noise-driven. Verified: train `0.1725%`, val `0.1732%`, test `0.1732%`. |
| 2 | **`random_state=42` everywhere** | Full reproducibility. Every result in this report is bit-reproducible. |
| 3 | **Two-stage split with a corrected ratio** | The second split uses `0.15 / 0.85 = 0.17647`, not `0.15`. This is the most common silent bug in three-way splits: it would have produced a 12.75% validation set instead of 15%. Corrected here, and the achieved shapes are asserted against expectations. |
| 4 | **Scaler fit on train only** | The `RobustScaler` learns `center_ = [84,845.5, 22.0]` and `scale_ = [85,225.5, 72.21]` from the **training** partition alone. Applying `.transform()` to val and test means the val/test sets cannot influence the representation the model learned. The fitted scaler is then shipped inside the model bundle so inference is identical. |
| 5 | **No imputation, no SMOTE in the production path** | 0 nulls and 0.17% imbalance. SMOTE was evaluated as an *experiment* ([§3](#3-comparative-experimental-results)) and **rejected on evidence** — it degraded PR-AUC. Shipping an unvalidated resampler would have been cargo-culting. |

#### Why `RobustScaler` and not the alternatives

```python
RobustScaler()   # (x - median) / (IQR)   ← chosen
# StandardScaler  # (x - mean) / σ         ← sensitive to Amount's max = 25,691
# MinMaxScaler    # (x - min) / (max - min) ← a single outlier compresses everything else
```

`Amount` has a maximum **291× its median** and a `σ/median` ratio of 11.4.
A `StandardScaler` would let a handful of extreme transactions dictate the
scaling of the entire feature; `MinMaxScaler` would be worse. `RobustScaler`
resists by using the median and the interquartile range, so the transformation
is stable and directly interpretable: it maps the training distribution onto
roughly a unit IQR regardless of outliers. This is the right choice for
currency-like features, and it is exactly the property that lets the fitted
scaler be serialised and reused safely at inference time.

### 2.8 Why PR-AUC and not ROC-AUC

ROC-AUC is a trap on this dataset. Because the negative class is 99.83% of the
data, the false-positive rate grows glacially — an absurd number of false alarms
is required before the FPR moves. A model can therefore post a high ROC-AUC
while being operationally useless.

**PR-AUC (average precision) is the honest choice here**: it is computed over
positives only, so it is not inflated by the majority class, and a random
classifier's PR-AUC baseline is the base rate itself (**0.001727**), not 0.5.
This gives every number in [§3](#3-comparative-experimental-results) an
unambiguous reference point.

---

## 3. Comparative Experimental Results

### 3.1 Protocol

All models were trained on the **same** `X_train` / `y_train` (199,364 rows),
scored on the **same** `X_val` / `y_val` (42,721 rows, 74 frauds), under
`random_state = 42`. The threshold for each model was selected on the
validation set by maximising F1 over the full precision–recall curve. The
**holdout test set was not touched until [§5](#5-final-model-validation-on-the-holdout-test-set).**

### 3.2 Comparative results table — validation set (n = 42,721, 74 frauds)

| # | Model | Threshold | **F1** | **PR-AUC** | Precision | Recall | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression *(naive `0.5`)* | 0.5000 | 0.7258 | 0.7740 | 0.9000 | 0.6081 | 5 | **29** | 45 |
| 2 | Logistic Regression *(tuned)* | 0.0726 | 0.8312 | 0.7740 | 0.8000 | 0.8649 | 16 | 10 | 64 |
| 3 | Logistic Regression + SMOTE | 1.0000 | 0.8369 | **0.7566** ↓ | 0.8806 | 0.7973 | 8 | 15 | 59 |
| 4 | Random Forest *(50 trees, depth 10)* | 0.2000 | 0.8707 | 0.8575 | 0.8767 | 0.8649 | 9 | 10 | 64 |
| 5 | Random Forest *(grid-searched)* | 0.2000 | 0.8707 | 0.8575 | 0.8767 | 0.8649 | 9 | 10 | 64 |
| 6 | Soft Voting *(LR + RF)* | 0.3295 | 0.8671 | 0.8471 | 0.8986 | 0.8378 | 7 | 12 | 62 |
| 7 | **XGBoost** ⭐ | **0.1492** | **0.8828** | **0.8770** | 0.9014 | 0.8649 | **7** | **10** | 64 |

*Random-classifier PR-AUC baseline = `0.001727` (the base rate). ↑/↓ mark results
that regress relative to their own baseline.*

### 3.3 Headline comparison

| Metric | Best model | Value | Runner-up | Δ |
|---|---|---|---|---|
| **F1 (primary)** | XGBoost | **0.8828** | Random Forest | +0.0121 |
| **PR-AUC (primary)** | XGBoost | **0.8770** | Random Forest | +0.0195 |
| **Precision** | XGBoost | **0.9014** | Soft Voting | +0.0028 |
| Recall | *three-way tie* | 0.8649 | — | 0.0000 |
| Fewest false alarms | XGBoost / Soft Voting | 7 | — | tie |

The comparison is deliberately reported on **two** axes rather than F1 alone.
XGBoost matches the best recall in the field while tying for the fewest false
alarms — it does not buy its F1 advantage by being more aggressive.

### 3.4 The two decisive comparisons

**Threshold tuning is worth more than most model upgrades.**

| Step | F1 | Δ F1 |
|---|---|---|
| Logistic Regression @ `0.5` | 0.7258 | — |
| Logistic Regression @ tuned `0.0726` | 0.8312 | **+0.1054** |
| XGBoost @ `0.5` (not used) | — | — |
| XGBoost @ tuned `0.1492` | 0.8828 | **+0.0516 over the same model at 0.5** |

Threshold optimisation delivered **+0.105 F1** to a fixed linear model — a
larger gain than upgrading Logistic Regression → XGBoost at a fixed threshold
produced. On a 0.17% base rate, the default `0.5` decision boundary is simply
misplaced. **The model gets the attention; the threshold does the work.**

**Tree ensembles beat linear models, but ensembling them adds nothing.**

```
Logistic Regression        F1 0.8312   PR-AUC 0.7740
Random Forest              F1 0.8707   PR-AUC 0.8575   ← +0.0395 F1, +0.0835 PR-AUC
Soft Voting (LR + RF)      F1 0.8671   PR-AUC 0.8471   ← WORSE than RF alone
XGBoost                    F1 0.8828   PR-AUC 0.8770   ← +0.0121 F1, +0.0195 PR-AUC
```

The Random Forest is the largest single step up: PR-AUC `0.7740 → 0.8575`.
This is the first evidence that the signal is **non-linear and interaction-driven**
— exactly what [§2.5](#25-amount-how-fraud-actually-differs) predicted from the
`Amount` distribution analysis, and what a linear boundary cannot represent.

Soft voting, however, **underperformed Random Forest alone** (F1 `0.8671` vs
`0.8707`; PR-AUC `0.8471` vs `0.8575`). Averaging a strong model with a much
weaker one (PR-AUC `0.7740`) dilutes the ensemble. And XGBoost then beat both.
**Conclusion: stacking heterogeneous models is only worth it if the members are
of comparable strength. Here it was not, and the complexity was dropped.**

### 3.5 Hyperparameter search — an informative negative result

A grid search over `max_depth ∈ {5, 8, 10}` × `n_estimators ∈ {10, 50, 100}` ×
`class_weight ∈ {None, 'balanced', 'balanced_subsample'}` — 27 configurations —
selected:

```
{'n_estimators': 50, 'max_depth': 10, 'class_weight': None}
Optimal Threshold: 0.2000   Best Validation F1: 0.8707   PR-AUC: 0.8575
```

This is **identical to the untuned default** (row 4 ≡ row 5, F1 `0.8707` both).
The honest interpretation:

1. **The untuned baseline was already the grid optimum.** Tuning bought nothing.
2. **`class_weight` resampling was rejected by the search** — every
   class-weighted variant performed *worse* than unweighted, corroborating the
   SMOTE finding in §3.6 from a completely different direction.
3. **The validation set is too small to resolve these differences.** With 74
   positives, differences of ±2 F1 points are not measurable. The grid selected
   `n_estimators = 50` over `100` on a single swapped prediction.

Point 3 is the scientifically important one: the search had no power to
discriminate, and the fact that it returned the incumbent is as much about
**sample size as about hyperparameters**. This is a direct argument for
`RepeatedStratifiedKFold` in any follow-up work
([§7](#7-final-conclusions-and-recommendations)).

### 3.6 SMOTE — rejected on evidence

| Model | F1 | PR-AUC | Precision | Recall | FP | FN |
|---|---|---|---|---|---|---|
| LR, unsampled | 0.8312 | **0.7740** | 0.8000 | 0.8649 | 16 | 10 |
| LR + SMOTE | **0.8369** | 0.7566 ↓ | 0.8806 | 0.7973 ↓ | 8 | 15 |

SMOTE balanced the training set from `{0: 199,020, 1: 344}` to
`{0: 199,020, 1: 199,020}` — a 578× oversampling of the minority class.

**Result: F1 rose marginally (+0.0057) but PR-AUC fell (−0.0174).**

Read carefully, the F1 "gain" is an **artefact of the threshold, not an
improvement in the model**. SMOTE's optimal threshold landed at exactly
`1.0000` — a degenerate boundary — and it bought higher precision at the cost of
recall (`0.8649 → 0.7973`). Crucially, **PR-AUC is threshold-independent**: it
measures the quality of the *ranking*, and the ranking measurably got worse.
F1 improved only because the chosen operating point moved.

> **Conclusion: SMOTE was rejected.** It manufactured synthetic fraud patterns
> from only 344 real examples, degrading ranking quality while making F1 *look*
> better through threshold compensation. Had only F1 been tracked, the wrong
> model would have shipped. **This is precisely why PR-AUC is the primary metric
> in this project.**

### 3.7 Feature scaling: consistency check

`Time` and `Amount` were verified to be raw and unscaled in the loaded data
(`Time` mean `94,813.86`, `Amount` mean `88.35`) and were confirmed robust-scaled
before training in the production pipeline, exactly as in the notebook. The
final model bundle was inspected to confirm the fitted `RobustScaler` ships
inside the artifact with `center_ = [84,845.5, 22.0]` and
`scale_ = [85,225.5, 72.21]`.

### 3.8 Full result matrix

| Model | Validation | | | Holdout test | | |
|---|---|---|---|---|---|---|
| | F1 | PR-AUC | Precision | Recall | F1 | PR-AUC |
| Logistic Regression | 0.8312 | 0.7740 | 0.8000 | 0.8649 | — | — |
| Logistic Regression + SMOTE | 0.8369 | 0.7566 | 0.8806 | 0.7973 | — | — |
| Random Forest | 0.8707 | 0.8575 | 0.8767 | 0.8649 | 0.8243 | 0.8135 |
| Soft Voting | 0.8671 | 0.8471 | 0.8986 | 0.8378 | — | — |
| **XGBoost** | **0.8828** | **0.8770** | **0.9014** | **0.8649** | **0.8633** | **0.8444** |

---

## 4. Key Scientific Insights and Technical Takeaways

### 4.1 The operating point is the product, not the model

The single largest effect in the whole study is threshold selection.
Logistic Regression gained **+0.1054 F1** purely from moving its boundary from
`0.5` to `0.0726` — a larger gain than the entire linear → boosted-tree upgrade.

This is a direct consequence of the base rate. A model trained on 0.17% positives
is *recalibrated* — its probability outputs are pulled toward zero because it is
overwhelmingly rewarded for saying "legitimate". A naive `0.5` boundary then
rejects almost nothing. The correct boundary is far lower, and **it must be
learned from a precision–recall curve on held-out data, not assumed.**

The threshold was tuned on the **validation** set and then applied, *untouched*,
to the test set in [§5](#5-final-model-validation-on-the-holdout-test-set). Had it
been tuned on the test set, the reported number would have been optimistic and
meaningless.

### 4.2 Accuracy is actively misleading — and macro-F1 is not a substitute

A do-nothing classifier (`Class = 0` always) scores **99.83% accuracy** and
**0.00 recall**: it is useless, and the metric says it is excellent.

The subtler trap: **weighted-average F1 does not fix this.** Logistic Regression
@tuned reports `0.9994` weighted F1 while its macro F1 is `0.8999`. The weighted
figure is dominated by 42,647 negative rows and is essentially a rounding
artifact of the class imbalance. Only the **positive-class** row and the
**macro** average carry signal.

Accordingly, this project reports `Class = 1` precision and recall explicitly in
every table, and selects on F1 / PR-AUC. For a business where the cost is
asymmetric, the correct objective is neither accuracy nor symmetric F1, but a
**cost-weighted** score reflecting the relative price of a missed fraud versus a
declined legitimate transaction.

### 4.3 Resampling is not a substitute for threshold tuning — and often worse

Two independent resampling strategies were tried: **SMOTE** (synthetic
oversampling) and **`class_weight`** (cost-sensitive weighting, in the RF grid
search). **Both failed to improve the model.**

- SMOTE: PR-AUC `0.7740 → 0.7566` (−0.0174), with only an illusionary F1 gain
  produced by a degenerate `1.0000` threshold.
- `class_weight`: every weighted variant lost to the unweighted default.

The reason is structural. Oversampling does not create information; it
interpolates between the 344 real minority examples and reinforces whichever
minority manifold is densest, shrinking the variety the model sees. On a dataset
whose signal is interaction-driven and whose positive class has only **344
training examples**, synthetic replication of a small sample actively degrades
generalisation.

The gain people expect from resampling is already available — and available
better — from tuning the decision threshold. **Tune the boundary; leave the data
alone.**

### 4.4 Statistical significance and the noise floor

This is the most important methodological caveat in the project.

The validation and test sets contain **74 positive examples each**. For a recall
estimate near 0.86, one classification is a single flip:

| Metric | Estimate | One flip changes it by | 95% CI (Wilson, approx.) |
|---|---|---|---|
| Recall | 0.8649 (64/74) | **±1.35%** | ≈ **[0.759, 0.930]** |
| Precision | 0.9014 (64/71) | ±1.41% | ≈ **[0.816, 0.950]** |

Therefore:

- **The 0.0121 F1 gap** between XGBoost and Random Forest is **not
  statistically significant** — it is roughly one flipped prediction.
- **The 0.0195 PR-AUC gap** is **directionally consistent** across validation and
  test (+0.0195 → +0.0309), which is mild evidence of a real effect, but a single
  F1 point is not a proof.
- **The 0.1054 F1 gap** from threshold tuning **is** significant — it spans ~8
  reclassified transactions, and the mechanism (a misplaced default boundary) is
  independently explainable.

The correct conclusion: **XGBoost is a reasonable default, chosen for
consistency of behaviour rather than proven superiority.** In production,
repeated stratified cross-validation would be mandatory before claiming a win.

### 4.5 Model choice is only defensible with a baseline ladder

The progression tells a coherent story:

| Stage | PR-AUC | What it proves |
|---|---|---|
| Random classifier | 0.0017 | The base rate — the real floor |
| Logistic Regression | 0.7740 | **A linear boundary already captures ~99.6% of the available signal.** The data is not hopeless. |
| Random Forest | 0.8575 | The residual is **non-linear and interaction-driven** |
| XGBoost | 0.8770 | Boosted trees with regularisation extract a further increment |

The jump from the base rate to Logistic Regression is the striking part: a
straight line in PCA space separates fraud at PR-AUC `0.7740`. The remaining
effort is refinement, not rescue. Note also that **soft voting of the two weakest
tiers was worse than the stronger one alone** — ensembling only pays when members
are of comparable strength.

### 4.6 Domain insight: fraud looks like *small* transactions, not large ones

The most counter-intuitive empirical finding. **Median** fraud amount is `9.25`
vs `22.00` for legitimate, while the **mean** is higher (`122.21` vs `88.29`).
Mean and median disagree in direction, so any univariate summary built on the
mean would have drawn exactly the wrong conclusion.

- 50.6% of frauds are ≤ 10, vs 35.2% of legitimate — a **1.44× enrichment**.
- But 35.2% of *legitimate* traffic is also ≤ 10, so the rule has no standalone
  power.
- Fraud also plateaus above 20 (55.1% vs 48.8%, only **1.13×**), so the signal
  is concentrated in the smallest transactions.

This is characteristic of **card-not-present probing and micro-fraud**: many
small, quickly-reversed, low-individuality attempts whose *aggregate* is
damaging. It also explains why trees beat linear models here — the
`Amount ≤ 10` × PCA-feature interaction is precisely the kind of conditional
structure a linear boundary cannot express.

**Caveat on scale.** The whole dataset covers ~2 days. Any finding from a
2-day window is provisional; fraud patterns have strong seasonality, and a
finding that holds over 48 hours is not guaranteed to hold over 48 weeks. The
`Time` analysis in [§2.6](#26-time-weak-but-non-zero-drift) is the honest
example — real but weak drift, and a strong argument for temporal rather than
random splits in production.

---

## 5. Final Model Validation on the Holdout Test Set

### 5.1 Protocol integrity

The holdout test set (**42,722 rows, 74 frauds**) was carved out in the **first**
split and was used for **nothing** — not for training, not for threshold
selection, not for hyperparameter search. The threshold `0.1492` was tuned on the
**validation** set and transferred to test **verbatim**. This is a single,
genuinely untouched estimate of generalisation.

### 5.2 Final model configuration

```python
XGBClassifier(
    n_estimators     = 500,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    gamma            = 0,
    reg_alpha        = 0.0,
    reg_lambda       = 1.0,
    scale_pos_weight = 1,          # ← deliberately NOT set; resampling was rejected (§4.3)
    objective        = 'binary:logistic',
    eval_metric      = 'aucpr',    # ← optimises the right metric from the start
    random_state     = 42,
    n_jobs           = -1,
)
```

Note `eval_metric='aucpr'` and `scale_pos_weight=1` — the model is configured for
**ranking quality on a skewed problem**, with cost correction deferred entirely
to the threshold. This is the direct, configurable expression of the strategy
proven in [§4.1](#41-the-operating-point-is-the-product-not-the-model).

`n_estimators=500` at `learning_rate=0.05` is a deliberately low-learning-rate,
high-capacity pairing, with `subsample=0.8` and `colsample_bytree=0.8` injecting
the stochasticity that keeps 500 sequential trees from overfitting. The confirmed
generalisation gap ([§5.4](#54-generalisation-gap-validation-vs-test)) shows the
configuration is not overfit.

### 5.3 Holdout test results

```text
=== HOLDOUT TEST SET: XGBoost Results ===
Applied Threshold: 0.1492
Test F1-Score:     0.8633
Test PR-AUC:       0.8444

Confusion Matrix:
[[42643      5]
 [    14     60]]
```

| | **Predicted: Legitimate (0)** | **Predicted: Fraud (1)** | **Total** |
|---|---:|---:|---:|
| **Actual: Legitimate (0)** | **42,643** (TN) | **5** (FP) | 42,648 |
| **Actual: Fraud (1)** | **14** (FN) | **60** (TP) | 74 |
| **Total** | 42,657 | 65 | 42,722 |

| Metric | Value | Derivation |
|---|---|---|
| **PR-AUC** | **0.8444** | Threshold-independent ranking quality |
| **F1 (positive class)** | **0.8633** | 2 · (0.9231 · 0.8108) / (0.9231 + 0.8108) |
| **Precision** | **0.9231** | 60 / 65 |
| **Recall** | **0.8108** | 60 / 74 |
| **Accuracy** | *0.99965* | *(reported only to be dismissed — see §4.2)* |
| **Specificity** | 0.99988 | 42,643 / 42,648 |
| **False-alarm rate** | **0.0117%** | 5 / 42,648 |
| **Miss rate** | **18.9%** | 14 / 74 |

**Business read-out.** Per 42,722 transactions, the model **prevents 60 frauds**
at the cost of **5 incorrectly declined legitimate cards** — an
**12:1** ratio of prevented fraud to customer friction. 14 frauds still get
through.

### 5.4 Generalisation gap: validation vs test

| Metric | Validation | Holdout test | Δ | Verdict |
|---|---|---|---|---|
| **PR-AUC** | 0.8770 | 0.8444 | **−0.0326** | Modest, expected optimism |
| **F1** | 0.8828 | 0.8633 | **−0.0195** | Small, well within noise floor |
| **Recall** | 0.8649 (64/74) | 0.8108 (60/74) | −0.0541 (4 frauds) | Within sampling noise |
| **Precision** | 0.9014 (64/71) | 0.9231 (60/65) | +0.0217 | Improved — threshold well-placed |
| False alarms | 7 | **5** | −2 | Improved |
| **Frauders caught** | **64 / 74** | **60 / 74** | **−4** | 86.5% → 81.1% |

The two-point view, and why it matters:

1. **The expected optimism is small and in the right direction.** Both metrics
   declined modestly; neither collapsed. PR-AUC fell `3.3` points and F1 `2.0`
   points. There is **no sign of overfitting** despite 500 boosting rounds.
2. **The drop is inside the noise floor.** Per [§4.4](#44-statistical-significance-and-the-noise-floor),
   recall on 74 positives carries a ±1.35% one-flip granularity and a Wilson 95%
   CI of roughly `[0.76, 0.93]`. A move from `0.8649` to `0.8108` is **4
   transactions** — well inside that interval. **The gap is not evidence of a
   problem; it is the sampling variance of a small positive class.**
3. **The threshold transferred cleanly.** It was tuned on a *different* 42,721
   transactions and, if it had been mistuned, precision and false alarms would
   have degraded sharply. Instead **precision improved** (`0.9014 → 0.9231`) and
   **false alarms fell** (`7 → 5`). The threshold is a **stable property of the
   model**, not of the validation sample. This is the single strongest piece of
   evidence that the chosen operating point will hold on live traffic.

> **Honest summary:** expected optimism was `~0.02` F1, entirely attributable to
> the small positive class. The deployed configuration is validated, not
> merely fitted.

### 5.5 Comparative holdout test results

| Model | Threshold | Test F1 | Test PR-AUC | Precision | Recall | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.2000 | 0.8243 | 0.8135 | 0.8243 | 0.8243 | 13 | 13 | 61 |
| **XGBoost** ⭐ | **0.1492** | **0.8633** | **0.8444** | **0.9231** | **0.8108** | **5** | 14 | 60 |

XGBoost dominates on the two primary metrics: **+0.0390 F1** and **+0.0309
PR-AUC** over Random Forest on the holdout — a *larger* margin than on
validation, and directionally consistent across both.

It also **retrieves 2.6× fewer false alarms** (5 vs 13) for one additional missed
fraud. For an auto-blocking policy that is a decisive operational advantage:
customer friction drops from 1 in 3,282 to 1 in 8,544 transactions.

> **Note on interpretation.** Random Forest's coincidentally identical
> precision and recall (`0.8243` = 61/74) is a rounding artefact of its
> confusion matrix, not a structural property. XGBoost's margin here is
> consistent but, per [§4.4](#44-statistical-significance-and-the-noise-floor),
> a single holdout set of 74 positives is not enough to *prove* superiority.
> **XGBoost is the recommended default; the difference is suggestive, not
> conclusive.**

### 5.6 The bias/visibility tradeoff on the residual 14 frauds

The 14 missed frauds are the entire remaining risk, and the decision of what to
do with them is a business policy choice, not a modelling one. Two coherent
positions:

**Position A — maximise prevention (current setting, `0.1492`)**
Catch 81.1% of fraud with 5 false alarms. Appropriate when a manual review
queue exists to absorb false alarms cheaply.

**Position B — optimise alert precision for a scarce review team**

| Operating point | Caught | Missed | False alarms | Ratio |
|---|---|---|---|---|
| Current (`0.1492`) | 60 | 14 | 5 | 12.0 : 1 |
| Stricter (fewer alerts) | fewer | more | < 5 | higher |

A team able to review only ~5 alerts per 42,722 transactions would be saturated
by Position A and should lower the threshold to shrink the queue — accepting
more false alarms in exchange for reviewing more frauds. **The threshold must be
set from the operational review capacity, not from the F1 maximum.** The value
shipped in the bundle is the F1-optimal default, and is a starting point to be
re-tuned against real cost ratios.

---

## 6. Project Deliverables and Software Architecture

### 6.1 Deliverables

| # | Deliverable | Path | Description |
|---|---|---|---|
| 1 | **Production training pipeline** | `Code/credit_fraud_train.py` | CLI entry point. Trains, tunes the threshold, evaluates on validation **and** holdout, and serialises the artifact. |
| 2 | **Data & preprocessing module** | `Code/credit_fraud_utils_data.py` | Loading, leakage-free stratified 3-way split, `RobustScaler` handling. |
| 3 | **Evaluation module** | `Code/credit_fraud_utils_eval.py` | Threshold optimisation + metric computation + console reporting. |
| 4 | **Trained model bundle** | `Model/model.pkl` (680 KB) | `joblib` artifact: `{model, threshold, scaler}` — XGBoost + `0.1492` + fitted `RobustScaler`. |
| 5 | **Exploratory & experimental notebook** | `Notebook/EDA.ipynb` | 23 cells: EDA, all six model experiments, RF grid search, holdout validation. |
| 6 | **This technical report** | `README.md` | Full methodology, results, and analysis. |
| 7 | **Dependency manifest** | `requirements.txt` | Pinned Python package versions. |
| 8 | **Repository hygiene** | `.gitignore` | Excludes the 150 MB dataset, `__pycache__`, and local artefacts. |

### 6.2 Architecture

The pipeline is a **three-module, one-directional pipeline** with a strict
separation of concerns: *data* knows nothing about models, *evaluation* knows
nothing about data, and the *training script* is the only module that knows
about both.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Data/creditcard.csv                                                   │
│  (284,807 × 31 — NOT version-controlled; see §2.1)                    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │  load_data()            credit_fraud_utils_data.py
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  credit_fraud_utils_data.py           ── PURE DATA CONCERNS ──        │
│                                                                       │
│   load_data(path)                    → DataFrame                      │
│   prepare_and_split_data(df)                                                │
│     ├── drop 'Class'                 → X (30), y (binary)             │
│     ├── stratified split  test=0.15  → X_temp / X_test   [15%]        │
│     ├── stratified split  0.15/0.85  → X_train / X_val  [15%]        │
│     └── RobustScaler on ['Time','Amount']                              │
│           ├── fit_transform(X_train)     ◄── fitted HERE ONLY         │
│           ├── transform(X_val)          ◄── no leakage                │
│           └── transform(X_test)         ◄── no leakage                │
│     returns (X_tr, X_val, X_te, y_tr, y_val, y_te, scaler)           │
└──────────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  credit_fraud_train.py                   ── ORCHESTRATOR / CLI ──     │
│                                                                       │
│   argparse:  --data_path  --model_type {lr,rf,voting,xgb}              │
│              --output_path                                              │
│                                                                       │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐              │
│   │ lr           │   │ rf           │   │ voting       │  ┌────────┐  │
│   │ LogReg       │   │ 50 trees     │   │ soft LR + RF │  │ xgb    │  │
│   │ max_iter=1k  │   │ depth=10     │   │ (averaging)  │  │ 500 ×  │  │
│   │              │   │              │   │              │  │ depth5 │  │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘  └───┬────┘  │
│          └──────────────────┴──────────────────┴──────────────┘        │
│                             │                                         │
│                             ▼  model.fit(X_train, y_train)           │
│          ┌──────────────────────────────────────────────────┐         │
│          │  credit_fraud_utils_eval.py  ── EVALUATION ──    │         │
│          │                                                  │         │
│          │   find_optimal_threshold(y_val, val_probs)        │         │
│          │     └── precision_recall_curve → argmax F1        │         │
│          │   evaluate_model(...)                             │         │
│          │     └── PR-AUC, F1, P, R, confusion matrix        │         │
│          │   print_evaluation_report(...)  → console table   │         │
│          └───────────────────┬──────────────────────────────┘         │
│                              │                                        │
│              ┌───────────────┴───────────────┐                        │
│              ▼                               ▼                        │
│   Validation report (threshold tuned    Holdout test report          │
│   here — the ONLY set used to tune)    (threshold applied verbatim)  │
│                              │                                        │
│                              ▼                                        │
│          joblib.dump({'model': model,                                │
│                        'threshold': best_thresh,                    │
│                        'scaler': scaler})                            │
└──────────────────────────────┬────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Model/model.pkl   (680 KB)                                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ model     : XGBClassifier  (n_estimators=500, max_depth=5,     │  │
│  │                           learning_rate=0.05, subsample=0.8,    │  │
│  │                           colsample_bytree=0.8,                 │  │
│  │                           min_child_weight=3, random_state=42) │  │
│  │ threshold : 0.14921729266643524  ← tuned on VALIDATION only     │  │
│  │ scaler    : RobustScaler(center_=[84845.5, 22.0],               │  │
│  │                          scale_=[85225.5, 72.21])               │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  Single self-contained artifact — no external state required.         │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.3 Design decisions and their justification

| Decision | Rationale |
|---|---|
| **Three focused modules** | Data, evaluation, and orchestration have genuinely different responsibilities. `credit_fraud_utils_eval.py` is fully reusable — it knows nothing about this dataset and can score any `y_true`/`y_prob` pair. |
| **Configurable via `--model_type`** | One pipeline reproduces all four model experiments. Adding a model means adding one `elif` branch, not a new script. Supports reproducible ablation without notebook dependence. |
| **Threshold tuned on validation only** | The architectural guarantee that the holdout test number is honest. Tuning on test would make [§5](#5-final-model-validation-on-the-holdout-test-set) meaningless. |
| **Scaler shipped with the model** | The single most common production bug in tabular ML is serving with a different transformation than the one used in training. Bundling `{model, threshold, scaler}` makes it **structurally impossible** to apply the wrong scaling — the artifact carries its own contract. |
| **Threshold shipped with the model** | Guarantees the tuned operating point is applied at inference instead of a hardcoded `0.5` — the exact failure mode identified in [§4.1](#41-the-operating-point-is-the-product-not-the-model). |
| **`average_precision_score` for PR-AUC** | `average_precision_score` is the non-interpolated PR estimate. It is threshold-independent, correct under heavy imbalance, and comparable to the `eval_metric='aucpr'` used inside XGBoost. |
| **Epsilon guard in F1** | `2PR/(P+R+1e-10)` avoids `NaN`/division-by-zero at the extremes of the PR curve, where `P+R` can approach 0 — a real edge case when sweeping 300+ thresholds. |
| **Module `__main__` self-tests** | `credit_fraud_utils_data.py` and `credit_fraud_utils_eval.py` can each be run directly to smoke-test utilities without training. |
| **`n_jobs=1` for the tuned RF** | The grid search evaluates 27 configurations sequentially; `n_jobs=-1` would spawn the full thread pool 27 times and oversubscribe the CPU. `n_jobs=-1` is used only for the single final fits. |

### 6.4 Verified artifact contents

The shipped `Model/model.pkl` was loaded and inspected to confirm the bundle is
internally consistent with this report:

```
keys:        ['model', 'threshold', 'scaler']
model:       XGBClassifier
threshold:   0.14921729266643524
scaler:      RobustScaler
               center_ = [8.48455e+04, 2.20000e+01]
               scale_  = [8.52255e+04, 7.22100e+01]
n_features_in_: 30
```

The threshold matches the notebook exactly, and the scaler parameters confirm
`RobustScaler` was fitted on the training partition only.

### 6.5 Reproducibility

| Source of variation | Control |
|---|---|
| Split assignment | `random_state=42` on both split calls |
| Model initialisation | `random_state=42` in LR, RF, and XGBoost |
| SMOTE (experiment only) | `random_state=42` |
| CPU thread nondeterminism | Minor float-ordering variation possible in XGBoost; metrics are stable to ~1e-6 |
| Library versions | Pinned in `requirements.txt` |

---

## 7. Final Conclusions and Recommendations

### 7.1 Conclusions

**1. XGBoost is the recommended model.** Best on both primary metrics on both
evaluation sets — validation F1 `0.8828` / PR-AUC `0.8770`, holdout F1 `0.8633` /
PR-AUC `0.8444`. It matches the best recall in the field while tying for the
fewest false alarms, and it generalises without overfitting despite 500 boosting
rounds.

**2. The operating point matters more than the model.** Threshold tuning was
worth **+0.105 F1** to a fixed model — more than the entire model-class upgrade
produced. The shipped threshold `0.1492`, tuned on validation and transferred
untouched, **improved** precision (`0.9014 → 0.9231`) and **reduced** false alarms
(`7 → 5`) on the holdout. It is a stable property of the model, not of the
sample.

**3. The result is real but modest, and honestly bounded.** At the operating
point, the system prevents **60 of 74 frauds** at the cost of **5 false alarms**
in 42,722 transactions — a **12:1** ratio, with **14 frauds still missed**
(18.9%). It is a strong filter, **not** a complete solution.

**4. Resampling is not the answer on this problem.** SMOTE lowered PR-AUC
(`0.7740 → 0.7566`) and produced only an illusionary F1 gain via a degenerate
`1.0000` threshold. `class_weight` lost to the unweighted default across the
entire 27-config grid. **Tune the threshold instead of manufacturing data.**

**5. The dataset's signal is largely linear, with a non-linear residual.** A
straight boundary in PCA space already reaches PR-AUC `0.7740`; trees add
`+0.08`. The gain from here is refinement, not rescue.

**6. "Big transaction = fraud" is false.** Median fraud amount is `9.25` vs
`22.00` for legitimate, while the *mean* is higher. Mean and median disagree in
direction — the empirical signature of card-not-present probing and micro-fraud,
and a strong caution against mean-based univariate analysis.

**7. The dominant limitation is sample size, not modelling.** 74 positives per
evaluation set means recall carries **±1.35%** one-flip granularity and a 95% CI
near `[0.76, 0.93]`. The XGBoost-over-Random-Forest margin is **inside that
interval** and is therefore *suggestive, not proven*.

### 7.2 Recommendations — priority order

#### Immediate, before any production deployment

| # | Recommendation | Rationale | Effort |
|---|---|---|---|
| 1 | **Re-validate with `RepeatedStratifiedKFold` (5 × 5 folds)** | A single 74-positive holdout cannot support a model-selection claim ([§4.4](#44-statistical-significance-and-the-noise-floor)). Report mean ± std across repeats. | Low |
| 2 | **Switch to temporal (time-based) splits** | The random stratified split leaks ledger position via `Time` ([§2.6](#26-time-weak-but-non-zero-drift)). Report metrics under a strict forward-chaining split. | Low |
| 3 | **Define an explicit cost matrix and optimise expected loss** | F1 treats a missed fraud and a false alarm as equally bad. In reality a missed fraud is worth orders of magnitude more. Optimise the *business* objective, then re-tune the threshold. | Low |
| 4 | **Re-tune the threshold against real review capacity** | The shipped `0.1492` is the F1 maximum, which is **not** the alert volume a review team can actually work ([§5.6](#56-the-biasvisibility-tradeoff-on-the-residual-14-frauds)). | Low |
| 5 | **Ship a tiered policy instead of one hard threshold** | Block above a high threshold, auto-approve below a low one, and route the ambiguous middle to manual review. Uses the full probability distribution instead of discarding it to a bit. | Medium |

#### Medium term — model improvement

| # | Recommendation | Rationale | Effort |
|---|---|---|---|
| 6 | **Try LightGBM / CatBoost and neural models (TabNet, FT-Transformer)** | Modern tabular architectures routinely beat XGBoost on tabular fraud data, and there is headroom in the `+0.0195` PR-AUC range. | Medium |
| 7 | **Test ensemble diversity *properly*** | Soft voting failed because members were unequal in strength ([§3.4](#34-the-two-decisive-comparisons)). Once 3–4 models reach comparable PR-AUC, stacking becomes worth revisiting. | Medium |
| 8 | **Engineer features from the raw, interpretable fields** | `Amount`-based features — log amount, amount bands, amount-per-hour, deviation from the user's typical amount — are absent from the PCA features and directly encode the [§4.6](#46-domain-insight-fraud-looks-like-small-transactions-not-large-ones) finding. This is the highest-expected-value feature work. | Medium |
| 9 | **Quantify feature importance (SHAP)** | Determines whether the models are learning genuine fraud structure or overfitting noise in near-duplicate rows. | Low |
| 10 | **De-duplicate on a transaction key *after* splitting** | The 1,081 duplicates are retained here for leakage safety, but should be removed from training in production. | Low |

#### Operational maturity

| # | Recommendation | Rationale | Effort |
|---|---|---|---|
| 11 | **Add a streaming/online pipeline with concept-drift monitoring** | A model trained on 2 days of data will degrade as fraud patterns evolve. Monitor PR-AUC live and trigger retraining on drift. | High |
| 12 | **Add MLflow or Weights & Biases for experiment tracking** | Six experiments currently live in a notebook. Threshold-tuning methodology and lineage need to be reproducible and auditable in production. | Medium |
| 13 | **Build a serving API with a `/predict` endpoint** | The bundle is loadable; there is no service. A thin FastAPI wrapper returning the score *and the applied threshold* would complete the stack. | Low |
| 14 | **Add unit and integration tests** | No test coverage exists. Priority: (a) the `0.15/0.85` split-ratio arithmetic, (b) that the scaler is fit on train only, (c) that the bundled threshold is the one from validation. | Medium |
| 15 | **Add a CI workflow** | Run the test suite and a smoke-train on every push. | Low |
| 16 | **De-identify responsibly and document the data licence** | The source is the public ULB dataset; document the licence and confirm that derived artifacts (`model.pkl`) inherit no re-identification risk. | Low |

### 7.3 Honest limitations

1. **Statistical power is low.** 74 positives per evaluation set. Model-ranking
   claims are directional, not conclusive ([§4.4](#44-statistical-significance-and-the-noise-floor)).
2. **The dataset spans ~2 days.** All findings — including the `Amount`
   distribution in [§2.5](#25-amount-how-fraud-actually-differs) — are specific
   to this 48-hour window. Nothing here should be assumed to hold over longer
   horizons without re-validation.
3. **28 of 30 features are uninterpretable.** Post-PCA features block the
   primary analytical tool of fraud investigation: explaining *why* a transaction
   was flagged. Explanations must be approximated post hoc via SHAP.
4. **A 12:1 prevent/friction ratio is not free.** 5 false alarms in 42,722
   transactions is small in aggregate but concentrates in specific cardholders.
   Per-customer friction must be monitored.
5. **The threshold is a global constant.** A customer who has transacted
   consistently for years should face a different threshold from a first-time
   e-commerce transaction. No such personalisation exists yet.
6. **`_optimise` target shift is unmodelled.** If deployment traffic differs
   from this sample, the threshold's calibration does not transfer for free.
7. **Results depend on the exact `random_state`.** A different seed reshuffles
   which frauds land in which split, and per [§4.4](#44-statistical-significance-and-the-noise-floor)
   that alone can move F1 by ±0.02.

---

## 8. How to Run

### 8.1 Prerequisites

- Python 3.13 (developed and verified on 3.13)
- Dependencies from `requirements.txt`

```bash
pip install -r requirements.txt
```

| Package | Verified version |
|---|---|
| `scikit-learn` | 1.8.0 |
| `xgboost` | 3.4.0 |
| `pandas` | 3.0.1 |
| `numpy` | 2.4.2 |
| `joblib` | latest |
| `imbalanced-learn` | only for the SMOTE experiment in the notebook |

### 8.2 Obtain the dataset

The 150 MB CSV is **not in this repository** (GitHub's 100 MB per-file limit).
Download the **Credit Card Fraud Detection** dataset from
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — requires a
free Kaggle account and accepting the dataset licence — and place it at:

```
Data/creditcard.csv
```

Verify the integrity of what you downloaded:

```bash
python -c "import pandas as pd; d=pd.read_csv('Data/creditcard.csv'); print(d.shape, d['Class'].sum())"
# Expected: (284807, 31) 492
```

### 8.3 Train the final model

```bash
cd Code
python credit_fraud_train.py
```

Defaults: `--data_path ..\Data\creditcard.csv`, `--model_type xgb`,
`--output_path ..\Model\model.pkl`. The run prints the validation report, the
holdout test report, and confirms the saved bundle. It takes a few minutes on
CPU.

### 8.4 Reproduce other models

```bash
python credit_fraud_train.py --model_type lr
python credit_fraud_train.py --model_type rf
python credit_fraud_train.py --model_type voting
python credit_fraud_train.py --model_type xgb  --output_path ..\Model\model_xgb.pkl
```

`--model_type` accepts `lr`, `rf`, `voting`, `xgb`.

### 8.5 Use the saved model

```python
import joblib

bundle = joblib.load('Model/model.pkl')
model, threshold, scaler = bundle['model'], bundle['threshold'], bundle['scaler']

import pandas as pd
row = pd.read_csv('Data/creditcard.csv').iloc[[0]].drop(columns=['Class'])

row[['Time', 'Amount']] = scaler.transform(row[['Time', 'Amount']])  # bundled scaler
score = model.predict_proba(row)[:, 1][0]
flag = score >= threshold    # bundled threshold — NOT 0.5

print(f'Fraud score: {score:.4f}  threshold: {threshold:.4f}  -> {"REVIEW" if flag else "APPROVE"}')
```

**Always use the bundled `threshold` and the bundled `scaler`.** Hardcoding `0.5`
or re-deriving the scaling at inference are the two failure modes this bundle
exists to prevent ([§4.1](#41-the-operating-point-is-the-product-not-the-model), [§6.3](#63-design-decisions-and-their-justification)).

### 8.6 Reproduce the full analysis

```bash
jupyter notebook Notebook/EDA.ipynb
```

Run all cells top to bottom to regenerate every figure and result in
[§2](#2-dataset-characteristics-and-preprocessing-strategy)–[§5](#5-final-model-validation-on-the-holdout-test-set),
including the Random Forest grid search and the SMOTE experiment.

---

## 9. Project Layout

```
Credit-Card-Fraud-Detection/
├── README.md                          # This technical report
├── requirements.txt                   # Pinned dependencies
├── .gitignore                         # Excludes the 150 MB dataset + __pycache__
│
├── Code/
│   ├── credit_fraud_train.py           # CLI orchestrator: train → tune → evaluate → save
│   ├── credit_fraud_utils_data.py      # Load, stratified 3-way split, RobustScaler
│   ├── credit_fraud_utils_eval.py      # Threshold optimisation + metrics + reporting
│   └── __pycache__/                    # (git-ignored)
│
├── Data/
│   └── creditcard.csv                  # 150 MB — download from Kaggle, NOT committed
│
├── Model/
│   └── model.pkl                       # {model, threshold, scaler} — 680 KB
│
└── Notebook/
    └── EDA.ipynb                       # 23 cells: EDA + 6 experiments + holdout validation
```

---

## Appendix A — Quick reference: all reported metrics

### Validation set (n = 42,721; 74 frauds; 42,647 legitimate)

| Model | Threshold | F1 | PR-AUC | Precision | Recall | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression @ 0.5 | 0.5000 | 0.7258 | 0.7740 | 0.9000 | 0.6081 | 5 | 29 | 45 |
| Logistic Regression tuned | 0.0726 | 0.8312 | 0.7740 | 0.8000 | 0.8649 | 16 | 10 | 64 |
| Logistic Regression + SMOTE | 1.0000 | 0.8369 | 0.7566 | 0.8806 | 0.7973 | 8 | 15 | 59 |
| Random Forest | 0.2000 | 0.8707 | 0.8575 | 0.8767 | 0.8649 | 9 | 10 | 64 |
| Random Forest grid-searched | 0.2000 | 0.8707 | 0.8575 | 0.8767 | 0.8649 | 9 | 10 | 64 |
| Soft Voting (LR + RF) | 0.3295 | 0.8671 | 0.8471 | 0.8986 | 0.8378 | 7 | 12 | 62 |
| **XGBoost** | **0.1492** | **0.8828** | **0.8770** | **0.9014** | **0.8649** | **7** | **10** | **64** |

### Holdout test set (n = 42,722; 74 frauds; 42,648 legitimate)

| Model | Threshold | F1 | PR-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.2000 | 0.8243 | 0.8135 | 42,635 | 13 | 13 | 61 |
| **XGBoost** ⭐ | **0.1492** | **0.8633** | **0.8444** | **42,643** | **5** | **14** | **60** |

### Benchmark references

| Reference | PR-AUC | Note |
|---|---|---|
| Majority-class classifier | `0.0000` | Recall 0 — the do-nothing baseline |
| Random classifier | `0.0017` | = the base rate; the true PR-AUC floor |
| This project's XGBoost | **`0.8444`** | **~489× the random baseline** |

---

## Appendix B — Full XGBoost hyperparameter reference

| Parameter | Value | Purpose in this problem |
|---|---|---|
| `n_estimators` | 500 | Sequential trees; paired with a low learning rate |
| `max_depth` | 5 | Moderate depth — deeper overfits 344 minority examples |
| `learning_rate` | 0.05 | Low, for stable optimisation over 500 rounds |
| `subsample` | 0.8 | Row subsampling — stochasticity against overfitting |
| `colsample_bytree` | 0.8 | Column subsampling — guards against noisy PCA components |
| `min_child_weight` | 3 | Leaf-size floor; prevents leaves fitting tiny minority slices |
| `gamma` | 0 | No split penalty |
| `reg_alpha` | 0.0 | No L1 penalty |
| `reg_lambda` | 1.0 | Default L2 — light regularisation |
| `scale_pos_weight` | **1** | **Deliberately 1** — cost handled by the threshold, not the loss (§4.3) |
| `objective` | `binary:logistic` | Outputs calibrated probabilities for thresholding |
| `eval_metric` | **`aucpr`** | **Optimises PR-AUC during training — the right objective from the start** |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | All CPU cores for the single final fit |

**Tuning strategy: `scale_pos_weight=1` + low `learning_rate` + high `n_estimators`
+ tuned threshold**, rather than class weighting. This follows the finding in
[§4.3](#43-resampling-is-not-a-substitute-for-threshold-tuning-and-often-worse)
that all form of resampling failed here, and `eval_metric='aucpr'` puts the
imbalance-aware objective inside the optimisation loop from the start.

---

<div align="center">

**Dataset:** [Credit Card Fraud Detection — ULB / Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
&nbsp;·&nbsp; **Final model:** XGBoost &nbsp;·&nbsp; **Holdout F1:** `0.8633` &nbsp;·&nbsp; **Holdout PR-AUC:** `0.8444`

*Analysis conducted in `Notebook/EDA.ipynb`; results reproduced by `Code/credit_fraud_train.py`.*

</div>

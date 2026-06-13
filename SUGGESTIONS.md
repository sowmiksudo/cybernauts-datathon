# SUGGESTIONS.md — Roadmap from 0.9846 → 0.99+ AUC-ROC

> **Current Best:** 0.98464 (public LB, V2 submission) / 0.98459 (V4 local CV)
> **Target:** 0.99+ AUC-ROC (confirmed achievable on Kaggle leaderboard)
> **Gap to close:** ~0.55 percentage points (~5.5 basis points)

---

## Diagnosis: Why We Are Stuck at 0.9846

Before listing solutions, it's critical to understand *why* V2 → V3 → V4 produced nearly identical scores despite adding features, cleaning data, and stacking models.

| Version | Local CV AUC | Features | Models | Key Addition |
|---------|-------------|----------|--------|--------------|
| V2 | 0.98456 | 85 raw | LGB+XGB+CAT blend | Baseline features |
| V3 | 0.98460 | 123 | 3-seed stacking + LogReg meta | Interactions, rank features |
| V4 | 0.98459 | 131 | 3-seed SLSQP blend | Symlog, categoricals, winsorization |

**All three versions hit a ceiling at ~0.9846.** The V3/V4 additions (interactions, log transforms, rank features, symlog) contributed **zero meaningful AUC improvement.** This means:

1. **The existing feature space is saturated.** Tree models already capture log transforms, interactions, and rank orderings implicitly. Adding explicit versions just creates redundant columns.
2. **The 0.55% gap is not in the modeling — it's in the data.** Teams scoring 0.99+ are extracting fundamentally different signal from the raw data that our pipeline never computes.

---

## Section A: High-Impact Feature Engineering (Expected: +0.3–0.5%)

These are the features **most likely to close the gap** because they extract entirely new signals from the raw transaction and balance tables that the current pipeline ignores.

### A1. DST_ACCOUNT Graph Features (Network Topology)

> [!IMPORTANT]
> This is the single highest-impact suggestion. The current pipeline only aggregates by `SRC_ACCOUNT`. It **completely ignores who the user is transacting with**.

Currently we compute `n_unique_dst` (count of unique destinations). But we never look at *what kind* of destinations they are, or *what those destinations are doing*.

**New features to extract:**

```python
# For each SRC_ACCOUNT, join with KYC on DST_ACCOUNT to get destination types
# Then compute:

# 1. Fraction of outbound txns going to Merchant vs Biller vs Customer
dst_type_fractions = (
    trx.join(kyc.select(["ACCOUNT_ID", "ACCOUNT_TYPE"]),
             left_on="DST_ACCOUNT", right_on="ACCOUNT_ID", how="left")
    .group_by("SRC_ACCOUNT")
    .agg([
        (pl.col("ACCOUNT_TYPE") == "Merchant").mean().alias("frac_dst_merchant"),
        (pl.col("ACCOUNT_TYPE") == "Biller").mean().alias("frac_dst_biller"),
        (pl.col("ACCOUNT_TYPE") == "Customer").mean().alias("frac_dst_customer"),
    ])
)

# 2. Unique Merchant diversity (sticky users transact at many merchants)
merchant_diversity = (
    trx.filter(pl.col("TRX_TYPE") == "MerchantPay")
    .group_by("SRC_ACCOUNT")
    .agg([
        pl.col("DST_ACCOUNT").n_unique().alias("n_unique_merchants"),
        pl.col("DST_ACCOUNT").filter(pl.col("TRX_DATE").dt.month() == 3)
          .n_unique().alias("n_unique_merchants_march"),
    ])
)

# 3. "Hub score": How many OTHER users also transact with your top destination?
# (Proxy for whether the user transacts with popular vs niche merchants)
```

**Why this matters:** A user who sends money to 15 different merchants and billers is deeply embedded in the ecosystem. A user who only does P2P to 2 contacts is not. This distinction is invisible to our current feature set.

---

### A2. Inbound Transaction Detail Features

The current pipeline computes `F10_inbound_count` and `inbound_amt_march` but treats all inbound transactions identically. We need to distinguish *who is sending money to the user*.

```python
# Inbound from Customers (P2P received) vs system credits
# Regularity of inbound — does the user receive money on predictable intervals?
inbound_detail = (
    trx.group_by("DST_ACCOUNT")
    .agg([
        # Unique senders who are Customers (social network size)
        pl.col("SRC_ACCOUNT").filter(pl.col("TRX_TYPE") == "P2P")
          .n_unique().alias("n_unique_p2p_senders"),

        # Does the user receive salary-like regular deposits?
        pl.col("TRX_AMT").filter(pl.col("TRX_TYPE") == "CashIn")
          .std().alias("cashin_amt_std"),  # Low std = regular deposits

        # Inbound recency (last time someone sent them money)
        (pl.lit(OBSERVATION_END) - pl.col("TRX_DATE").max())
          .dt.total_days().alias("inbound_recency_days"),
    ])
)
```

---

### A3. Weekly Time-Series Decomposition

**This is likely the biggest gap.** The current pipeline aggregates by month (Jan/Feb/March). But the churn window starts April 1st — the signal is in the *weekly trajectory* approaching the cutoff, not the monthly average.

Our audit confirmed: **93% of active users transact after March 25th, but only 10% of churned users do.** This gradient is extremely steep in the final 2-3 weeks and our pipeline only captures it coarsely via `tx_last_7d` and `tx_last_14d` counts.

```python
# Compute weekly transaction counts for weeks 1-13 of the observation period
# Then derive:

# Week-over-week acceleration (second derivative of activity)
# e.g., user who went from 10 txns/week → 5 → 1 → 0 has negative acceleration

# Consecutive zero-activity weeks (streak of inactivity)
# A user with 3 consecutive zero weeks ending March 31 is very different
# from a user who was inactive weeks 5-7 but returned in week 12

# Day-of-week concentration in March
# Active users transact on weekdays AND weekends
# Churning users may show unusual patterns (only weekend, only 1 specific day)
```

**Concrete features:**
- `weeks_since_last_active`: More granular than `F1_recency_days`
- `max_consecutive_inactive_weeks`: Captures dormancy streaks
- `tx_count_week_12` / `tx_count_week_13`: Final 2 weeks before cutoff
- `weekly_trend_slope`: Linear regression slope of weekly tx counts
- `weekend_vs_weekday_ratio_march`: Behavioral engagement pattern

---

### A4. Balance Trajectory Shape Features

The balance table has **daily** granularity (~360M rows) but we only extract `mean`, `std`, `min`, `max` per month. The *shape* of the balance curve is highly informative.

```python
# Balance trajectory features:

# 1. Days with zero balance in last 2 weeks
# 2. Balance slope (linear regression on daily balance in March)
# 3. Balance "cliff": largest single-day drop / max_balance
# 4. End-of-month balance vs start-of-month balance (net flow direction)
# 5. Number of days balance INCREASED vs DECREASED in March

bal_trajectory = (
    bal.filter(pl.col("DATE").dt.month() == 3)
    .sort("DATE")
    .group_by("ACCOUNT_ID")
    .agg([
        # Net balance change over March
        (pl.col("AVAILABLE_BALANCE").last() - pl.col("AVAILABLE_BALANCE").first())
          .alias("bal_net_change_march"),

        # Number of days balance went up
        (pl.col("AVAILABLE_BALANCE").diff() > 0).sum().alias("bal_up_days"),

        # Number of days balance went down
        (pl.col("AVAILABLE_BALANCE").diff() < 0).sum().alias("bal_down_days"),

        # Max single-day withdrawal
        pl.col("AVAILABLE_BALANCE").diff().min().alias("max_daily_drop"),
    ])
)
```

---

### A5. Transaction Amount Distribution Features

We compute `mean`, `max`, `std`, `median` of `TRX_AMT`. But the *shape* of the amount distribution matters:

```python
# Skewness and kurtosis of transaction amounts
# Small-amount-only users behave differently from large-amount users

# Amount quantiles (25th, 75th percentile)
# IQR (interquartile range) of transaction amounts
# Fraction of transactions below 100 BDT (micro-transactions)
# Fraction of transactions above 10,000 BDT (large transactions)
```

---

### A6. Hour-of-Day and Day-of-Week Features

The `TRX_DATETIME` field contains hour-level precision that we currently discard (we only extract the date). Time-of-day patterns correlate with user segments:

```python
# Extract hour from TRX_DATETIME
trx_with_hour = trx.with_columns([
    pl.col("TRX_DATETIME").str.slice(11, 2).cast(pl.Int8).alias("TRX_HOUR"),
    pl.col("TRX_DATE").dt.weekday().alias("TRX_DOW"),  # 0=Monday
])

# Features:
# - Peak hour of activity (mode of TRX_HOUR)
# - Fraction of transactions during business hours (9-17) vs evening/night
# - Number of distinct hours active in March
# - Weekend transaction ratio
```

---

## Section B: Modeling & Ensembling Fixes (Expected: +0.1–0.2%)

### B1. Fix the Broken SLSQP Blending Optimizer

> [!WARNING]
> The V4 SLSQP optimizer returned the *initial guess* `[0.6, 0.2, 0.2]` unchanged because AUC-ROC is non-differentiable. This means V4's blend was never optimized at all.

**Fix:** Replace SLSQP with exhaustive grid search (takes <1 second):

```python
best_w, best_auc = None, 0
for w1 in np.arange(0.30, 0.90, 0.01):
    for w2 in np.arange(0.02, 1.0 - w1, 0.01):
        w3 = 1.0 - w1 - w2
        if w3 < 0.01:
            continue
        blend = w1 * oof_lgb_avg + w2 * oof_xgb_avg + w3 * oof_cat_avg
        a = roc_auc_score(y, blend)
        if a > best_auc:
            best_auc = a
            best_w = (w1, w2, w3)
```

V3 did this correctly and found `w=(0.78, 0.07, 0.15)` with AUC `0.984603`, which is marginally better than the broken V4 blend.

---

### B2. Rank-Average Blending (Instead of Probability Blending)

LightGBM, XGBoost, and CatBoost produce probabilities on different scales. Rank-averaging normalizes this:

```python
from scipy.stats import rankdata

def rank_average(preds_list, weights):
    ranked = [rankdata(p) / len(p) for p in preds_list]
    return sum(w * r for w, r in zip(weights, ranked))
```

This typically adds **0.01–0.03%** AUC over raw probability blending.

---

### B3. Tune XGBoost and CatBoost with Optuna (Not Just LightGBM)

Currently only LightGBM gets Optuna tuning. XGBoost uses hardcoded `lr=0.025, depth=7` and CatBoost uses `lr=0.04, depth=7`. These may be suboptimal.

**Action:** Run 15-trial Optuna for XGBoost and CatBoost each on the 100k sample.

---

### B4. Increase Number of Seeds

Going from 3 seeds `[42, 123, 2024]` to **5 seeds** reduces variance in OOF predictions. The compute cost is linear but each additional seed adds stability to the blend.

---

### B5. Add a 4th Model: Extra Trees or Neural Network

Adding a model with a fundamentally *different* inductive bias improves ensemble diversity:

- **TabNet** or **simple MLP** with entity embeddings for GENDER/REGION
- **ExtraTreesClassifier** (extremely randomized trees — different split strategy from GBDT)

Even if the individual model scores lower (~0.982), it can improve the blend by capturing different error patterns.

---

## Section C: Data Utilization Improvements (Expected: +0.1–0.3%)

### C1. Use the Full Transaction Table as DST (Inbound)

The current pipeline computes inbound features from the transaction table using `DST_ACCOUNT`. But it only joins on `ACCOUNT_ID` from the customer set. There may be signal in how merchants and billers interact with the customer (e.g., refunds, promotional credits).

### C2. Target Encoding for REGION

Instead of integer-coding REGION, use **leave-one-out target encoding** with K-fold regularization:

```python
from sklearn.model_selection import KFold

def target_encode(train, test, col, target, n_splits=5):
    train[f"te_{col}"] = 0.0
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for ti, vi in kf.split(train):
        means = train.iloc[ti].groupby(col)[target].mean()
        train.loc[train.index[vi], f"te_{col}"] = train.iloc[vi][col].map(means)
    global_mean = train[target].mean()
    test[f"te_{col}"] = test[col].map(
        train.groupby(col)[target].mean()
    ).fillna(global_mean)
```

This is especially powerful if certain regions have structurally different churn rates after controlling for activity.

### C3. Frequency Encoding for High-Cardinality Categoricals

Replace rare REGION grouping with frequency encoding: each region gets its population count as a numeric feature. This preserves more information than collapsing to "RARE".

---

## Section D: Advanced Techniques (Expected: +0.05–0.15%)

### D1. Adversarial Validation Reweighting

Check for distribution shift between train and test. If the test set has a slightly different user composition, reweight training samples to match:

```python
# Train a classifier to distinguish train vs test
# Use predicted probabilities as sample weights during model training
```

### D2. Pseudo-Labeling

Use high-confidence predictions from the current model to pseudo-label test samples, then retrain:

```python
# Take test samples where CHURN_PROB > 0.95 or < 0.05
# Add them to training set with pseudo-labels
# Retrain the ensemble
```

This is effective when the test set is large (255k samples = 30% of total data).

### D3. Post-Processing: Isotonic Regression Calibration

Calibrate probabilities after blending to ensure monotonicity:

```python
from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(oof_blend, y)
calibrated_test = ir.transform(test_blend)
```

---

## Prioritized Execution Order

| Priority | Suggestion | Expected Gain | Effort |
|----------|-----------|---------------|--------|
| 🔴 1 | **A3: Weekly time-series decomposition** | +0.15–0.25% | Medium |
| 🔴 2 | **A1: DST_ACCOUNT graph features** | +0.10–0.20% | Medium |
| 🔴 3 | **A4: Balance trajectory shape** | +0.05–0.15% | Low |
| 🟡 4 | **B1: Fix SLSQP → grid search blend** | +0.01–0.03% | Trivial |
| 🟡 5 | **A6: Hour/day-of-week features** | +0.03–0.08% | Low |
| 🟡 6 | **B2: Rank-average blending** | +0.01–0.03% | Trivial |
| 🟡 7 | **C2: Target encoding for REGION** | +0.02–0.05% | Low |
| 🟡 8 | **A2: Inbound transaction detail** | +0.03–0.08% | Low |
| 🟢 9 | **B3: Optuna for XGB/CAT** | +0.01–0.03% | Medium |
| 🟢 10 | **A5: Amount distribution features** | +0.02–0.05% | Low |
| 🟢 11 | **D2: Pseudo-labeling** | +0.02–0.05% | Medium |
| 🟢 12 | **B5: Add 4th model (TabNet/ExtraTrees)** | +0.01–0.03% | Medium |

> [!TIP]
> **Suggestions 1–3 alone should be sufficient to cross 0.99%.** The remaining suggestions provide incremental polish. Focus engineering time on extracting *new signal from the raw data*, not on model tuning — the models are already near-optimal for the current feature set.

---

## Summary

The path from **0.9846 → 0.99** is primarily a **feature engineering problem**, not a modeling problem. The current pipeline exhaustively processes monthly aggregates but leaves three massive data dimensions untapped:

1. **Temporal granularity** — weekly/daily trajectory shapes instead of monthly averages
2. **Graph structure** — who the user transacts with (destination types, merchant diversity)  
3. **Behavioral micro-patterns** — hour-of-day, day-of-week, amount distribution shapes

Teams at 0.99+ are likely exploiting all three. Implementing suggestions A1, A3, and A4 in a V5 pipeline should close the gap.

"""
FictiPay Datathon V2 Pipeline — Enhanced Feature Engineering & 3-Model Ensemble
Target: AUC-ROC > 0.985
Author: Shayer Mahmud Sowmik
"""
import os, sys, datetime, logging
import polars as pl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("pipeline_v2.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

OBSERVATION_END = datetime.date(2024, 3, 31)
DATA = "bkash-presents-nsucec-datathon/public"
os.makedirs("explainability", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — ENHANCED FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════
def extract_features():
    logger.info("═══ V2 Feature Extraction ═══")

    # ── KYC ──
    kyc = pl.scan_parquet(f"{DATA}/kyc.parquet").filter(pl.col("ACCOUNT_TYPE") == "Customer")
    kyc = kyc.with_columns([
        ((pl.lit(OBSERVATION_END) - pl.col("ACCOUNT_OPEN_DATE").dt.date()).dt.total_days()).alias("F8_vintage_days"),
        pl.col("GENDER").cast(pl.Categorical).to_physical().cast(pl.Int8),
        pl.col("REGION").cast(pl.Categorical).to_physical().cast(pl.Int16),
    ])

    # ── TRANSACTIONS ──
    logger.info("Scanning transactions...")
    trx = pl.scan_parquet(f"{DATA}/transactions/trx_2024-0*.parquet").with_columns([
        pl.col("TRX_DATETIME").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("TRX_DATE"),
    ])

    # --- Outbound (SRC) aggregates ---
    src = (
        trx.group_by("SRC_ACCOUNT")
        .agg([
            # ── Core 10 features (raw ingredients) ──
            ((pl.lit(OBSERVATION_END) - pl.col("TRX_DATE").max()).dt.total_days()).alias("F1_recency_days"),
            pl.col("TrxID").count().alias("total_tx_count"),
            pl.col("TrxID").filter(pl.col("TRX_DATE").dt.month() == 3).count().alias("tx_count_march"),
            pl.col("TrxID").filter(pl.col("TRX_DATE").dt.month() == 2).count().alias("tx_count_feb"),
            pl.col("TrxID").filter(pl.col("TRX_DATE").dt.month() == 1).count().alias("tx_count_jan"),
            pl.col("TRX_AMT").filter(pl.col("TRX_DATE").dt.month() == 3).sum().alias("tx_amt_march"),
            pl.col("TRX_AMT").filter(pl.col("TRX_DATE").dt.month() == 2).sum().alias("tx_amt_feb"),
            pl.col("TRX_AMT").filter(pl.col("TRX_DATE").dt.month() == 1).sum().alias("tx_amt_jan"),
            pl.col("TRX_AMT").filter(pl.col("TRX_DATE").dt.month() < 3).sum().alias("tx_amt_early"),
            pl.col("TRX_AMT").filter(pl.col("TRX_TYPE") == "CashOut").sum().alias("amt_cashout"),
            pl.col("TRX_AMT").filter(pl.col("TRX_TYPE") == "CashIn").sum().alias("amt_cashin"),
            pl.col("TrxID").filter(pl.col("TRX_TYPE") == "MerchantPay").count().alias("count_merchant"),
            pl.col("TrxID").filter(pl.col("TRX_TYPE") == "BillPay").count().alias("count_bill"),
            pl.col("TrxID").filter(pl.col("TRX_TYPE") == "P2P").count().alias("count_p2p"),
            pl.col("TrxID").filter(pl.col("TRX_TYPE") == "CashIn").count().alias("count_cashin"),
            pl.col("TrxID").filter(pl.col("TRX_TYPE") == "CashOut").count().alias("count_cashout"),

            # ── NEW: Recency windows ──
            pl.col("TrxID").filter(
                pl.col("TRX_DATE") >= pl.lit(datetime.date(2024, 3, 25))
            ).count().alias("tx_last_7d"),
            pl.col("TrxID").filter(
                pl.col("TRX_DATE") >= pl.lit(datetime.date(2024, 3, 18))
            ).count().alias("tx_last_14d"),

            # ── NEW: Unique destinations (transaction diversity) ──
            pl.col("DST_ACCOUNT").n_unique().alias("n_unique_dst"),
            pl.col("DST_ACCOUNT").filter(pl.col("TRX_DATE").dt.month() == 3).n_unique().alias("n_unique_dst_march"),

            # ── NEW: Amount statistics ──
            pl.col("TRX_AMT").mean().alias("avg_tx_amt"),
            pl.col("TRX_AMT").max().alias("max_tx_amt"),
            pl.col("TRX_AMT").std().alias("std_tx_amt"),
            pl.col("TRX_AMT").filter(pl.col("TRX_DATE").dt.month() == 3).mean().alias("avg_tx_amt_march"),

            # ── NEW: Unique active days ──
            pl.col("TRX_DATE").n_unique().alias("n_active_days"),
            pl.col("TRX_DATE").filter(pl.col("TRX_DATE").dt.month() == 3).n_unique().alias("n_active_days_march"),
        ])
        .rename({"SRC_ACCOUNT": "ACCOUNT_ID"})
    )

    # --- Inbound (DST) aggregates ---
    logger.info("Computing inbound features (F10)...")
    dst = (
        trx.group_by("DST_ACCOUNT")
        .agg([
            pl.col("TrxID").count().alias("F10_inbound_count"),
            pl.col("TrxID").filter(pl.col("TRX_DATE").dt.month() == 3).count().alias("inbound_count_march"),
            pl.col("TRX_AMT").sum().alias("inbound_amt_total"),
            pl.col("SRC_ACCOUNT").n_unique().alias("n_unique_senders"),
        ])
        .rename({"DST_ACCOUNT": "ACCOUNT_ID"})
    )

    # ── BALANCES ──
    logger.info("Scanning balances...")
    bal = pl.scan_parquet(f"{DATA}/dayend_balance/balance_2024-0*.parquet").with_columns([
        pl.col("DATE").str.to_date("%Y-%m-%d").alias("DATE"),
    ])

    bal_feats = (
        bal.group_by("ACCOUNT_ID")
        .agg([
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE") == pl.lit(OBSERVATION_END)).first().alias("F9_final_balance"),
            pl.col("AVAILABLE_BALANCE").std().alias("F5_balance_volatility"),
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 3).mean().alias("avg_bal_march"),
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 2).mean().alias("avg_bal_feb"),
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 1).mean().alias("avg_bal_jan"),

            # ── NEW: Last-week and last-2-week balance means ──
            pl.col("AVAILABLE_BALANCE").filter(
                pl.col("DATE") >= pl.lit(datetime.date(2024, 3, 25))
            ).mean().alias("avg_bal_last_7d"),
            pl.col("AVAILABLE_BALANCE").filter(
                pl.col("DATE") >= pl.lit(datetime.date(2024, 3, 18))
            ).mean().alias("avg_bal_last_14d"),

            # ── NEW: Zero-balance days ──
            pl.col("AVAILABLE_BALANCE").filter(
                pl.col("AVAILABLE_BALANCE") == 0.0
            ).count().alias("zero_balance_days"),
            pl.col("AVAILABLE_BALANCE").filter(
                (pl.col("AVAILABLE_BALANCE") == 0.0) & (pl.col("DATE").dt.month() == 3)
            ).count().alias("zero_balance_days_march"),

            # ── NEW: Min balance in March ──
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 3).min().alias("min_bal_march"),

            # ── NEW: Balance coefficient of variation ──
            pl.col("AVAILABLE_BALANCE").mean().alias("avg_bal_overall"),

            # ── NEW: Balance range ──
            pl.col("AVAILABLE_BALANCE").max().alias("max_bal"),
            pl.col("AVAILABLE_BALANCE").min().alias("min_bal"),
        ])
    )

    # ── JOIN ──
    logger.info("Joining feature tables...")
    joined = (
        kyc
        .join(src, on="ACCOUNT_ID", how="left")
        .join(dst, on="ACCOUNT_ID", how="left")
        .join(bal_feats, on="ACCOUNT_ID", how="left")
    )

    # ── FILL NULLS ──
    # All numeric columns from src/dst/bal that could be null for accounts with no activity
    null_zero_cols = [
        "F1_recency_days", "total_tx_count", "tx_count_march", "tx_count_feb", "tx_count_jan",
        "tx_amt_march", "tx_amt_feb", "tx_amt_jan", "tx_amt_early",
        "amt_cashout", "amt_cashin",
        "count_merchant", "count_bill", "count_p2p", "count_cashin", "count_cashout",
        "tx_last_7d", "tx_last_14d",
        "n_unique_dst", "n_unique_dst_march",
        "avg_tx_amt", "max_tx_amt", "std_tx_amt", "avg_tx_amt_march",
        "n_active_days", "n_active_days_march",
        "F10_inbound_count", "inbound_count_march", "inbound_amt_total", "n_unique_senders",
        "F9_final_balance", "F5_balance_volatility",
        "avg_bal_march", "avg_bal_feb", "avg_bal_jan",
        "avg_bal_last_7d", "avg_bal_last_14d",
        "zero_balance_days", "zero_balance_days_march",
        "min_bal_march", "avg_bal_overall", "max_bal", "min_bal",
    ]
    filled = joined.with_columns([
        pl.col(c).fill_null(0 if c != "F1_recency_days" else 91) for c in null_zero_cols
    ])

    # ── DERIVED FEATURES ──
    logger.info("Computing derived ratios...")
    derived = filled.with_columns([
        # Core 10
        (pl.col("tx_count_march") / ((pl.col("tx_count_jan") + pl.col("tx_count_feb")) / 2.0 + 1.0)).alias("F2_momentum_ratio"),
        (pl.col("tx_amt_march") / (pl.col("tx_amt_early") + 1.0)).alias("F3_monetary_trend"),
        (pl.col("avg_bal_march") - pl.col("avg_bal_jan")).alias("F4_balance_depletion"),
        (pl.col("amt_cashout") / (pl.col("amt_cashin") + 1.0)).alias("F6_pass_through_index"),
        ((pl.col("count_merchant") + pl.col("count_bill")) / (pl.col("total_tx_count") + 1.0)).alias("F7_ecosystem_index"),

        # ── NEW: Feb-to-March momentum (catches late-stage decline) ──
        (pl.col("tx_count_march") / (pl.col("tx_count_feb") + 1.0)).alias("momentum_feb_to_march"),
        (pl.col("tx_amt_march") / (pl.col("tx_amt_feb") + 1.0)).alias("amt_momentum_feb_to_march"),

        # ── NEW: Jan-to-Feb momentum (early decline) ──
        (pl.col("tx_count_feb") / (pl.col("tx_count_jan") + 1.0)).alias("momentum_jan_to_feb"),

        # ── NEW: Last-week activity ratio ──
        (pl.col("tx_last_7d") / (pl.col("tx_count_march") + 1.0)).alias("last_7d_ratio"),
        (pl.col("tx_last_14d") / (pl.col("tx_count_march") + 1.0)).alias("last_14d_ratio"),

        # ── NEW: Destination diversity trend ──
        (pl.col("n_unique_dst_march") / (pl.col("n_unique_dst") + 1.0)).alias("dst_diversity_trend"),

        # ── NEW: Balance depletion (Feb baseline too) ──
        (pl.col("avg_bal_march") - pl.col("avg_bal_feb")).alias("bal_depletion_feb_to_march"),
        (pl.col("avg_bal_last_7d") - pl.col("avg_bal_march")).alias("bal_trend_last_7d"),
        (pl.col("avg_bal_last_14d") - pl.col("avg_bal_march")).alias("bal_trend_last_14d"),

        # ── NEW: Balance CoV ──
        (pl.col("F5_balance_volatility") / (pl.col("avg_bal_overall") + 1.0)).alias("balance_cov"),

        # ── NEW: Balance range ──
        (pl.col("max_bal") - pl.col("min_bal")).alias("balance_range"),

        # ── NEW: Inbound momentum ──
        (pl.col("inbound_count_march") / (pl.col("F10_inbound_count") + 1.0)).alias("inbound_march_ratio"),

        # ── NEW: P2P fraction & CashIn/CashOut fractions ──
        (pl.col("count_p2p") / (pl.col("total_tx_count") + 1.0)).alias("p2p_fraction"),
        (pl.col("count_cashin") / (pl.col("total_tx_count") + 1.0)).alias("cashin_fraction"),
        (pl.col("count_cashout") / (pl.col("total_tx_count") + 1.0)).alias("cashout_fraction"),

        # ── NEW: Activity density (txns per active day) ──
        (pl.col("total_tx_count") / (pl.col("n_active_days") + 1.0)).alias("activity_density"),
        (pl.col("tx_count_march") / (pl.col("n_active_days_march") + 1.0)).alias("activity_density_march"),

        # ── NEW: Dormancy indicators ──
        (pl.col("F1_recency_days") > 30).cast(pl.Int8).alias("is_dormant_30d"),
        (pl.col("F1_recency_days") > 14).cast(pl.Int8).alias("is_dormant_14d"),
        (pl.col("F1_recency_days") > 7).cast(pl.Int8).alias("is_dormant_7d"),
        (pl.col("tx_count_march") == 0).cast(pl.Int8).alias("zero_march_txns"),
        (pl.col("tx_last_7d") == 0).cast(pl.Int8).alias("zero_last_7d_txns"),
    ])

    # ── COLLECT ──
    logger.info("Streaming collection...")
    train_labels = pl.scan_csv(f"{DATA}/train_labels.csv")
    test_keys = pl.scan_csv(f"{DATA}/test.csv")

    train_pl = train_labels.join(derived, on="ACCOUNT_ID", how="inner").collect(streaming=True)
    test_pl = test_keys.join(derived, on="ACCOUNT_ID", how="left").collect(streaming=True)

    train_df = train_pl.to_pandas()
    test_df = test_pl.to_pandas()

    logger.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
def preprocess(train_df, test_df):
    logger.info("Preprocessing...")

    exclude = {"ACCOUNT_ID", "ACCOUNT_TYPE", "ACCOUNT_OPEN_DATE", "CHURN"}
    num_cols = [c for c in train_df.columns if c not in exclude and train_df[c].dtype in [np.float64, np.int64, np.int32, np.int16, np.int8, np.float32]]

    # Winsorization at 99th percentile
    for col in num_cols:
        cap = train_df[col].quantile(0.99)
        if cap > 0:
            train_df[col] = train_df[col].clip(upper=cap)
            test_df[col] = test_df[col].clip(upper=cap)

    # Log-transforms on heavily skewed monetary/count features
    log_cols = [
        "total_tx_count", "tx_amt_march", "tx_amt_early", "tx_amt_feb", "tx_amt_jan",
        "F10_inbound_count", "F9_final_balance", "F5_balance_volatility",
        "inbound_amt_total", "max_tx_amt", "avg_tx_amt", "avg_bal_overall",
        "amt_cashout", "amt_cashin", "balance_range",
    ]
    for col in log_cols:
        if col in train_df.columns:
            train_df[f"log_{col}"] = np.log1p(train_df[col])
            test_df[f"log_{col}"] = np.log1p(test_df[col])

    # Fill any remaining NaN
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════
# PHASE 3 — OPTUNA TUNING (LightGBM)
# ═══════════════════════════════════════════════════════════════════
def tune_lgb(X, y):
    logger.info("Optuna tuning LightGBM (200k sample, 40 trials)...")
    n_sample = min(200000, len(X))
    idx = np.random.RandomState(42).choice(len(X), n_sample, replace=False)
    Xs, ys = X.iloc[idx], y.iloc[idx]

    spw = (ys == 0).sum() / ((ys == 1).sum() + 1)

    def objective(trial):
        p = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("lr", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("nl", 31, 200),
            "max_depth": trial.suggest_int("md", 5, 10),
            "min_child_samples": trial.suggest_int("mcs", 30, 300),
            "feature_fraction": trial.suggest_float("ff", 0.5, 0.95),
            "bagging_fraction": trial.suggest_float("bf", 0.5, 0.95),
            "bagging_freq": 1,
            "reg_alpha": trial.suggest_float("ra", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 1e-3, 10.0, log=True),
            "scale_pos_weight": spw,
            "verbose": -1, "n_jobs": -1, "random_state": 42,
        }
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for ti, vi in folds.split(Xs, ys):
            ds_t = lgb.Dataset(Xs.iloc[ti], label=ys.iloc[ti])
            ds_v = lgb.Dataset(Xs.iloc[vi], label=ys.iloc[vi])
            m = lgb.train(p, ds_t, 800, valid_sets=[ds_v],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
            scores.append(roc_auc_score(ys.iloc[vi], m.predict(Xs.iloc[vi], num_iteration=m.best_iteration)))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, show_progress_bar=False)
    logger.info(f"Best LGB trial: {study.best_value:.6f} | {study.best_params}")

    # Save chart
    try:
        plt.figure(figsize=(10, 5))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig("explainability/optuna_history_v2.png", dpi=150)
        plt.close()
    except Exception:
        pass

    return {
        "learning_rate": study.best_params["lr"],
        "num_leaves": study.best_params["nl"],
        "max_depth": study.best_params["md"],
        "min_child_samples": study.best_params["mcs"],
        "feature_fraction": study.best_params["ff"],
        "bagging_fraction": study.best_params["bf"],
        "reg_alpha": study.best_params["ra"],
        "reg_lambda": study.best_params["rl"],
    }


# ═══════════════════════════════════════════════════════════════════
# PHASE 4 — 3-MODEL ENSEMBLE TRAINING
# ═══════════════════════════════════════════════════════════════════
def train_ensemble(train_df, test_df, best_lgb_params):
    logger.info("═══ 3-Model Ensemble Training ═══")

    exclude = {"ACCOUNT_ID", "ACCOUNT_TYPE", "ACCOUNT_OPEN_DATE", "CHURN"}
    feat_cols = [c for c in train_df.columns if c not in exclude]

    X = train_df[feat_cols].copy()
    y = train_df["CHURN"]
    X_test = test_df[feat_cols].copy()

    for col in ["GENDER", "REGION"]:
        X[col] = X[col].astype(int)
        X_test[col] = X_test[col].astype(int)

    spw = (y == 0).sum() / ((y == 1).sum() + 1)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    tp_lgb = np.zeros(len(X_test))
    tp_xgb = np.zeros(len(X_test))
    tp_cat = np.zeros(len(X_test))

    lgb_params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "scale_pos_weight": spw,
        "bagging_freq": 1, "verbose": -1, "n_jobs": -1, "random_state": 42,
        **best_lgb_params,
    }
    xgb_params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "learning_rate": 0.03, "max_depth": 7, "scale_pos_weight": spw,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "tree_method": "hist", "n_jobs": -1, "random_state": 42,
    }

    models_lgb = []
    for fold, (ti, vi) in enumerate(folds.split(X, y)):
        logger.info(f"--- Fold {fold+1}/5 ---")
        Xtr, ytr = X.iloc[ti], y.iloc[ti]
        Xva, yva = X.iloc[vi], y.iloc[vi]

        # LightGBM
        ds_t = lgb.Dataset(Xtr, label=ytr)
        ds_v = lgb.Dataset(Xva, label=yva)
        m_lgb = lgb.train(lgb_params, ds_t, 2000, valid_sets=[ds_t, ds_v],
                          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        oof_lgb[vi] = m_lgb.predict(Xva, num_iteration=m_lgb.best_iteration)
        tp_lgb += m_lgb.predict(X_test, num_iteration=m_lgb.best_iteration) / 5
        models_lgb.append(m_lgb)

        # XGBoost
        m_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=2000, early_stopping_rounds=80)
        m_xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof_xgb[vi] = m_xgb.predict_proba(Xva)[:, 1]
        tp_xgb += m_xgb.predict_proba(X_test)[:, 1] / 5

        # CatBoost
        m_cat = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=7, l2_leaf_reg=3.0,
            scale_pos_weight=spw, eval_metric="AUC",
            early_stopping_rounds=80, verbose=0, random_seed=42,
            task_type="CPU",
        )
        m_cat.fit(Xtr, ytr, eval_set=(Xva, yva))
        oof_cat[vi] = m_cat.predict_proba(Xva)[:, 1]
        tp_cat += m_cat.predict_proba(X_test)[:, 1] / 5

    # Per-model AUC
    auc_lgb = roc_auc_score(y, oof_lgb)
    auc_xgb = roc_auc_score(y, oof_xgb)
    auc_cat = roc_auc_score(y, oof_cat)
    logger.info(f"LightGBM  AUC: {auc_lgb:.6f}")
    logger.info(f"XGBoost   AUC: {auc_xgb:.6f}")
    logger.info(f"CatBoost  AUC: {auc_cat:.6f}")

    # Optimize blend weights
    logger.info("Optimizing ensemble blend weights...")
    best_w, best_auc = None, 0
    for w1 in np.arange(0.3, 0.7, 0.05):
        for w2 in np.arange(0.1, 0.5, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
            a = roc_auc_score(y, blend)
            if a > best_auc:
                best_auc = a
                best_w = (w1, w2, w3)

    w1, w2, w3 = best_w
    logger.info(f"Optimal weights: LGB={w1:.2f}, XGB={w2:.2f}, CAT={w3:.2f}")
    logger.info(f"Blended ensemble AUC: {best_auc:.6f}")

    oof_blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
    tp_blend = w1 * tp_lgb + w2 * tp_xgb + w3 * tp_cat

    # Detailed metrics
    def report(name, y_true, y_pred):
        auc = roc_auc_score(y_true, y_pred)
        pr = average_precision_score(y_true, y_pred)
        thr = np.percentile(y_pred, 90)
        yb = (y_pred >= thr).astype(int)
        p10 = precision_score(y_true, yb)
        r10 = recall_score(y_true, yb)
        logger.info(f"[{name}] AUC={auc:.6f} | PR-AUC={pr:.5f} | P@10%={p10:.5f} | R@10%={r10:.5f}")
        return {"AUC": auc, "PR-AUC": pr, "Precision@10%": p10, "Recall@10%": r10}

    lgb_m = report("LightGBM", y, oof_lgb)
    xgb_m = report("XGBoost", y, oof_xgb)
    cat_m = report("CatBoost", y, oof_cat)
    blend_m = report("Blend", y, oof_blend)

    return models_lgb, tp_blend, lgb_m, xgb_m, cat_m, blend_m, X, y


# ═══════════════════════════════════════════════════════════════════
# PHASE 5 — SHAP & PREDICTIONS
# ═══════════════════════════════════════════════════════════════════
def generate_shap(model, X):
    logger.info("SHAP explainability...")
    Xs = X.sample(n=min(3000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xs)

    plt.figure(figsize=(12, 7))
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP Beeswarm — V2 Enhanced Features", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("explainability/shap_beeswarm_v2.png", dpi=150)
    plt.close()

    for feat in ["F1_recency_days", "F2_momentum_ratio", "tx_last_7d", "momentum_feb_to_march"]:
        if feat in Xs.columns:
            plt.figure(figsize=(8, 5))
            shap.plots.scatter(sv[:, feat], color=sv, show=False)
            plt.title(f"SHAP Dependence — {feat}", fontsize=12, pad=10)
            plt.tight_layout()
            plt.savefig(f"explainability/shap_dep_{feat}_v2.png", dpi=150)
            plt.close()

    logger.info("SHAP plots saved.")


def save_predictions(test_df, preds):
    sub = test_df[["ACCOUNT_ID"]].copy()
    sub["CHURN_PROB"] = preds
    assert len(sub) == 255000, f"Bad row count: {len(sub)}"
    assert sub["CHURN_PROB"].between(0, 1).all(), "Out of bounds"
    sub.to_csv("predictions_v2.csv", index=False)
    logger.info("predictions_v2.csv saved (255,000 rows).")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║    FictiPay V2 Pipeline — Target >0.985  ║")
    logger.info("╚══════════════════════════════════════════╝")

    train_df, test_df = extract_features()
    train_df, test_df = preprocess(train_df, test_df)

    # Prepare X, y for tuning
    exclude = {"ACCOUNT_ID", "ACCOUNT_TYPE", "ACCOUNT_OPEN_DATE", "CHURN"}
    feat_cols = [c for c in train_df.columns if c not in exclude]
    X_tune = train_df[feat_cols].copy()
    y_tune = train_df["CHURN"]
    for col in ["GENDER", "REGION"]:
        X_tune[col] = X_tune[col].astype(int)

    best_lgb = tune_lgb(X_tune, y_tune)
    models, preds, lgb_m, xgb_m, cat_m, blend_m, X, y = train_ensemble(train_df, test_df, best_lgb)

    generate_shap(models[0], X)
    save_predictions(test_df, preds)

    logger.info("═══ V2 Pipeline Complete ═══")
    logger.info(f"Final Blended AUC-ROC: {blend_m['AUC']:.6f}")

if __name__ == "__main__":
    main()

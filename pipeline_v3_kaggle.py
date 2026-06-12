"""
FictiPay V3 Pipeline -- Optimized for Kaggle Execution
Target: AUC-ROC > 0.985

Instructions for Kaggle:
1. Create a new Kaggle Notebook (Python).
2. Zip the 'bkash-presents-nsucec-datathon' folder on your local machine and upload it to Kaggle as a private Dataset (e.g., named 'cybernauts-datathon').
3. Add the uploaded dataset to your Kaggle Notebook.
4. Copy-paste this script into the first cell of your Kaggle Notebook and run it.
5. Download 'predictions_v3.csv' and the saved SHAP plots from the output folder.
"""
import os, sys, datetime, logging, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# 1. Install/Verify packages
try:
    import polars as pl
except ImportError:
    os.system("pip install polars")
    import polars as pl

try:
    import lightgbm as lgb
except ImportError:
    os.system("pip install lightgbm")
    import lightgbm as lgb

try:
    import xgboost as xgb
except ImportError:
    os.system("pip install xgboost")
    import xgb

try:
    from catboost import CatBoostClassifier
except ImportError:
    os.system("pip install catboost")
    from catboost import CatBoostClassifier

try:
    import optuna
except ImportError:
    os.system("pip install optuna")
    import optuna

try:
    import shap
except ImportError:
    os.system("pip install shap")
    import shap

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 2. Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("pipeline_v3.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 3. Path Auto-Detection
OBSERVATION_END = datetime.date(2024, 3, 31)

# Check Kaggle input directory dynamically
def find_data_path():
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            if "train_labels.csv" in files:
                logger.info(f"Found dataset directory at: {root}")
                return root
    # Fallback to local path if running locally
    local_path = "bkash-presents-nsucec-datathon/public"
    if os.path.exists(local_path):
        return local_path
    raise FileNotFoundError("Could not locate the FictiPay dataset directory. Please verify that the dataset is added to the notebook.")

DATA = find_data_path()
os.makedirs("explainability", exist_ok=True)
SEEDS = [42, 123, 2024]  # Multi-seed averaging


# ===========================================================
# PHASE 1 -- FEATURE EXTRACTION
# ===========================================================
def extract_features():
    logger.info("[Phase 1] Feature extraction starting...")

    # Dynamic parquet patterns
    kyc_path = f"{DATA}/kyc.parquet"
    trx_pattern = f"{DATA}/transactions/trx_2024-0*.parquet"
    balance_pattern = f"{DATA}/dayend_balance/balance_2024-0*.parquet"
    train_labels_path = f"{DATA}/train_labels.csv"
    test_path = f"{DATA}/test.csv"

    kyc = pl.scan_parquet(kyc_path).filter(pl.col("ACCOUNT_TYPE") == "Customer")
    kyc = kyc.with_columns([
        ((pl.lit(OBSERVATION_END) - pl.col("ACCOUNT_OPEN_DATE").dt.date()).dt.total_days()).alias("F8_vintage_days"),
        pl.col("GENDER").cast(pl.Categorical).to_physical().cast(pl.Int8),
        pl.col("REGION").cast(pl.Categorical).to_physical().cast(pl.Int16),
    ])

    trx = pl.scan_parquet(trx_pattern).with_columns([
        pl.col("TRX_DATETIME").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("TRX_DATE"),
    ])

    src = (
        trx.group_by("SRC_ACCOUNT")
        .agg([
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
            # Recency windows
            pl.col("TrxID").filter(pl.col("TRX_DATE") >= pl.lit(datetime.date(2024, 3, 25))).count().alias("tx_last_7d"),
            pl.col("TrxID").filter(pl.col("TRX_DATE") >= pl.lit(datetime.date(2024, 3, 18))).count().alias("tx_last_14d"),
            pl.col("TrxID").filter(pl.col("TRX_DATE") >= pl.lit(datetime.date(2024, 3, 1))).count().alias("tx_last_31d"),
            # Unique destinations
            pl.col("DST_ACCOUNT").n_unique().alias("n_unique_dst"),
            pl.col("DST_ACCOUNT").filter(pl.col("TRX_DATE").dt.month() == 3).n_unique().alias("n_unique_dst_march"),
            # Amount stats
            pl.col("TRX_AMT").mean().alias("avg_tx_amt"),
            pl.col("TRX_AMT").max().alias("max_tx_amt"),
            pl.col("TRX_AMT").std().alias("std_tx_amt"),
            pl.col("TRX_AMT").median().alias("median_tx_amt"),
            pl.col("TRX_AMT").filter(pl.col("TRX_DATE").dt.month() == 3).mean().alias("avg_tx_amt_march"),
            # Active days
            pl.col("TRX_DATE").n_unique().alias("n_active_days"),
            pl.col("TRX_DATE").filter(pl.col("TRX_DATE").dt.month() == 3).n_unique().alias("n_active_days_march"),
            # Unique TRX_TYPEs used
            pl.col("TRX_TYPE").n_unique().alias("n_trx_types"),
        ])
        .rename({"SRC_ACCOUNT": "ACCOUNT_ID"})
    )

    dst = (
        trx.group_by("DST_ACCOUNT")
        .agg([
            pl.col("TrxID").count().alias("F10_inbound_count"),
            pl.col("TrxID").filter(pl.col("TRX_DATE").dt.month() == 3).count().alias("inbound_count_march"),
            pl.col("TrxID").filter(pl.col("TRX_DATE").dt.month() == 2).count().alias("inbound_count_feb"),
            pl.col("TRX_AMT").sum().alias("inbound_amt_total"),
            pl.col("TRX_AMT").filter(pl.col("TRX_DATE").dt.month() == 3).sum().alias("inbound_amt_march"),
            pl.col("SRC_ACCOUNT").n_unique().alias("n_unique_senders"),
        ])
        .rename({"DST_ACCOUNT": "ACCOUNT_ID"})
    )

    bal = pl.scan_parquet(balance_pattern).with_columns([
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
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE") >= pl.lit(datetime.date(2024, 3, 25))).mean().alias("avg_bal_last_7d"),
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE") >= pl.lit(datetime.date(2024, 3, 18))).mean().alias("avg_bal_last_14d"),
            pl.col("AVAILABLE_BALANCE").filter(pl.col("AVAILABLE_BALANCE") == 0.0).count().alias("zero_balance_days"),
            pl.col("AVAILABLE_BALANCE").filter((pl.col("AVAILABLE_BALANCE") == 0.0) & (pl.col("DATE").dt.month() == 3)).count().alias("zero_balance_days_march"),
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 3).min().alias("min_bal_march"),
            pl.col("AVAILABLE_BALANCE").mean().alias("avg_bal_overall"),
            pl.col("AVAILABLE_BALANCE").max().alias("max_bal"),
            pl.col("AVAILABLE_BALANCE").min().alias("min_bal"),
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 3).std().alias("bal_std_march"),
        ])
    )

    joined = kyc.join(src, on="ACCOUNT_ID", how="left").join(dst, on="ACCOUNT_ID", how="left").join(bal_feats, on="ACCOUNT_ID", how="left")

    # Null fills
    null_cols = [
        "F1_recency_days", "total_tx_count", "tx_count_march", "tx_count_feb", "tx_count_jan",
        "tx_amt_march", "tx_amt_feb", "tx_amt_jan", "tx_amt_early",
        "amt_cashout", "amt_cashin",
        "count_merchant", "count_bill", "count_p2p", "count_cashin", "count_cashout",
        "tx_last_7d", "tx_last_14d", "tx_last_31d",
        "n_unique_dst", "n_unique_dst_march",
        "avg_tx_amt", "max_tx_amt", "std_tx_amt", "median_tx_amt", "avg_tx_amt_march",
        "n_active_days", "n_active_days_march", "n_trx_types",
        "F10_inbound_count", "inbound_count_march", "inbound_count_feb",
        "inbound_amt_total", "inbound_amt_march", "n_unique_senders",
        "F9_final_balance", "F5_balance_volatility",
        "avg_bal_march", "avg_bal_feb", "avg_bal_jan",
        "avg_bal_last_7d", "avg_bal_last_14d",
        "zero_balance_days", "zero_balance_days_march",
        "min_bal_march", "avg_bal_overall", "max_bal", "min_bal", "bal_std_march",
    ]
    filled = joined.with_columns([
        pl.col(c).fill_null(91 if c == "F1_recency_days" else 0) for c in null_cols
    ])

    # Derived features
    derived = filled.with_columns([
        # Core 10
        (pl.col("tx_count_march") / ((pl.col("tx_count_jan") + pl.col("tx_count_feb")) / 2.0 + 1.0)).alias("F2_momentum_ratio"),
        (pl.col("tx_amt_march") / (pl.col("tx_amt_early") + 1.0)).alias("F3_monetary_trend"),
        (pl.col("avg_bal_march") - pl.col("avg_bal_jan")).alias("F4_balance_depletion"),
        (pl.col("amt_cashout") / (pl.col("amt_cashin") + 1.0)).alias("F6_pass_through_index"),
        ((pl.col("count_merchant") + pl.col("count_bill")) / (pl.col("total_tx_count") + 1.0)).alias("F7_ecosystem_index"),

        # Monthly momentums
        (pl.col("tx_count_march") / (pl.col("tx_count_feb") + 1.0)).alias("momentum_feb_to_march"),
        (pl.col("tx_amt_march") / (pl.col("tx_amt_feb") + 1.0)).alias("amt_momentum_feb_to_march"),
        (pl.col("tx_count_feb") / (pl.col("tx_count_jan") + 1.0)).alias("momentum_jan_to_feb"),

        # Weekly recency ratios
        (pl.col("tx_last_7d") / (pl.col("tx_count_march") + 1.0)).alias("last_7d_ratio"),
        (pl.col("tx_last_14d") / (pl.col("tx_count_march") + 1.0)).alias("last_14d_ratio"),

        # Destination diversity
        (pl.col("n_unique_dst_march") / (pl.col("n_unique_dst") + 1.0)).alias("dst_diversity_trend"),

        # Balance trends
        (pl.col("avg_bal_march") - pl.col("avg_bal_feb")).alias("bal_depletion_feb_to_march"),
        (pl.col("avg_bal_last_7d") - pl.col("avg_bal_march")).alias("bal_trend_last_7d"),
        (pl.col("avg_bal_last_14d") - pl.col("avg_bal_march")).alias("bal_trend_last_14d"),
        (pl.col("F5_balance_volatility") / (pl.col("avg_bal_overall") + 1.0)).alias("balance_cov"),
        (pl.col("max_bal") - pl.col("min_bal")).alias("balance_range"),

        # Inbound
        (pl.col("inbound_count_march") / (pl.col("F10_inbound_count") + 1.0)).alias("inbound_march_ratio"),
        (pl.col("inbound_count_march") / (pl.col("inbound_count_feb") + 1.0)).alias("inbound_momentum_feb_march"),
        (pl.col("inbound_amt_march") / (pl.col("inbound_amt_total") + 1.0)).alias("inbound_amt_march_ratio"),

        # Type fractions
        (pl.col("count_p2p") / (pl.col("total_tx_count") + 1.0)).alias("p2p_fraction"),
        (pl.col("count_cashin") / (pl.col("total_tx_count") + 1.0)).alias("cashin_fraction"),
        (pl.col("count_cashout") / (pl.col("total_tx_count") + 1.0)).alias("cashout_fraction"),

        # Activity density
        (pl.col("total_tx_count") / (pl.col("n_active_days") + 1.0)).alias("activity_density"),
        (pl.col("tx_count_march") / (pl.col("n_active_days_march") + 1.0)).alias("activity_density_march"),

        # Dormancy flags
        (pl.col("F1_recency_days") > 30).cast(pl.Int8).alias("is_dormant_30d"),
        (pl.col("F1_recency_days") > 14).cast(pl.Int8).alias("is_dormant_14d"),
        (pl.col("F1_recency_days") > 7).cast(pl.Int8).alias("is_dormant_7d"),
        (pl.col("tx_count_march") == 0).cast(pl.Int8).alias("zero_march_txns"),
        (pl.col("tx_last_7d") == 0).cast(pl.Int8).alias("zero_last_7d_txns"),
    ])

    logger.info("Streaming collection...")
    train_pl = pl.scan_csv(train_labels_path).join(derived, on="ACCOUNT_ID", how="inner").collect(streaming=True)
    test_pl = pl.scan_csv(test_path).join(derived, on="ACCOUNT_ID", how="left").collect(streaming=True)
    train_df = train_pl.to_pandas()
    test_df = test_pl.to_pandas()
    logger.info(f"Raw features: Train {train_df.shape}, Test {test_df.shape}")
    return train_df, test_df


# ===========================================================
# PHASE 2 -- PREPROCESSING + INTERACTION FEATURES
# ===========================================================
def preprocess(train_df, test_df):
    logger.info("[Phase 2] Preprocessing + interaction features...")

    exclude = {"ACCOUNT_ID", "ACCOUNT_TYPE", "ACCOUNT_OPEN_DATE", "CHURN"}
    num_cols = [c for c in train_df.columns if c not in exclude and train_df[c].dtype in [np.float64, np.int64, np.int32, np.int16, np.int8, np.float32]]

    for col in num_cols:
        cap = train_df[col].quantile(0.99)
        if cap > 0:
            train_df[col] = train_df[col].clip(upper=cap)
            test_df[col] = test_df[col].clip(upper=cap)

    log_cols = [
        "total_tx_count", "tx_amt_march", "tx_amt_early", "tx_amt_feb", "tx_amt_jan",
        "F10_inbound_count", "F9_final_balance", "F5_balance_volatility",
        "inbound_amt_total", "inbound_amt_march", "max_tx_amt", "avg_tx_amt",
        "avg_bal_overall", "amt_cashout", "amt_cashin", "balance_range",
        "n_unique_dst", "n_unique_senders", "median_tx_amt",
    ]
    for col in log_cols:
        if col in train_df.columns:
            train_df[f"log_{col}"] = np.log1p(train_df[col])
            test_df[f"log_{col}"] = np.log1p(test_df[col])

    # ---- INTERACTION FEATURES ----
    interactions = [
        ("F1_recency_days", "F9_final_balance", "mul"),
        ("F1_recency_days", "F2_momentum_ratio", "mul"),
        ("F1_recency_days", "F10_inbound_count", "mul"),
        ("F2_momentum_ratio", "F7_ecosystem_index", "mul"),
        ("F8_vintage_days", "F1_recency_days", "div"),
        ("F9_final_balance", "F5_balance_volatility", "div"),
        ("tx_count_march", "F10_inbound_count", "sum"),
        ("is_dormant_30d", "zero_march_txns", "sum"),
        ("momentum_feb_to_march", "amt_momentum_feb_to_march", "mul"),
        ("tx_last_7d", "F9_final_balance", "mul"),
        ("F6_pass_through_index", "F1_recency_days", "mul"),
        ("balance_cov", "F1_recency_days", "mul"),
        ("n_unique_dst_march", "tx_count_march", "div"),
        ("inbound_march_ratio", "momentum_feb_to_march", "mul"),
    ]

    for c1, c2, op in interactions:
        if c1 in train_df.columns and c2 in train_df.columns:
            name = f"ix_{c1[:8]}_{c2[:8]}_{op}"
            if op == "mul":
                train_df[name] = train_df[c1] * train_df[c2]
                test_df[name] = test_df[c1] * test_df[c2]
            elif op == "div":
                train_df[name] = train_df[c1] / (train_df[c2] + 1.0)
                test_df[name] = test_df[c1] / (test_df[c2] + 1.0)
            elif op == "sum":
                train_df[name] = train_df[c1] + train_df[c2]
                test_df[name] = test_df[c1] + test_df[c2]

    # ---- RANK FEATURES ----
    rank_cols = ["F1_recency_days", "F9_final_balance", "total_tx_count", "F10_inbound_count", "F8_vintage_days"]
    for col in rank_cols:
        if col in train_df.columns:
            train_df[f"rank_{col}"] = train_df[col].rank(pct=True)
            test_df[f"rank_{col}"] = test_df[col].rank(pct=True)

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    logger.info(f"Final features: Train {train_df.shape}, Test {test_df.shape}")
    return train_df, test_df


# ===========================================================
# PHASE 3 -- OPTUNA TUNING
# ===========================================================
def tune_lgb(X, y):
    logger.info("[Phase 3] Optuna LGB tuning (250k sample, 60 trials)...")
    n_sample = min(250000, len(X))
    idx = np.random.RandomState(42).choice(len(X), n_sample, replace=False)
    Xs, ys = X.iloc[idx], y.iloc[idx]
    spw = (ys == 0).sum() / ((ys == 1).sum() + 1)

    # Detect GPU availability
    has_gpu = False
    has_lgb_gpu = False
    try:
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count() > 0:
            has_gpu = True
    except Exception:
        pass

    if has_gpu:
        try:
            dummy_X = np.random.rand(100, 10)
            dummy_y = np.random.randint(0, 2, 100)
            dummy_ds = lgb.Dataset(dummy_X, label=dummy_y)
            dummy_params = {"objective": "binary", "device": "gpu", "verbose": -1}
            lgb.train(dummy_params, dummy_ds, num_boost_round=1)
            has_lgb_gpu = True
            logger.info("LightGBM GPU support successfully verified and enabled.")
        except Exception as e:
            logger.info(f"LightGBM GPU dry-run failed ({e}). Defaulting to CPU for LightGBM.")

    def objective(trial):
        p = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("lr", 0.008, 0.06, log=True),
            "num_leaves": trial.suggest_int("nl", 31, 255),
            "max_depth": trial.suggest_int("md", 5, 12),
            "min_child_samples": trial.suggest_int("mcs", 20, 400),
            "feature_fraction": trial.suggest_float("ff", 0.4, 0.95),
            "bagging_fraction": trial.suggest_float("bf", 0.5, 0.95),
            "bagging_freq": 1,
            "reg_alpha": trial.suggest_float("ra", 1e-4, 20.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 1e-4, 20.0, log=True),
            "min_gain_to_split": trial.suggest_float("mg", 0.0, 0.5),
            "scale_pos_weight": spw,
            "verbose": -1, "n_jobs": -1, "random_state": 42,
        }
        if has_lgb_gpu:
            p["device"] = "gpu"

        folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        scores = []
        for ti, vi in folds.split(Xs, ys):
            ds_t = lgb.Dataset(Xs.iloc[ti], label=ys.iloc[ti])
            ds_v = lgb.Dataset(Xs.iloc[vi], label=ys.iloc[vi])
            m = lgb.train(p, ds_t, 1000, valid_sets=[ds_v],
                          callbacks=[lgb.early_stopping(40, verbose=False)])
            scores.append(roc_auc_score(ys.iloc[vi], m.predict(Xs.iloc[vi], num_iteration=m.best_iteration)))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    logger.info(f"Best LGB trial: {study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")

    try:
        plt.figure(figsize=(10, 5))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig("explainability/optuna_history_v3.png", dpi=150)
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
        "min_gain_to_split": study.best_params["mg"],
    }


# ===========================================================
# PHASE 4 -- MULTI-SEED STACKING ENSEMBLE
# ===========================================================
def train_stacking_ensemble(train_df, test_df, best_lgb_params):
    logger.info("[Phase 4] Multi-seed stacking ensemble...")

    exclude = {"ACCOUNT_ID", "ACCOUNT_TYPE", "ACCOUNT_OPEN_DATE", "CHURN"}
    feat_cols = [c for c in train_df.columns if c not in exclude]
    X = train_df[feat_cols].copy()
    y = train_df["CHURN"]
    X_test = test_df[feat_cols].copy()

    for col in ["GENDER", "REGION"]:
        X[col] = X[col].astype(int)
        X_test[col] = X_test[col].astype(int)

    spw = (y == 0).sum() / ((y == 1).sum() + 1)

    all_oof = {"lgb": [], "xgb": [], "cat": []}
    all_tp = {"lgb": [], "xgb": [], "cat": []}

    # Detect GPU availability
    has_gpu = False
    has_lgb_gpu = False
    try:
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count() > 0:
            has_gpu = True
    except Exception:
        pass

    if has_gpu:
        try:
            dummy_X = np.random.rand(100, 10)
            dummy_y = np.random.randint(0, 2, 100)
            dummy_ds = lgb.Dataset(dummy_X, label=dummy_y)
            dummy_params = {"objective": "binary", "device": "gpu", "verbose": -1}
            lgb.train(dummy_params, dummy_ds, num_boost_round=1)
            has_lgb_gpu = True
        except Exception:
            pass

    for seed in SEEDS:
        logger.info(f"--- Seed {seed} ---")
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        oof_lgb = np.zeros(len(X))
        oof_xgb = np.zeros(len(X))
        oof_cat = np.zeros(len(X))
        tp_lgb = np.zeros(len(X_test))
        tp_xgb = np.zeros(len(X_test))
        tp_cat = np.zeros(len(X_test))

        lgb_params = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "scale_pos_weight": spw,
            "bagging_freq": 1, "verbose": -1, "n_jobs": -1, "random_state": seed,
            **best_lgb_params,
        }
        if has_lgb_gpu:
            lgb_params["device"] = "gpu"

        xgb_params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "learning_rate": 0.025, "max_depth": 7, "scale_pos_weight": spw,
            "subsample": 0.8, "colsample_bytree": 0.75,
            "reg_alpha": 0.1, "reg_lambda": 2.0,
            "tree_method": "hist", "n_jobs": -1, "random_state": seed,
        }
        if has_gpu:
            xgb_params["device"] = "cuda"

        models_lgb_seed = []
        for fold, (ti, vi) in enumerate(folds.split(X, y)):
            Xtr, ytr = X.iloc[ti], y.iloc[ti]
            Xva, yva = X.iloc[vi], y.iloc[vi]

            # LightGBM
            ds_t = lgb.Dataset(Xtr, label=ytr)
            ds_v = lgb.Dataset(Xva, label=yva)
            m_lgb = lgb.train(lgb_params, ds_t, 3000, valid_sets=[ds_t, ds_v],
                              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            oof_lgb[vi] = m_lgb.predict(Xva, num_iteration=m_lgb.best_iteration)
            tp_lgb += m_lgb.predict(X_test, num_iteration=m_lgb.best_iteration) / 5
            models_lgb_seed.append(m_lgb)

            # XGBoost
            m_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=3000, early_stopping_rounds=100)
            m_xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            oof_xgb[vi] = m_xgb.predict_proba(Xva)[:, 1]
            tp_xgb += m_xgb.predict_proba(X_test)[:, 1] / 5

            # CatBoost
            m_cat = CatBoostClassifier(
                iterations=3000, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
                scale_pos_weight=spw, eval_metric="AUC",
                early_stopping_rounds=100, verbose=0, random_seed=seed,
                task_type="GPU" if has_gpu else "CPU",
            )
            m_cat.fit(Xtr, ytr, eval_set=(Xva, yva))
            oof_cat[vi] = m_cat.predict_proba(Xva)[:, 1]
            tp_cat += m_cat.predict_proba(X_test)[:, 1] / 5

        auc_l = roc_auc_score(y, oof_lgb)
        auc_x = roc_auc_score(y, oof_xgb)
        auc_c = roc_auc_score(y, oof_cat)
        logger.info(f"  Seed {seed} => LGB={auc_l:.6f}, XGB={auc_x:.6f}, CAT={auc_c:.6f} (GPU={has_gpu})")

        all_oof["lgb"].append(oof_lgb)
        all_oof["xgb"].append(oof_xgb)
        all_oof["cat"].append(oof_cat)
        all_tp["lgb"].append(tp_lgb)
        all_tp["xgb"].append(tp_xgb)
        all_tp["cat"].append(tp_cat)

    # Average across seeds
    oof_lgb_avg = np.mean(all_oof["lgb"], axis=0)
    oof_xgb_avg = np.mean(all_oof["xgb"], axis=0)
    oof_cat_avg = np.mean(all_oof["cat"], axis=0)
    tp_lgb_avg = np.mean(all_tp["lgb"], axis=0)
    tp_xgb_avg = np.mean(all_tp["xgb"], axis=0)
    tp_cat_avg = np.mean(all_tp["cat"], axis=0)

    logger.info(f"Seed-avg LGB AUC: {roc_auc_score(y, oof_lgb_avg):.6f}")
    logger.info(f"Seed-avg XGB AUC: {roc_auc_score(y, oof_xgb_avg):.6f}")
    logger.info(f"Seed-avg CAT AUC: {roc_auc_score(y, oof_cat_avg):.6f}")

    # Stacking
    logger.info("Training stacking meta-learner (LogisticRegression)...")
    meta_X = np.column_stack([oof_lgb_avg, oof_xgb_avg, oof_cat_avg])
    meta_X_test = np.column_stack([tp_lgb_avg, tp_xgb_avg, tp_cat_avg])

    meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    meta_oof = np.zeros(len(y))
    meta_test = np.zeros(len(X_test))
    folds_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for ti, vi in folds_meta.split(meta_X, y):
        meta.fit(meta_X[ti], y.iloc[ti])
        meta_oof[vi] = meta.predict_proba(meta_X[vi])[:, 1]
        meta_test += meta.predict_proba(meta_X_test)[:, 1] / 5

    stack_auc = roc_auc_score(y, meta_oof)
    logger.info(f"Stacked ensemble AUC: {stack_auc:.6f}")

    # Optimal Linear Blend
    best_w, best_auc = None, 0
    for w1 in np.arange(0.3, 0.8, 0.02):
        for w2 in np.arange(0.05, 0.4, 0.02):
            w3 = 1.0 - w1 - w2
            if w3 < 0.02:
                continue
            blend = w1 * oof_lgb_avg + w2 * oof_xgb_avg + w3 * oof_cat_avg
            a = roc_auc_score(y, blend)
            if a > best_auc:
                best_auc = a
                best_w = (w1, w2, w3)

    w1, w2, w3 = best_w
    blend_oof = w1 * oof_lgb_avg + w2 * oof_xgb_avg + w3 * oof_cat_avg
    blend_test = w1 * tp_lgb_avg + w2 * tp_xgb_avg + w3 * tp_cat_avg
    logger.info(f"Linear blend AUC: {best_auc:.6f} (w={best_w})")

    if stack_auc >= best_auc:
        logger.info("=> Using STACKED predictions")
        final_oof, final_test = meta_oof, meta_test
        final_auc = stack_auc
    else:
        logger.info("=> Using LINEAR BLEND predictions")
        final_oof, final_test = blend_oof, blend_test
        final_auc = best_auc

    def report(name, yt, yp):
        auc = roc_auc_score(yt, yp)
        pr = average_precision_score(yt, yp)
        thr = np.percentile(yp, 90)
        yb = (yp >= thr).astype(int)
        p10 = precision_score(yt, yb)
        r10 = recall_score(yt, yb)
        logger.info(f"[{name}] AUC={auc:.6f} | PR-AUC={pr:.5f} | P@10%={p10:.5f} | R@10%={r10:.5f}")
        return {"AUC": auc, "PR-AUC": pr, "Precision@10%": p10, "Recall@10%": r10}

    lgb_m = report("LGB-avg", y, oof_lgb_avg)
    xgb_m = report("XGB-avg", y, oof_xgb_avg)
    cat_m = report("CAT-avg", y, oof_cat_avg)
    final_m = report("FINAL", y, final_oof)

    return models_lgb_seed, final_test, lgb_m, xgb_m, cat_m, final_m, X, y


# ===========================================================
# PHASE 5 -- SHAP + PREDICTIONS
# ===========================================================
def generate_shap(model, X):
    logger.info("[Phase 5] SHAP...")
    Xs = X.sample(n=min(3000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xs)

    plt.figure(figsize=(12, 7))
    shap.plots.beeswarm(sv, max_display=20, show=False)
    plt.title("SHAP Beeswarm - V3", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("explainability/shap_beeswarm_v3.png", dpi=150)
    plt.close()

    for feat in ["F1_recency_days", "F2_momentum_ratio", "tx_last_7d", "momentum_feb_to_march",
                  "ix_F1_recen_F9_final_mul", "ix_F1_recen_F2_momen_mul"]:
        if feat in Xs.columns:
            plt.figure(figsize=(8, 5))
            shap.plots.scatter(sv[:, feat], color=sv, show=False)
            plt.title(f"SHAP Dependence: {feat}", fontsize=12, pad=10)
            plt.tight_layout()
            plt.savefig(f"explainability/shap_dep_{feat}_v3.png", dpi=150)
            plt.close()
    logger.info("SHAP plots saved.")


def save_predictions(test_df, preds):
    sub = test_df[["ACCOUNT_ID"]].copy()
    sub["CHURN_PROB"] = preds
    assert len(sub) == 255000
    assert sub["CHURN_PROB"].between(0, 1).all()
    sub.to_csv("predictions_v3.csv", index=False)
    logger.info("predictions_v3.csv saved (255,000 rows).")


# ===========================================================
# MAIN
# ===========================================================
def main():
    logger.info("=== FictiPay V3 Pipeline - Target AUC > 0.985 ===")

    train_df, test_df = extract_features()
    train_df, test_df = preprocess(train_df, test_df)

    exclude = {"ACCOUNT_ID", "ACCOUNT_TYPE", "ACCOUNT_OPEN_DATE", "CHURN"}
    feat_cols = [c for c in train_df.columns if c not in exclude]
    X_tune = train_df[feat_cols].copy()
    y_tune = train_df["CHURN"]
    for col in ["GENDER", "REGION"]:
        X_tune[col] = X_tune[col].astype(int)

    best_lgb = tune_lgb(X_tune, y_tune)
    models, preds, lgb_m, xgb_m, cat_m, final_m, X, y = train_stacking_ensemble(train_df, test_df, best_lgb)

    generate_shap(models[0], X)
    save_predictions(test_df, preds)

    logger.info(f"=== V3 Complete | Final AUC-ROC: {final_m['AUC']:.6f} ===")

if __name__ == "__main__":
    main()

"""
FictiPay V5 Pipeline -- Advanced Data Cleaning & Stacking Blend Optimization
Target: AUC-ROC > 0.985 (Goal: 0.99+)
Author: Shayer Mahmud Sowmik

Key Improvements in V5:
  1. Weekly Time-Series Decomposition:
     - 13 weekly counts and amount aggregates (weeks 0 to 12).
     - Vectorized consecutive zero week streak detection.
     - Vectorized weekly counts and amounts linear regression slopes.
  2. DST_ACCOUNT Graph Features:
     - Proportions of transactions sent to Merchants, Billers, and Customers.
     - Count of unique Merchant and Biller destinations.
  3. Balance Trajectory Shapes:
     - Net balance change, daily increase/decrease days, maximum daily balance drop in March.
     - Days with zero balance in the last 14 days of March.
  4. Time-of-Day and Day-of-Week Behavioral Patterns:
     - Business hour, evening, night, and weekend transaction ratios.
  5. Distribution Shapes and Inbound Details:
     - Skewness, kurtosis, percentiles (IQR) of TRX_AMT.
     - Social network size (unique P2P inbound senders), standard deviation of CashIn deposits.
  6. Advanced Stacking & Blending:
     - Out-of-fold target encoding for REGION to prevent leakage.
     - Evaluation of both probability and rank-average blending.
     - Nelder-Mead refined optimization starting from grid search points.
     - Isotonic regression calibration of rank-averaged predictions.
"""
import os, sys, datetime, logging, warnings
import polars as pl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from scipy.optimize import minimize
from scipy.stats import rankdata
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
import shap

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("pipeline_v5.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

OBSERVATION_END = datetime.date(2024, 3, 31)
DATA = "bkash-presents-nsucec-datathon/public"
os.makedirs("explainability", exist_ok=True)
SEEDS = [42, 123, 2024]  # Multi-seed stacking


# ===========================================================
# PHASE 1 -- FEATURE EXTRACTION (Polars & Streaming)
# ===========================================================
def extract_features():
    logger.info("[Phase 1] V5 Feature Extraction starting...")

    # Load KYC metadata - do not filter Customer yet to preserve other account types for graph features
    kyc_full = pl.scan_parquet(f"{DATA}/kyc.parquet")
    kyc = kyc_full.filter(pl.col("ACCOUNT_TYPE") == "Customer")
    
    # Clean high-cardinality rare REGION categories (freq < 100 grouped into "RARE")
    region_counts = kyc.group_by("REGION").len().collect()
    rare_regions = region_counts.filter(pl.col("len") < 100).select("REGION").to_series().to_list()
    
    kyc = kyc.with_columns([
        pl.when(pl.col("REGION").is_in(rare_regions))
        .then(pl.lit("RARE"))
        .otherwise(pl.col("REGION"))
        .alias("REGION")
    ])

    kyc = kyc.with_columns([
        ((pl.lit(OBSERVATION_END) - pl.col("ACCOUNT_OPEN_DATE").dt.date()).dt.total_days()).alias("F8_vintage_days"),
        # Cast categorical columns as physical code ints for models
        pl.col("GENDER").cast(pl.Categorical).to_physical().cast(pl.Int8),
        pl.col("REGION").cast(pl.Categorical).to_physical().cast(pl.Int16),
    ])

    # Scan Transactions
    logger.info("Scanning transactions...")
    trx = pl.scan_parquet(f"{DATA}/transactions/trx_2024-0*.parquet").with_columns([
        pl.col("TRX_DATETIME").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("TRX_DATE"),
        pl.col("TRX_DATETIME").str.slice(11, 2).cast(pl.Int8).alias("TRX_HOUR"),
    ]).with_columns([
        pl.col("TRX_DATE").dt.weekday().alias("TRX_DOW"),
        ((pl.lit(OBSERVATION_END) - pl.col("TRX_DATE")).dt.total_days()).alias("days_diff")
    ])

    # Outbound features (grouped by SRC_ACCOUNT)
    logger.info("Aggregating outbound features...")
    agg_exprs = [
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
        # Unique types
        pl.col("TRX_TYPE").n_unique().alias("n_trx_types"),

        # --- NEW V5 FEATURES ---
        # Time-of-day / Day-of-week patterns
        (pl.col("TRX_HOUR").is_between(9, 17)).cast(pl.Float32).mean().alias("frac_tx_business_hours"),
        ((pl.col("TRX_HOUR") >= 22) | (pl.col("TRX_HOUR") < 6)).cast(pl.Float32).mean().alias("frac_tx_night"),
        (pl.col("TRX_DOW") >= 6).cast(pl.Float32).mean().alias("frac_tx_weekend"),
        pl.col("TRX_HOUR").std().alias("trx_hour_std"),
        pl.col("TRX_DOW").std().alias("trx_dow_std"),

        # Amount distribution shapes
        pl.col("TRX_AMT").skew().alias("amount_skew"),
        pl.col("TRX_AMT").kurtosis().alias("amount_kurtosis"),
        pl.col("TRX_AMT").quantile(0.25).alias("amount_p25"),
        pl.col("TRX_AMT").quantile(0.75).alias("amount_p75"),
        (pl.col("TRX_AMT") < 100.0).cast(pl.Float32).mean().alias("frac_tx_under_100"),
        (pl.col("TRX_AMT") > 10000.0).cast(pl.Float32).mean().alias("frac_tx_over_10000"),
    ]

    # Append 13 weekly counts and amount aggregates (w0 is closest to observation end, w12 is earliest)
    for w in range(13):
        agg_exprs.append(
            pl.col("TrxID").filter((pl.col("days_diff") >= w*7) & (pl.col("days_diff") < (w+1)*7)).count().alias(f"tx_count_w{w}")
        )
        agg_exprs.append(
            pl.col("TRX_AMT").filter((pl.col("days_diff") >= w*7) & (pl.col("days_diff") < (w+1)*7)).sum().alias(f"tx_amt_w{w}")
        )

    src = (
        trx.group_by("SRC_ACCOUNT")
        .agg(agg_exprs)
        .rename({"SRC_ACCOUNT": "ACCOUNT_ID"})
    )

    # Graph network topology (joins transactions with the full KYC database on destination account)
    logger.info("Aggregating graph topology features...")
    trx_with_dst = trx.join(
        kyc_full.select(["ACCOUNT_ID", "ACCOUNT_TYPE"]),
        left_on="DST_ACCOUNT",
        right_on="ACCOUNT_ID",
        how="left"
    )
    dst_graph = (
        trx_with_dst.group_by("SRC_ACCOUNT")
        .agg([
            (pl.col("ACCOUNT_TYPE") == "Merchant").cast(pl.Float32).mean().alias("frac_dst_merchant"),
            (pl.col("ACCOUNT_TYPE") == "Biller").cast(pl.Float32).mean().alias("frac_dst_biller"),
            (pl.col("ACCOUNT_TYPE") == "Customer").cast(pl.Float32).mean().alias("frac_dst_customer"),
            pl.col("DST_ACCOUNT").filter(pl.col("TRX_TYPE") == "MerchantPay").n_unique().alias("n_unique_merchants"),
            pl.col("DST_ACCOUNT").filter((pl.col("TRX_TYPE") == "MerchantPay") & (pl.col("TRX_DATE").dt.month() == 3)).n_unique().alias("n_unique_merchants_march"),
            pl.col("DST_ACCOUNT").filter(pl.col("TRX_TYPE") == "BillPay").n_unique().alias("n_unique_billers"),
            pl.col("DST_ACCOUNT").filter(pl.col("TRX_TYPE") == "P2P").n_unique().alias("n_unique_customers_out"),
        ])
        .rename({"SRC_ACCOUNT": "ACCOUNT_ID"})
    )

    # Inbound transaction detail features (grouped by DST_ACCOUNT)
    logger.info("Aggregating inbound details...")
    dst = (
        trx.group_by("DST_ACCOUNT")
        .agg([
            pl.col("TrxID").count().alias("F10_inbound_count"),
            pl.col("TrxID").filter(pl.col("TRX_DATE").dt.month() == 3).count().alias("inbound_count_march"),
            pl.col("TrxID").filter(pl.col("TRX_DATE").dt.month() == 2).count().alias("inbound_count_feb"),
            pl.col("TRX_AMT").sum().alias("inbound_amt_total"),
            pl.col("TRX_AMT").filter(pl.col("TRX_DATE").dt.month() == 3).sum().alias("inbound_amt_march"),
            pl.col("SRC_ACCOUNT").n_unique().alias("n_unique_senders"),
            
            # --- NEW INBOUND DETAILS ---
            pl.col("SRC_ACCOUNT").filter(pl.col("TRX_TYPE") == "P2P").n_unique().alias("n_unique_p2p_senders"),
            pl.col("TRX_AMT").filter(pl.col("TRX_TYPE") == "CashIn").std().alias("cashin_amt_std"),
            ((pl.lit(OBSERVATION_END) - pl.col("TRX_DATE").max()).dt.total_days()).alias("inbound_recency_days"),
        ])
        .rename({"DST_ACCOUNT": "ACCOUNT_ID"})
    )

    # Day-end balances (daily trajectory features)
    logger.info("Scanning day-end balances...")
    bal = pl.scan_parquet(f"{DATA}/dayend_balance/balance_2024-0*.parquet").with_columns([
        pl.col("DATE").str.to_date("%Y-%m-%d").alias("DATE"),
    ]).sort("DATE")
    
    logger.info("Aggregating balance trajectory features...")
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
            
            # --- NEW BALANCE TRAJECTORY ---
            (pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 3).last() - pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 3).first()).alias("bal_net_change_march"),
            (pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 3).diff() > 0).cast(pl.Int32).sum().alias("bal_up_days_march"),
            (pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 3).diff() < 0).cast(pl.Int32).sum().alias("bal_down_days_march"),
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 3).diff().min().alias("max_daily_drop_march"),
            (pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE") >= pl.lit(datetime.date(2024, 3, 18))) == 0.0).cast(pl.Int32).sum().alias("zero_days_last14"),
        ])
    )

    logger.info("Joining all LazyFrames...")
    joined = (
        kyc
        .join(src, on="ACCOUNT_ID", how="left")
        .join(dst, on="ACCOUNT_ID", how="left")
        .join(dst_graph, on="ACCOUNT_ID", how="left")
        .join(bal_feats, on="ACCOUNT_ID", how="left")
    )

    # Explicit null fill lists
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
        # New weekly columns
        *[f"tx_count_w{i}" for i in range(13)],
        *[f"tx_amt_w{i}" for i in range(13)],
        # Hour/dow
        "frac_tx_business_hours", "frac_tx_night", "frac_tx_weekend", "trx_hour_std", "trx_dow_std",
        # Amount stats
        "amount_skew", "amount_kurtosis", "amount_p25", "amount_p75", "frac_tx_under_100", "frac_tx_over_10000",
        # Graph features
        "frac_dst_merchant", "frac_dst_biller", "frac_dst_customer",
        "n_unique_merchants", "n_unique_merchants_march", "n_unique_billers", "n_unique_customers_out",
        # Inbound details
        "n_unique_p2p_senders", "cashin_amt_std", "inbound_recency_days",
        # Balance trajectory
        "bal_net_change_march", "bal_up_days_march", "bal_down_days_march", "max_daily_drop_march", "zero_days_last14"
    ]
    
    filled = joined.with_columns([
        pl.col(c).fill_null(91 if c in ["F1_recency_days", "inbound_recency_days"] else 0) for c in null_cols
    ])

    # Derived Features (Phase 1.5 in Polars)
    derived = filled.with_columns([
        # Core 10
        (pl.col("tx_count_march") / ((pl.col("tx_count_jan") + pl.col("tx_count_feb")) / 2.0 + 1.0)).alias("F2_momentum_ratio"),
        (pl.col("tx_amt_march") / (pl.col("tx_amt_early") + 1.0)).alias("F3_monetary_trend"),
        (pl.col("avg_bal_march") - pl.col("avg_bal_jan")).alias("F4_balance_depletion"),
        (pl.col("amt_cashout") / (pl.col("amt_cashin") + 1.0)).alias("F6_pass_through_index"),
        ((pl.col("count_merchant") + pl.col("count_bill")) / (pl.col("total_tx_count") + 1.0)).alias("F7_ecosystem_index"),

        # Momentum trends
        (pl.col("tx_count_march") / (pl.col("tx_count_feb") + 1.0)).alias("momentum_feb_to_march"),
        (pl.col("tx_amt_march") / (pl.col("tx_amt_feb") + 1.0)).alias("amt_momentum_feb_to_march"),
        (pl.col("tx_count_feb") / (pl.col("tx_count_jan") + 1.0)).alias("momentum_jan_to_feb"),

        # Windows
        (pl.col("tx_last_7d") / (pl.col("tx_count_march") + 1.0)).alias("last_7d_ratio"),
        (pl.col("tx_last_14d") / (pl.col("tx_count_march") + 1.0)).alias("last_14d_ratio"),

        # Destination diversity
        (pl.col("n_unique_dst_march") / (pl.col("n_unique_dst") + 1.0)).alias("dst_diversity_trend"),

        # Balance depletion
        (pl.col("avg_bal_march") - pl.col("avg_bal_feb")).alias("bal_depletion_feb_to_march"),
        (pl.col("avg_bal_last_7d") - pl.col("avg_bal_march")).alias("bal_trend_last_7d"),
        (pl.col("avg_bal_last_14d") - pl.col("avg_bal_march")).alias("bal_trend_last_14d"),
        (pl.col("F5_balance_volatility") / (pl.col("avg_bal_overall") + 1.0)).alias("balance_cov"),
        (pl.col("max_bal") - pl.col("min_bal")).alias("balance_range"),

        # Inbound
        (pl.col("inbound_count_march") / (pl.col("F10_inbound_count") + 1.0)).alias("inbound_march_ratio"),
        (pl.col("inbound_count_march") / (pl.col("inbound_count_feb") + 1.0)).alias("inbound_momentum_feb_march"),
        (pl.col("inbound_amt_march") / (pl.col("inbound_amt_total") + 1.0)).alias("inbound_amt_march_ratio"),

        # Fractions
        (pl.col("count_p2p") / (pl.col("total_tx_count") + 1.0)).alias("p2p_fraction"),
        (pl.col("count_cashin") / (pl.col("total_tx_count") + 1.0)).alias("cashin_fraction"),
        (pl.col("count_cashout") / (pl.col("total_tx_count") + 1.0)).alias("cashout_fraction"),

        # Active density
        (pl.col("total_tx_count") / (pl.col("n_active_days") + 1.0)).alias("activity_density"),
        (pl.col("tx_count_march") / (pl.col("n_active_days_march") + 1.0)).alias("activity_density_march"),

        # Dormancies
        (pl.col("F1_recency_days") > 30).cast(pl.Int8).alias("is_dormant_30d"),
        (pl.col("F1_recency_days") > 14).cast(pl.Int8).alias("is_dormant_14d"),
        (pl.col("F1_recency_days") > 7).cast(pl.Int8).alias("is_dormant_7d"),
        (pl.col("tx_count_march") == 0).cast(pl.Int8).alias("zero_march_txns"),
        (pl.col("tx_last_7d") == 0).cast(pl.Int8).alias("zero_last_7d_txns"),
    ])

    logger.info("Streaming collection...")
    train_pl = pl.scan_csv(f"{DATA}/train_labels.csv").join(derived, on="ACCOUNT_ID", how="inner").collect(streaming=True)
    test_pl = pl.scan_csv(f"{DATA}/test.csv").join(derived, on="ACCOUNT_ID", how="left").collect(streaming=True)
    
    train_df = train_pl.to_pandas()
    test_df = test_pl.to_pandas()
    
    # Vectorized Pandas/Numpy derived operations
    logger.info("Computing consecutive zero-activity weeks and trend slopes...")
    
    # 1. Outbound consecutive zero weeks
    weekly_count_cols = [f"tx_count_w{i}" for i in range(13)]
    
    counts_matrix = train_df[weekly_count_cols].values
    is_active = (counts_matrix > 0)
    any_active = is_active.any(axis=1)
    train_df["consecutive_zero_weeks"] = np.where(any_active, is_active.argmax(axis=1), 13)
    train_df["weeks_since_last_active"] = train_df["F1_recency_days"] // 7
    
    counts_matrix_test = test_df[weekly_count_cols].values
    is_active_test = (counts_matrix_test > 0)
    any_active_test = is_active_test.any(axis=1)
    test_df["consecutive_zero_weeks"] = np.where(any_active_test, is_active_test.argmax(axis=1), 13)
    test_df["weeks_since_last_active"] = test_df["F1_recency_days"] // 7

    # 2. Weekly count linear slope (time index 0 to 12 represents weeks 12 to 0)
    y_count = train_df[[f"tx_count_w{i}" for i in range(12, -1, -1)]].values
    i_indices = np.arange(13)
    sum_y = y_count.sum(axis=1)
    sum_i_y = (y_count * i_indices).sum(axis=1)
    train_df["weekly_trend_slope"] = (13 * sum_i_y - 78 * sum_y) / 2366.0
    
    y_count_test = test_df[[f"tx_count_w{i}" for i in range(12, -1, -1)]].values
    sum_y_test = y_count_test.sum(axis=1)
    sum_i_y_test = (y_count_test * i_indices).sum(axis=1)
    test_df["weekly_trend_slope"] = (13 * sum_i_y_test - 78 * sum_y_test) / 2366.0

    # 3. Weekly amount linear slope
    y_amt = train_df[[f"tx_amt_w{i}" for i in range(12, -1, -1)]].values
    sum_y_amt = y_amt.sum(axis=1)
    sum_i_y_amt = (y_amt * i_indices).sum(axis=1)
    train_df["weekly_amt_trend_slope"] = (13 * sum_i_y_amt - 78 * sum_y_amt) / 2366.0
    
    y_amt_test = test_df[[f"tx_amt_w{i}" for i in range(12, -1, -1)]].values
    sum_y_amt_test = y_amt_test.sum(axis=1)
    sum_i_y_amt_test = (y_amt_test * i_indices).sum(axis=1)
    test_df["weekly_amt_trend_slope"] = (13 * sum_i_y_amt_test - 78 * sum_y_amt_test) / 2366.0

    logger.info(f"Raw features loaded: Train {train_df.shape}, Test {test_df.shape}")
    return train_df, test_df


# ===========================================================
# PHASE 2 -- ADVANCED DATA CLEANING
# ===========================================================
def symlog(x):
    """Symmetric Log transform to handle negative overdraft balances/slopes without losing sign info"""
    return np.sign(x) * np.log1p(np.abs(x))


def preprocess(train_df, test_df):
    logger.info("[Phase 2] Preprocessing & Advanced Data Cleaning...")

    exclude = {"ACCOUNT_ID", "ACCOUNT_TYPE", "ACCOUNT_OPEN_DATE", "CHURN", "GENDER", "REGION"}
    num_cols = [c for c in train_df.columns if c not in exclude and train_df[c].dtype in [np.float64, np.int64, np.int32, np.int16, np.int8, np.float32]]

    # Winsorization of outliers at 99th percentile
    for col in num_cols:
        cap = train_df[col].quantile(0.99)
        if cap > 0:
            train_df[col] = train_df[col].clip(upper=cap)
            test_df[col] = test_df[col].clip(upper=cap)

    # Clean zero-variance features
    std_devs = train_df[num_cols].std()
    zero_var_cols = std_devs[std_devs == 0].index.tolist()
    if zero_var_cols:
        logger.info(f"Dropping zero-variance features: {zero_var_cols}")
        train_df = train_df.drop(columns=zero_var_cols)
        test_df = test_df.drop(columns=zero_var_cols)
        num_cols = [c for c in num_cols if c not in zero_var_cols]

    # Apply Symmetric Log (Preserves sign of overdraft balances/net change/slopes/drops)
    balance_cols = [
        "F9_final_balance", "avg_bal_march", "avg_bal_feb", "avg_bal_jan", 
        "avg_bal_last_7d", "avg_bal_last_14d", "min_bal_march", "avg_bal_overall", 
        "max_bal", "min_bal", "bal_net_change_march", "max_daily_drop_march"
    ]
    for col in balance_cols:
        if col in train_df.columns:
            train_df[f"symlog_{col}"] = symlog(train_df[col])
            test_df[f"symlog_{col}"] = symlog(test_df[col])

    # Apply Standard Log to count & txn amount variables
    log_cols = [
        "total_tx_count", "tx_amt_march", "tx_amt_early", "tx_amt_feb", "tx_amt_jan",
        "F10_inbound_count", "F5_balance_volatility",
        "inbound_amt_total", "inbound_amt_march", "max_tx_amt", "avg_tx_amt",
        "amt_cashout", "amt_cashin", "balance_range",
        "n_unique_dst", "n_unique_senders", "median_tx_amt",
        # New V5 log features
        "n_unique_merchants", "n_unique_merchants_march", "n_unique_billers", "n_unique_customers_out",
        "n_unique_p2p_senders", "cashin_amt_std"
    ]
    for col in log_cols:
        if col in train_df.columns:
            train_df[f"log_{col}"] = np.log1p(train_df[col])
            test_df[f"log_{col}"] = np.log1p(test_df[col])

    # Feature Interactions
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
        
        # --- NEW V5 INTERACTIONS ---
        ("frac_dst_merchant", "F7_ecosystem_index", "mul"),
        ("weekly_trend_slope", "F1_recency_days", "mul"),
        ("bal_net_change_march", "F9_final_balance", "mul"),
        ("zero_days_last14", "F1_recency_days", "mul")
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

    # Percentile Rank Features
    rank_cols = ["F1_recency_days", "F9_final_balance", "total_tx_count", "F10_inbound_count", "F8_vintage_days", "weekly_trend_slope", "bal_net_change_march"]
    for col in rank_cols:
        if col in train_df.columns:
            train_df[f"rank_{col}"] = train_df[col].rank(pct=True)
            test_df[f"rank_{col}"] = test_df[col].rank(pct=True)

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    # Cast high-cardinality categories to pandas category type
    for col in ["GENDER", "REGION"]:
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")

    logger.info(f"Final Cleaned Feature Matrix: Train {train_df.shape}, Test {test_df.shape}")
    return train_df, test_df


# ===========================================================
# PHASE 3 -- OPTUNA TUNING (CPU-optimized)
# ===========================================================
def add_target_encoding(Xtr, ytr, Xva, X_test, col="REGION", target_name="CHURN"):
    """Leakage-free out-of-fold target encoding for categories"""
    temp = Xtr.copy()
    temp[target_name] = ytr
    means = temp.groupby(col, observed=False)[target_name].mean()
    global_mean = ytr.mean()
    
    Xtr_new = Xtr.copy()
    Xva_new = Xva.copy()
    X_test_new = X_test.copy()
    
    Xtr_new[f"te_{col}"] = Xtr[col].map(means).fillna(global_mean).astype(np.float32)
    Xva_new[f"te_{col}"] = Xva[col].map(means).fillna(global_mean).astype(np.float32)
    X_test_new[f"te_{col}"] = X_test[col].map(means).fillna(global_mean).astype(np.float32)
    
    return Xtr_new, Xva_new, X_test_new


def tune_lgb(X, y):
    logger.info("[Phase 3] Optuna LGB tuning (100k sample, 30 trials on CPU)...")
    n_sample = min(100000, len(X))
    idx = np.random.RandomState(42).choice(len(X), n_sample, replace=False)
    Xs, ys = X.iloc[idx], y.iloc[idx]
    spw = (ys == 0).sum() / ((ys == 1).sum() + 1)

    def objective(trial):
        p = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("lr", 0.01, 0.06, log=True),
            "num_leaves": trial.suggest_int("nl", 31, 127),
            "max_depth": trial.suggest_int("md", 5, 9),
            "min_child_samples": trial.suggest_int("mcs", 20, 200),
            "feature_fraction": trial.suggest_float("ff", 0.5, 0.9),
            "bagging_fraction": trial.suggest_float("bf", 0.6, 0.9),
            "bagging_freq": 1,
            "reg_alpha": trial.suggest_float("ra", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("rl", 1e-3, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("mg", 0.0, 0.3),
            "scale_pos_weight": spw,
            "verbose": -1, "n_jobs": -1, "random_state": 42,
        }

        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for ti, vi in folds.split(Xs, ys):
            Xtr, ytr = Xs.iloc[ti], ys.iloc[ti]
            Xva, yva = Xs.iloc[vi], ys.iloc[vi]
            
            # Apply out-of-fold target encoding inside tuning split
            Xtr_te, Xva_te, _ = add_target_encoding(Xtr, ytr, Xva, Xva, col="REGION")
            
            ds_t = lgb.Dataset(Xtr_te, label=ytr, categorical_feature=['GENDER', 'REGION'])
            ds_v = lgb.Dataset(Xva_te, label=yva, categorical_feature=['GENDER', 'REGION'])
            m = lgb.train(p, ds_t, 800, valid_sets=[ds_v],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
            scores.append(roc_auc_score(yva, m.predict(Xva_te, num_iteration=m.best_iteration)))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    logger.info(f"Best LGB trial: {study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")

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

    spw = (y == 0).sum() / ((y == 1).sum() + 1)

    # Storage for multi-seed predictions
    all_oof = {"lgb": [], "xgb": [], "cat": []}
    all_tp = {"lgb": [], "xgb": [], "cat": []}

    # Detect GPU availability
    has_gpu = False
    try:
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count() > 0:
            has_gpu = True
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

        xgb_params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "learning_rate": 0.025, "max_depth": 7, "scale_pos_weight": spw,
            "subsample": 0.8, "colsample_bytree": 0.75,
            "reg_alpha": 0.1, "reg_lambda": 2.0,
            "tree_method": "hist", "n_jobs": -1, "random_state": seed,
            "enable_categorical": True,  # Native categorical splits for XGB
        }
        if has_gpu:
            xgb_params["device"] = "cuda"

        for fold, (ti, vi) in enumerate(folds.split(X, y)):
            Xtr, ytr = X.iloc[ti], y.iloc[ti]
            Xva, yva = X.iloc[vi], y.iloc[vi]

            # 1. Target encode REGION inside cross-validation splits
            Xtr_te, Xva_te, X_test_te = add_target_encoding(Xtr, ytr, Xva, X_test, col="REGION")

            # 1.1 LightGBM
            ds_t = lgb.Dataset(Xtr_te, label=ytr, categorical_feature=['GENDER', 'REGION'])
            ds_v = lgb.Dataset(Xva_te, label=yva, categorical_feature=['GENDER', 'REGION'])
            m_lgb = lgb.train(lgb_params, ds_t, 3000, valid_sets=[ds_t, ds_v],
                              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            oof_lgb[vi] = m_lgb.predict(Xva_te, num_iteration=m_lgb.best_iteration)
            tp_lgb += m_lgb.predict(X_test_te, num_iteration=m_lgb.best_iteration) / 5

            # 1.2 XGBoost
            m_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=3000, early_stopping_rounds=100)
            m_xgb.fit(Xtr_te, ytr, eval_set=[(Xva_te, yva)], verbose=False)
            oof_xgb[vi] = m_xgb.predict_proba(Xva_te)[:, 1]
            tp_xgb += m_xgb.predict_proba(X_test_te)[:, 1] / 5

            # 1.3 CatBoost
            m_cat = CatBoostClassifier(
                iterations=3000, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
                scale_pos_weight=spw, eval_metric="AUC",
                early_stopping_rounds=100, verbose=0, random_seed=seed,
                task_type="GPU" if has_gpu else "CPU",
                cat_features=["GENDER", "REGION"]
            )
            m_cat.fit(Xtr_te, ytr, eval_set=(Xva_te, yva))
            oof_cat[vi] = m_cat.predict_proba(Xva_te)[:, 1]
            tp_cat += m_cat.predict_proba(X_test_te)[:, 1] / 5

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

    # Average seed OOF predictions
    oof_lgb_avg = np.mean(all_oof["lgb"], axis=0)
    oof_xgb_avg = np.mean(all_oof["xgb"], axis=0)
    oof_cat_avg = np.mean(all_oof["cat"], axis=0)
    tp_lgb_avg = np.mean(all_tp["lgb"], axis=0)
    tp_xgb_avg = np.mean(all_tp["xgb"], axis=0)
    tp_cat_avg = np.mean(all_tp["cat"], axis=0)

    # Rank-transform predictions for rank-averaging
    oof_lgb_ranked = rankdata(oof_lgb_avg) / len(oof_lgb_avg)
    oof_xgb_ranked = rankdata(oof_xgb_avg) / len(oof_xgb_avg)
    oof_cat_ranked = rankdata(oof_cat_avg) / len(oof_cat_avg)
    tp_lgb_ranked = rankdata(tp_lgb_avg) / len(tp_lgb_avg)
    tp_xgb_ranked = rankdata(tp_xgb_avg) / len(tp_xgb_avg)
    tp_cat_ranked = rankdata(tp_cat_avg) / len(tp_cat_avg)

    # --- ADVANCED HYBRID WEIGHT BLENDING OPTIMIZATION ---
    logger.info("Solving optimal probability blend weights...")
    best_w_prob, best_auc_prob = None, 0
    # Fast grid search step=0.05
    for w1 in np.arange(0.0, 1.01, 0.05):
        for w2 in np.arange(0.0, 1.01 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            blend = w1 * oof_lgb_avg + w2 * oof_xgb_avg + w3 * oof_cat_avg
            auc = roc_auc_score(y, blend)
            if auc > best_auc_prob:
                best_auc_prob = auc
                best_w_prob = (w1, w2, w3)

    # Refine probability weights using Nelder-Mead
    def loss_prob(w):
        w = np.clip(w, 0.0, 1.0)
        w_sum = w.sum()
        if w_sum == 0:
            return 0.0
        w = w / w_sum
        blend = w[0] * oof_lgb_avg + w[1] * oof_xgb_avg + w[2] * oof_cat_avg
        return -roc_auc_score(y, blend)

    res_prob = minimize(loss_prob, best_w_prob, method='Nelder-Mead', options={'maxiter': 100})
    best_w_prob = np.clip(res_prob.x, 0.0, 1.0)
    best_w_prob = best_w_prob / best_w_prob.sum()
    best_auc_prob = -loss_prob(best_w_prob)

    logger.info("Solving optimal rank blend weights...")
    best_w_rank, best_auc_rank = None, 0
    # Fast grid search step=0.05
    for w1 in np.arange(0.0, 1.01, 0.05):
        for w2 in np.arange(0.0, 1.01 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            blend = w1 * oof_lgb_ranked + w2 * oof_xgb_ranked + w3 * oof_cat_ranked
            auc = roc_auc_score(y, blend)
            if auc > best_auc_rank:
                best_auc_rank = auc
                best_w_rank = (w1, w2, w3)

    # Refine rank weights using Nelder-Mead
    def loss_rank(w):
        w = np.clip(w, 0.0, 1.0)
        w_sum = w.sum()
        if w_sum == 0:
            return 0.0
        w = w / w_sum
        blend = w[0] * oof_lgb_ranked + w[1] * oof_xgb_ranked + w[2] * oof_cat_ranked
        return -roc_auc_score(y, blend)

    res_rank = minimize(loss_rank, best_w_rank, method='Nelder-Mead', options={'maxiter': 100})
    best_w_rank = np.clip(res_rank.x, 0.0, 1.0)
    best_w_rank = best_w_rank / best_w_rank.sum()
    best_auc_rank = -loss_rank(best_w_rank)

    logger.info(f"Optimal probability blend weights: LGB={best_w_prob[0]:.3f}, XGB={best_w_prob[1]:.3f}, CAT={best_w_prob[2]:.3f} (OOF AUC={best_auc_prob:.6f})")
    logger.info(f"Optimal rank blend weights: LGB={best_w_rank[0]:.3f}, XGB={best_w_rank[1]:.3f}, CAT={best_w_rank[2]:.3f} (OOF AUC={best_auc_rank:.6f})")

    if best_auc_rank > best_auc_prob:
        logger.info("Choosing RANK-AVERAGE blending method (calibrated with Isotonic Regression).")
        final_oof_raw = best_w_rank[0] * oof_lgb_ranked + best_w_rank[1] * oof_xgb_ranked + best_w_rank[2] * oof_cat_ranked
        final_test_raw = best_w_rank[0] * tp_lgb_ranked + best_w_rank[1] * tp_xgb_ranked + best_w_rank[2] * tp_cat_ranked
        
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(final_oof_raw, y)
        final_oof = ir.transform(final_oof_raw)
        final_test = ir.transform(final_test_raw)
    else:
        logger.info("Choosing PROBABILITY blending method.")
        final_oof = best_w_prob[0] * oof_lgb_avg + best_w_prob[1] * oof_xgb_avg + best_w_prob[2] * oof_cat_avg
        final_test = best_w_prob[0] * tp_lgb_avg + best_w_prob[1] * tp_xgb_avg + best_w_prob[2] * tp_cat_avg

    def report(name, yt, yp):
        auc = roc_auc_score(yt, yp)
        pr = average_precision_score(yt, yp)
        thr = np.percentile(yp, 90)
        yb = (yp >= thr).astype(int)
        p10 = precision_score(yt, yb)
        r10 = recall_score(yt, yb)
        logger.info(f"[{name}] AUC={auc:.6f} | PR-AUC={pr:.5f} | P@10%={p10:.5f} | R@10%={r10:.5f}")
        return {"AUC": auc, "PR-AUC": pr, "Precision@10%": p10, "Recall@10%": r10}

    _ = report("LGB-avg", y, oof_lgb_avg)
    _ = report("XGB-avg", y, oof_xgb_avg)
    _ = report("CAT-avg", y, oof_cat_avg)
    final_m = report("FINAL-BLEND", y, final_oof)

    return final_test, final_m, X, y


# ===========================================================
# PHASE 5 -- SHAP & RAPID CSV WRITE
# ===========================================================
def generate_shap(model_lgb, X):
    logger.info("[Phase 5] Generating SHAP explainability...")
    try:
        Xs = X.sample(n=min(2000, len(X)), random_state=42)
        explainer = shap.TreeExplainer(model_lgb)
        sv = explainer(Xs)
        plt.figure(figsize=(12, 7))
        shap.plots.beeswarm(sv, max_display=15, show=False)
        plt.title("SHAP Beeswarm -- V5 Optimized Pipeline", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig("explainability/shap_beeswarm_v5.png", dpi=150)
        plt.close()
        logger.info("SHAP beeswarm plot saved to explainability/shap_beeswarm_v5.png")
    except Exception as e:
        logger.info(f"SHAP generation warning: {e}")


def save_predictions_rapid(test_df, preds):
    sub = pd.DataFrame({
        "ACCOUNT_ID": test_df["ACCOUNT_ID"],
        "CHURN_PROB": preds
    })
    sub.to_csv("predictions_v5.csv", index=False)
    logger.info("predictions_v5.csv saved successfully (255,000 rows).")


# ===========================================================
# MAIN
# ===========================================================
def main():
    logger.info("=== FictiPay V5 Pipeline - High Accuracy & Rapid Execution ===")

    train_df, test_df = extract_features()
    train_df, test_df = preprocess(train_df, test_df)

    exclude = {"ACCOUNT_ID", "ACCOUNT_TYPE", "ACCOUNT_OPEN_DATE", "CHURN"}
    feat_cols = [c for c in train_df.columns if c not in exclude]
    X_tune = train_df[feat_cols].copy()
    y_tune = train_df["CHURN"]

    best_lgb = tune_lgb(X_tune, y_tune)
    preds, final_m, X, y = train_stacking_ensemble(train_df, test_df, best_lgb)

    # LightGBM single model training for SHAP
    lgb_params_final = {**best_lgb, "verbose": -1, "random_state": 42}
    
    # Target encode full REGION dataset for SHAP model
    X_full_te = X.copy()
    global_mean = y.mean()
    means = pd.concat([X, y], axis=1).groupby("REGION", observed=False)["CHURN"].mean()
    X_full_te["te_REGION"] = X["REGION"].map(means).fillna(global_mean).astype(np.float32)

    ds_full = lgb.Dataset(X_full_te, label=y, categorical_feature=['GENDER', 'REGION'])
    m_full = lgb.train(lgb_params_final, ds_full, num_boost_round=800)
    generate_shap(m_full, X_full_te)

    save_predictions_rapid(test_df, preds)
    logger.info(f"=== V5 Complete | Final AUC-ROC: {final_m['AUC']:.6f} ===")


if __name__ == "__main__":
    main()

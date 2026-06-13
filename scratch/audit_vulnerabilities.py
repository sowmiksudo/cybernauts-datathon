import polars as pl
import numpy as np

# Load train labels
train_labels = pl.read_csv("bkash-presents-nsucec-datathon/public/train_labels.csv")

# Ingest transaction dates
print("Ingesting transactions...")
trx = pl.scan_parquet("bkash-presents-nsucec-datathon/public/transactions/trx_2024-0*.parquet")
trx_dates = (
    trx.group_by("SRC_ACCOUNT")
    .agg([
        pl.col("TRX_DATETIME").str.slice(0, 10).str.to_date("%Y-%m-%d").max().alias("max_trx_date"),
        pl.col("TRX_DATETIME").str.slice(0, 10).str.to_date("%Y-%m-%d").min().alias("min_trx_date")
    ])
    .rename({"SRC_ACCOUNT": "ACCOUNT_ID"})
    .collect()
)

# Join with labels
train_merged = train_labels.join(trx_dates, on="ACCOUNT_ID", how="left")

# 1. Audit Timestamp Truncation
print("\n--- Auditing Timestamp Truncation ---")
churned_max_dates = train_merged.filter(pl.col("CHURN") == 1).select("max_trx_date").drop_nulls().to_series()
active_max_dates = train_merged.filter(pl.col("CHURN") == 0).select("max_trx_date").drop_nulls().to_series()

import datetime

if len(churned_max_dates) > 0:
    print(f"Churned users max transaction date range: {churned_max_dates.min()} to {churned_max_dates.max()}")
    # Let's count how many churned users have transactions in late March (e.g. after March 25)
    late_churned = (churned_max_dates >= datetime.date(2024, 3, 25)).sum()
    print(f"Number of churned users active after March 25th: {late_churned} / {len(churned_max_dates)} ({late_churned/len(churned_max_dates)*100:.2f}%)")
else:
    print("No transactions found for churned users!")

if len(active_max_dates) > 0:
    print(f"Active users max transaction date range: {active_max_dates.min()} to {active_max_dates.max()}")
    late_active = (active_max_dates >= datetime.date(2024, 3, 25)).sum()
    print(f"Number of active users active after March 25th: {late_active} / {len(active_max_dates)} ({late_active/len(active_max_dates)*100:.2f}%)")

# 2. Audit ID Sequence Ordering
print("\n--- Auditing ID Sequence Ordering ---")
# Let's look at the correlation of ACCOUNT_ID numeric part with target
# Extract the digits from ACCOUNT_ID
print("Sample ACCOUNT_IDs:")
print(train_labels.select("ACCOUNT_ID").head(10).to_series().to_list())

id_digits = (
    train_labels.with_columns([
        pl.col("ACCOUNT_ID").str.extract(r"(\d+)").cast(pl.Int64).alias("id_num")
    ])
)
# Calculate correlation between id_num and CHURN
corr_id = id_digits.select(pl.corr("id_num", "CHURN")).to_series()[0]
print(f"Correlation between ACCOUNT_ID number and CHURN: {corr_id:.6f}")

# Let's check if there are blocks of churned users by deciles
deciles = np.percentile(id_digits.select("id_num").to_series().to_numpy(), np.linspace(0, 100, 11))
for i in range(10):
    subset = id_digits.filter((pl.col("id_num") >= deciles[i]) & (pl.col("id_num") < deciles[i+1]))
    if len(subset) > 0:
        rate = subset.select(pl.col("CHURN").mean()).to_series()[0]
        print(f"Decile {i+1} (id_num {deciles[i]:.0f} to {deciles[i+1]:.0f}): Churn rate = {rate*100:.4f}%")

# 3. Audit TrxID Chronological Alignment
print("\n--- Auditing TrxID Chronological Alignment ---")
# Let's read a sample of transactions and check if TrxID is sorted chronologically
sample_trx = (
    trx.select(["TrxID", "TRX_DATETIME"])
    .head(10000)
    .collect()
)
# Extract numeric part of TrxID (assuming format like TX0001 or similar)
print("Sample TrxIDs:")
print(sample_trx.select("TrxID").head(10).to_series().to_list())

sample_trx = (
    sample_trx.with_columns([
        pl.col("TrxID").str.extract(r"(\d+)").cast(pl.Int64).alias("trx_num"),
        pl.col("TRX_DATETIME").str.slice(0, 19).str.to_datetime("%Y-%m-%d %H:%M:%S").alias("trx_dt")
    ])
)

# Correlation between trx_num and timestamp (converted to float epoch)
sample_trx_clean = sample_trx.drop_nulls()
if len(sample_trx_clean) > 0:
    timestamps = sample_trx_clean.select(pl.col("trx_dt").cast(pl.Int64)).to_series()
    trx_nums = sample_trx_clean.select("trx_num").to_series()
    corr_trx = pl.DataFrame({"ts": timestamps, "num": trx_nums}).select(pl.corr("ts", "num")).to_series()[0]
    print(f"Correlation between TrxID numeric part and Timestamp: {corr_trx:.6f}")
else:
    print("Could not parse timestamps in transaction sample.")


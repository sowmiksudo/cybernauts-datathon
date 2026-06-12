FICTIPAY DATATHON MASTER PLAYBOOKEnd-to-End Customer Churn Prediction Blueprint1. Quick Reference DashboardDimensionSpecificationNotes / DetailsProblem TypeBinary ClassificationPredict Churn vs. Non-Churn for Customer AccountsEvaluation MetricAUC-ROCArea under the Receiver Operating Characteristic CurveTarget VariableCHURN1 = Churn, 0 = No ChurnLabel Base Distribution12.7% Churn (75,435) / 87.3% Active (519,565)Significant class imbalance; total training labels = 595,000Data Scope (Train)Jan 1, 2024 $\rightarrow$ Mar 31, 2024Observation WindowData Scope (Labels)Apr 1, 2024 $\rightarrow$ Apr 30, 2024Churn Window (Active = $\ge 1$ outbound transaction in this window)Dataset Scale~200M Transactions, ~360M Daily BalancesToo large for standard pandas; Out-of-Core Processing is required2. Table Schemas & Data RelationshipsThe database is built on three main physical tables linked by transaction flows and historical user IDs. Focus exclusively on Customer accounts for predictions, but analyze their interactions with Merchant and Biller accounts.       [ KYC Table ] 
     (2 Million Accounts)
              │
              ├───► Customer Accounts (Target Set: 595k Train, 255k Test)
              ├───► Merchant Accounts (Destination nodes)
              └───► Biller Accounts (Destination nodes)
              │
              ▼
   [ Transaction Table ] ◄──────────────► [ DayEndBalance Table ]
  (200M+ Rows, Parquet Split)             (360M Rows, Parquet Split)
  - SRC_ACCOUNT (Sender)                  - ACCOUNT_ID
  - DST_ACCOUNT (Receiver)                - DATE
  - TRX_TYPE (P2P, Bill, etc.)            - AVAILABLE_BALANCE
Table 1: KYC Metadata (kyc.parquet ~2M Rows)ACCOUNT_ID (Primary Key): Unique string identifier for all entity types.ACCOUNT_TYPE (Categorical): Customer, Merchant, or Biller.ACCOUNT_OPEN_DATE (Date): Timestamp of account creation.GENDER (Categorical): Account owner gender.REGION (Categorical): Geographic region code.Table 2: Transactions (transactions/trx_2024-0*.parquet ~200M Rows)TrxID (Primary Key): Unique transaction ID.TRX_DATETIME (Timestamp): Exact execution time of the transaction.SRC_ACCOUNT (Foreign Key): Initiating account.DST_ACCOUNT (Foreign Key): Receiving account (Merchant, Biller, or other Customer).TRX_TYPE (Categorical): P2P, MerchantPay, BillPay, CashIn, CashOut.TRX_AMT (Continuous): Value transacted in local currency.Table 3: Daily Balances (dayend_balance/balance_2024-0*.parquet ~360M Rows)ACCOUNT_ID (Foreign Key / Composite Key): Owner identifier.DATE (Composite Key): Calendar tracking date.AVAILABLE_BALANCE (Continuous): Wallet balance recorded at the end of the day.3. Challenge Roadmap: The 8 Evaluative Stages (Total: 100 Points)To maximize human and auto-graded evaluation scores, follow this structured roadmap:[Phase 1] Scale & Ingestion (10 pts) ────► [Phase 2 & 3] Feature Matrix (25 pts) ────► [Phase 4] Sampling Logic (10 pts)
                                                                                               │
                                                                                               ▼
[Phase 8] Business Strategy (10 pts) ◄─── [Phase 7] Explainability (15 pts) ◄─── [Phase 5 & 6] Modeling (30 pts)
Stage 1: Large-Scale Data Handling (10 Points) — Human EvaluatedObjective: Ingest and process files that exceed RAM limits (e.g., Google Colab, Kaggle, or standard local hardware limits).Your Stack Choice: Use Polars LazyFrames or Apache PySpark.Polars Justification: It utilizes Rust-backed parallel query execution, lazy scanning (scan_parquet), memory mapping, and streaming execution algorithms (collect(streaming=True)) to process massive structures without JVM overhead.Strategy: Implement parquet partitioning based on the date columns and only materialize the computed feature aggregates instead of reading raw transactional rows directly into memory.Stage 2: Feature Engineering (15 Points) — Human EvaluatedObjective: Develop continuous, categorical, and sequence-based signals capturing wallet usage velocity and momentum.Deliverable: A table containing at least 10 highly creative features accompanied by robust domain-specific business rationales (see the Curated Feature Library in Section 5).Stage 3: Feature Quality & Pareto Analysis (10 Points) — Human EvaluatedObjective: Diagnose and normalize high-skewness and zero-inflated distributions typical of financial transaction data.Strategy:Sparsity: Identify dormant users with zero transactions in specific windows and construct explicit binary indicators (e.g., is_dormant_last_30d).Skewness: Plot transaction volumes and apply log-transforms: $x' = \log(x + 1)$ or power transformations (e.g., Box-Cox/Yeo-Johnson) to stabilize variance.Clipping: Apply winsorization or percentiles (e.g., 99th percentile capping) to limit the impact of extreme outliers.Stage 4: Class Imbalance & Sampling (10 Points) — Human EvaluatedObjective: Address the class imbalance (12.7% Churn rate / $\approx 1:7$ ratio) to ensure robust decision boundaries.Strategy:Avoid synthetic oversampling (like SMOTE) on very large tabular datasets, as it can introduce artificial patterns and significantly slow down processing pipelines.Recommended Method: Use algorithmic cost-sensitive learning (e.g., scale_pos_weight in XGBoost/LightGBM) set to:$$\text{scale\_pos\_weight} = \frac{N_{\text{majority}}}{N_{\text{minority}}} = \frac{519,565}{75,435} \approx 6.88$$This scales the loss gradient on minority class classification errors during boosting iterations without changing the underlying raw distribution.Stage 5: Model Selection & Training (20 Points) — Auto Evaluated ★Objective: Implement predictive models optimized for tabular data.Evaluation Metrics: Cross-validated AUC-ROC, Average Precision (PR-AUC), and top-tier metrics: Precision@10% and Recall@10%.Algorithm Strategy:LightGBM / XGBoost: Primary learners. They naturally handle sparse matrices, categorical splits, and can be trained out-of-core with histogram-based binning.CatBoost: Highly effective if categorical features (e.g., REGION, GENDER) are left unencoded to leverage CatBoost's target-statistics execution.Stage 6: Hyperparameter Tuning (10 Points) — Human EvaluatedObjective: Systematically discover optimal parameters without wasting compute.Method: Run Optuna to execute Bayesian Optimization using a Tree-structured Parzen Estimator (TPE).Target Search Space: Focus on learning_rate ($[0.01, 0.1]$), num_leaves ($[31, 127]$), min_child_samples ($[20, 200]$), feature_fraction ($[0.6, 0.9]$), and reg_alpha / reg_lambda regularization parameters.Stage 7: Model Explainability (15 Points) — Human EvaluatedObjective: Provide transparent, human-interpretable explanations of model predictions.Deliverable: Save SHAP (SHapley Additive exPlanations) beeswarm and dependence plots under the /explainability folder.Leakage Safeguard: Ensure no target labels or post-observation window features (April 2024 data) leak into the feature generation matrices.Stage 8: Business Recommendations & Communication (10 Points) — Human EvaluatedObjective: Translate model scores into actionable churn mitigation strategies.Strategy: Detail a risk-based targeting approach:Decision Rule: Segment the top 5% or 10% highest-risk customers (highest predicted CHURN_PROB).Calculation: Estimate the expected retention rate improvement and calculate the financial lift (e.g., savings from prevented customer acquisition costs vs. the cost of target incentives like cash-back offers).4. Required Submission Files ChecklistArtifact PathFormatStatusDescriptionpredictions.csvCSVREQUIREDExactly 255,000 predictions for the test set. See exact format below.features.mdMarkdownREQUIREDDocumentation of $\ge 10$ engineering features with rationales (PDF accepted).notebook.ipynbJupyter NotebookREQUIREDFully commented, self-contained, and reproducible execution script.report.pdfPDF DocumentREQUIREDStructured performance report with metric tables and Optuna tracking charts.explainability/DirectoryREQUIREDSaved SHAP summary/beeswarm plot files and interpretations.presentation.pdfPDF / PPTXREQUIREDA concise 5-slide deck or functional dashboard mockup for leadership.predictions.csv Formatting SchemaThe output predictions file must strictly match this structure to avoid parsing failures:Header: ACCOUNT_ID,CHURN_PROBData Rows: Exactly 255,000 unique rows corresponding to test.csv.Output Value Range: Float values representing predicted probabilities bounded between $[0.0, 1.0]$.ACCOUNT_ID,CHURN_PROB
CUST000000000001,0.823
CUST000000000002,0.041
CUST000000000003,0.912
CUST000000000004,0.156
5. Curated Feature Library (The 10 Required Features)This library contains 10 highly predictive features designed for FictiPay's transaction and balance tables, including their mathematical formulations and business rationales.Feature 1: Days Since Last Outbound Activity (Recency)Mathematical Representation:$$F_1 = T_{\text{end}} - \max \left(\{ t \in \text{TRX\_DATETIME} \mid \text{SRC\_ACCOUNT} = i \}\right)$$Where $T_{\text{end}} = \text{2024-03-31}$.Business Rationale: The length of time since a user last initiated an outbound transaction is often the single most predictive indicator of customer churn. Longer spans of inactivity correlate with a higher probability of abandonment.Feature 2: March Activity Momentum Ratio (Velocity)Mathematical Representation:$$F_2 = \frac{C_{i, \text{March}} + 1}{\frac{1}{2} (C_{i, \text{Jan}} + C_{i, \text{Feb}}) + 1}$$Where $C_{i, M}$ is the total transaction count for user $i$ in month $M$.Business Rationale: This ratio captures changes in transaction frequency as users approach the prediction window. A value less than $1.0$ indicates a decline in wallet usage, which is a strong signal of potential churn.Feature 3: Outbound Volume Concentration Trend (Monetary)Mathematical Representation:$$F_3 = \frac{\sum \text{TRX\_AMT}_{i, \text{March}}}{\sum \text{TRX\_AMT}_{i, \text{Jan-Feb}} + 1}$$Business Rationale: Beyond transaction frequency, a drop in the total amount transacted month-over-month indicates a decline in wallet usage and utility. It helps identify high-value transactors who are scaling back their wallet activity.Feature 4: Balance Depletion Rate (Trend)Mathematical Representation:$$F_4 = \overline{B}_{i, \text{March}} - \overline{B}_{i, \text{January}}$$Where $\overline{B}_{i, M}$ is the mean daily balance for user $i$ in month $M$.Business Rationale: A declining daily balance trend over the 90-day observation window often indicates that a user is systematically draining their account before abandoning the platform.Feature 5: Balance Volatility Index (Dispersion)Mathematical Representation:$$F_5 = \sigma(B_i) = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(B_{i,t} - \overline{B}_i)^2}$$Where $B_{i,t}$ represents the daily balance of user $i$ on day $t$.Business Rationale: High balance volatility indicates that a wallet is actively used for cash flow and transactions. Conversely, a standard deviation close to zero suggests a dormant, static, or inactive account.Feature 6: Platform Pass-Through Index (Behavioral)Mathematical Representation:$$F_6 = \frac{\sum \text{TRX\_AMT}_{i, \text{CashOut}}}{\sum \text{TRX\_AMT}_{i, \text{CashIn}} + 1}$$Business Rationale: A ratio close to $1.0$ indicates that the user primarily uses the wallet to cash out incoming funds quickly, rather than keeping money in the FictiPay ecosystem. These "pass-through" users are more susceptible to churn.Feature 7: Digital Ecosystem Interdependence (Utility)Mathematical Representation:$$F_7 = \frac{\text{Count}(\text{MerchantPay}) + \text{Count}(\text{BillPay})}{\text{Total Outbound Count} + 1}$$Business Rationale: Users who integrate digital wallet features like bill payments and merchant checkout into their daily routines tend to be stickier. Those who only use the wallet for basic peer-to-peer (P2P) transfers are generally easier to lose.Feature 8: Vintage Account Factor (Profile Profile)Mathematical Representation:$$F_8 = T_{\text{end}} - \text{ACCOUNT\_OPEN\_DATE}_i$$Business Rationale: Newly onboarded users typically exhibit higher churn rates than long-standing vintage accounts who have developed long-term usage habits on the platform.Feature 9: Absolute Balance Level (Liquidity)Mathematical Representation:$$F_9 = B_{i, \text{2024-03-31}}$$The final available balance recorded on the last day of the observation window.Business Rationale: Users with substantial funds remaining in their wallets on March 31st are statistically less likely to become completely inactive (churn) in April compared to users with zero or near-zero balances.Feature 10: Inbound Support Frequency (Network Engagement)Mathematical Representation:$$F_{10} = \text{Count}(t \in \text{Transactions} \mid \text{DST\_ACCOUNT} = i)$$Where the user acts as the destination recipient for inbound transfers.Business Rationale: A customer's engagement is also driven by their receiving activity. Regular inbound transfers from friends, family, or employers act as an external anchor that keeps the user active on the platform.6. High-Performance Out-of-Core Processing BlueprintBelow is an optimized code blueprint using Polars LazyFrames and LightGBM. This script is designed to process the entire ~200M transaction dataset within Kaggle's standard RAM limits."""
FictiPay Datathon End-to-End Execution Pipeline
Author: Shayer Mahmud Sowmik
Optimized for: Polars Streaming & LightGBM Binary Classification
"""

import os
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ----------------- Configuration & Directives -----------------
OBSERVATION_END = pl.date(2024, 3, 31)
app_id = "fictipay-churn-datathon"

def build_computational_graph():
    print("[1] Building KYC Metadata Tree...")
    # Lazy scan of the KYC parquet file
    kyc_lazy = pl.scan_parquet("kyc.parquet").filter(pl.col("ACCOUNT_TYPE") == "Customer")
    
    # Calculate basic demographic and profile features
    kyc_features = kyc_lazy.with_columns([
        ((OBSERVATION_END - pl.col("ACCOUNT_OPEN_DATE").dt.date()).dt.total_days()).alias("F8_vintage_days"),
        pl.col("GENDER").to_physical().cast(pl.Int8),
        pl.col("REGION").to_physical().cast(pl.Int16)
    ])

    print("[2] Processing Large-Scale Outbound Transactions (200M+ Rows)...")
    trx_lazy = pl.scan_parquet("transactions/trx_2024-0*.parquet")
    
    # Generate transactional aggregates by grouping outbound customer accounts
    src_features = (
        trx_lazy.group_by("SRC_ACCOUNT")
        .agg([
            # Feature 1: Recency
            ((OBSERVATION_END - pl.col("TRX_DATETIME").dt.date().max()).dt.total_days()).alias("F1_recency_days"),
            
            # Feature 2 & 3: Monthly Frequencies & Amounts for March vs Jan-Feb
            pl.col("TrxID").count().alias("total_tx_count"),
            pl.col("TrxID").filter(pl.col("TRX_DATETIME").dt.month() == 3).count().alias("tx_count_march"),
            pl.col("TrxID").filter(pl.col("TRX_DATETIME").dt.month() < 3).count().alias("tx_count_early"),
            pl.col("TRX_AMT").filter(pl.col("TRX_DATETIME").dt.month() == 3).sum().alias("tx_amt_march"),
            pl.col("TRX_AMT").filter(pl.col("TRX_DATETIME").dt.month() < 3).sum().alias("tx_amt_early"),
            
            # Feature 6 & 7: Categorical Transaction Splitting
            pl.col("TRX_AMT").filter(pl.col("TRX_TYPE") == "CashOut").sum().alias("amt_cashout"),
            pl.col("TRX_AMT").filter(pl.col("TRX_TYPE") == "CashIn").sum().alias("amt_cashin"),
            pl.col("TrxID").filter(pl.col("TRX_TYPE") == "MerchantPay").count().alias("count_merchant"),
            pl.col("TrxID").filter(pl.col("TRX_TYPE") == "BillPay").count().alias("count_bill")
        ])
        .rename({"SRC_ACCOUNT": "ACCOUNT_ID"})
    )

    print("[3] Scanning Day-End Balances (360M Rows)...")
    balance_lazy = pl.scan_parquet("dayend_balance/balance_2024-0*.parquet")
    
    # Calculate balances, averages, volatility, and trend slopes
    balance_features = (
        balance_lazy.group_by("ACCOUNT_ID")
        .agg([
            # Feature 9: Absolute Final Balance
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE") == OBSERVATION_END).first().alias("F9_final_balance"),
            
            # Feature 5: Balance Volatility
            pl.col("AVAILABLE_BALANCE").std().alias("F5_balance_volatility"),
            
            # Feature 4: Balance Trends
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 3).mean().alias("avg_bal_march"),
            pl.col("AVAILABLE_BALANCE").filter(pl.col("DATE").dt.month() == 1).mean().alias("avg_bal_jan")
        ])
    )

    print("[4] Executing Lazy Joins and Derived Feature Calculations...")
    # Combine feature tables
    joined_features = (
        kyc_features
        .join(src_features, on="ACCOUNT_ID", how="left")
        .join(balance_features, on="ACCOUNT_ID", how="left")
    )

    # Calculate ratios and momentum features
    final_features = joined_features.with_columns([
        # Feature 2: March Activity Momentum Ratio
        (pl.col("tx_count_march") / ((pl.col("tx_count_early") / 2.0) + 1.0)).alias("F2_momentum_ratio"),
        
        # Feature 3: Outbound Volume Concentration Trend
        (pl.col("tx_amt_march") / (pl.col("tx_amt_early") + 1.0)).alias("F3_monetary_trend"),
        
        # Feature 4: Balance Depletion Rate
        (pl.col("avg_bal_march") - pl.col("avg_bal_jan")).alias("F4_balance_depletion"),
        
        # Feature 6: Platform Pass-Through Index
        (pl.col("amt_cashout") / (pl.col("amt_cashin") + 1.0)).alias("F6_pass_through_index"),
        
        # Feature 7: Digital Ecosystem Interdependence
        ((pl.col("count_merchant") + pl.col("count_bill")) / (pl.col("total_tx_count") + 1.0)).alias("F7_ecosystem_index")
    ]).fill_null(0)

    # Lazy join with train labels and test keys
    train_labels = pl.scan_csv("train_labels.csv")
    test_keys = pl.scan_csv("test.csv")

    # Use streaming collect to process the pipeline in chunks
    print("[5] Collecting training and test datasets via Polars Streaming...")
    train_collected = train_labels.join(final_features, on="ACCOUNT_ID", how="inner").collect(streaming=True)
    test_collected = test_keys.join(final_features, on="ACCOUNT_ID", how="left").collect(streaming=True)

    return train_collected.to_pandas(), test_collected.to_pandas()

# Run compilation
train_df, test_df = build_computational_graph()

# ----------------- Model Training Phase -----------------
print("[6] Initializing Stratified K-Fold Training Sequence...")

target_column = 'CHURN'
exclude_cols = ['ACCOUNT_ID', 'ACCOUNT_TYPE', 'ACCOUNT_OPEN_DATE', target_column]
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

X = train_df[feature_cols]
y = train_df[target_column]
X_test = test_df[feature_cols]

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

# Compute class imbalance scaling factor (519565 / 75435)
imbalance_scale = 519565 / 75435

# Optimized hyperparameters for training
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 8,
    'scale_pos_weight': imbalance_scale,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"--- Training Fold {fold + 1} / 5 ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    train_dataset = lgb.Dataset(X_train, label=y_train)
    val_dataset = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=2000,
        valid_sets=[train_dataset, val_dataset],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )
    
    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

print(f"\n[!] Out-Of-Fold AUC-ROC Score: {roc_auc_score(y, oof_preds):.5f}")

# ----------------- Submission Generation -----------------
print("[7] Structuring Predictions & Exporting submission.csv...")
submission = test_df[['ACCOUNT_ID']].copy()
submission['CHURN_PROB'] = test_preds

# Validate range and output structure
assert len(submission) == 255000, f"Error: Row count of {len(submission)} does not match the required 255,000"
submission.to_csv("submission.csv", index=False)
print("[+] Build Complete. submission.csv is ready for scoring!")

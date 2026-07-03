# FictiPay Customer Churn Prediction — Cybernauts Datathon

This repository contains the complete, end-to-end data pipeline and machine learning models for the **bKash presents NSUCEC Datathon**. The goal of the competition is to predict customer churn for FictiPay (defined as having zero outbound transactions in April 2024) evaluated using the AUC-ROC metric.

## 🚀 Project Overview

The project processes large-scale transaction and daily balance data (~1.6 GB on disk, >500M combined rows) using out-of-core streaming in Polars, engineers 10 high-impact behavior and balance features, performs Stratified 5-Fold Cross-Validation, optimizes hyperparameters via Bayesian TPE (Optuna), and interprets model predictions with SHAP.

---

## 🛠️ Repository Deliverables

The following files have been prepared and compiled in the repository root:
1. **Predictions**: [predictions.csv](./predictions.csv) — Final test set predictions (`ACCOUNT_ID,CHURN_PROB`) for 255,000 customers.
2. **Jupyter Notebook**: [notebook.ipynb](./notebook.ipynb) — Explanatory workflow, modeling experiments, and data visualizations.
3. **Core Pipeline Script**: [pipeline_v5.py](./pipeline_v5.py) — Clean, production-grade Python script executing the entire end-to-end pipeline.
4. **Feature Documentation**: [features.md](./features.md) — Mathematical formulas and business justifications for all 10 engineered features.
5. **Business Presentation**: [presentation.pptx](./presentation.pptx) — A 5-slide widescreen PowerPoint presentation for leadership.
6. **Technical Report**: [report.pdf](./report.pdf) — A detailed PDF compiling model performance, hyperparameter logs, and SHAP explainability charts.

---

## 📐 Engineered Features

The model utilizes 10 key features engineered from transactions, balance history, and KYC records:

| Feature | Category | Source | Description |
|---|---|---|---|
| **F1** | Recency | Transactions | Days since last outbound activity |
| **F2** | Velocity | Transactions | March activity momentum ratio |
| **F3** | Monetary | Transactions | Outbound volume concentration trend |
| **F4** | Trend | DayEndBalance | Balance depletion rate |
| **F5** | Dispersion | DayEndBalance | Balance volatility index |
| **F6** | Behavioral | Transactions | Platform pass-through index |
| **F7** | Utility | Transactions | Digital ecosystem interdependence |
| **F8** | Profile | KYC | Vintage account factor (days active) |
| **F9** | Liquidity | DayEndBalance | Absolute balance level (as of March 31) |
| **F10** | Network | Transactions | Inbound support frequency |

---

## 📈 Model Performance & Validation

Validation metrics evaluated under **Stratified 5-Fold Cross-Validation**:

| Model Name | AUC-ROC | PR-AUC | Precision @ Top 10% | Recall @ Top 10% |
|---|---|---|---|---|
| LightGBM (Tuned) | 0.98349 | 0.91139 | 90.916% | 71.711% |
| XGBoost (Baseline) | 0.98324 | 0.91170 | 90.903% | 71.700% |
| **Ensemble Blend (50/50)** | **0.98348** | **0.91193** | **90.939%** | **71.729%** |

*Note: The final submissions use the blended LightGBM and XGBoost model outputs, capturing 71.7% of churners in the top 10% risk segment with a high 90.9% precision.*

---

## ⚙️ How to Run

### 1. Prerequisites
Ensure python 3.10+ and `uv` package manager (recommended) are installed. 

### 2. Setup Environment
Initialize virtual environment and install dependencies:
```bash
uv venv
uv pip install -r requirements.txt
```

### 3. Run Pipeline
To run the complete data loading, preprocessing, model training, explainability analysis, and predictions generation:
```bash
python pipeline_v5.py
```

import json
import os

def convert_py_to_ipynb(py_path, ipynb_path):
    with open(py_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Split the file by phase headers
    lines = code.split("\n")
    
    cells = []
    
    # Title Markdown Cell
    title_md = (
        "# FictiPay Customer Churn Prediction Pipeline - V5 Final Submission\n"
        "**Author**: Shayer Mahmud Sowmik\n\n"
        "This notebook implements the complete end-to-end churn prediction pipeline using Polars LazyFrames, LightGBM, XGBoost, and CatBoost.\n\n"
        "### Key Improvements in V5:\n"
        "1. **Weekly Time-Series Decomposition**:\n"
        "   - 13 weekly counts and amount aggregates (weeks 0 to 12).\n"
        "   - Vectorized consecutive zero week streak detection.\n"
        "   - Vectorized weekly counts and amounts linear regression slopes.\n"
        "2. **DST_ACCOUNT Graph Features**:\n"
        "   - Proportions of transactions sent to Merchants, Billers, and Customers.\n"
        "   - Count of unique Merchant and Biller destinations.\n"
        "3. **Balance Trajectory Shapes**:\n"
        "   - Net balance change, daily increase/decrease days, maximum daily balance drop in March.\n"
        "   - Days with zero balance in the last 14 days of March.\n"
        "4. **Time-of-Day and Day-of-Week Behavioral Patterns**:\n"
        "   - Business hour, evening, night, and weekend transaction ratios.\n"
        "5. **Distribution Shapes and Inbound Details**:\n"
        "   - Skewness, kurtosis, percentiles (IQR) of TRX_AMT.\n"
        "   - Social network size (unique P2P inbound senders), standard deviation of CashIn deposits.\n"
        "6. **Advanced Stacking & Blending**:\n"
        "   - Out-of-fold target encoding for REGION to prevent leakage.\n"
        "   - Evaluation of both probability and rank-average blending.\n"
        "   - Nelder-Mead refined optimization starting from grid search points.\n"
        "   - Isotonic regression calibration of rank-averaged predictions."
    )
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in title_md.split("\n")]
    })
    
    # Imports & logging setup block
    imports_lines = lines[0:59]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Setup & Environment Ingestion\n", "Importing core libraries and initializing logger configurations."]
    })
    # Remove matplotlib.use("Agg") if present, so plots can display inline in Jupyter
    clean_imports = []
    for l in imports_lines:
        if 'matplotlib.use("Agg")' not in l:
            clean_imports.append(l)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in clean_imports]
    })
    
    # Phase 1: Feature Extraction
    p1_lines = lines[59:391]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Phase 1 -- Feature Extraction (Polars & Streaming)\n", "Aggregating transaction counts, amounts, graph topology, inbound details, and balance trajectory features out-of-core using Polars LazyFrames."]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in p1_lines]
    })
    
    # Phase 2: Preprocessing
    p2_lines = lines[391:504]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Phase 2 -- Preprocessing & Advanced Data Cleaning\n", "Performing Winsorization of outliers, Log-Transforms of skewed features, Symlog scaling, and Target Encoding of the REGION feature."]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in p2_lines]
    })
    
    # Phase 3: Optuna Tuning
    p3_lines = lines[504:583]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Phase 3 -- Hyperparameter Tuning\n", "Bayesian Optimization using Optuna (TPE Sampler) on a 100k subset of the train set to find optimal LightGBM parameters."]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in p3_lines]
    })
    
    # Phase 4: Stacking Ensemble
    p4_lines = lines[583:791]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Phase 4 -- Multi-Seed Stacking & Blending Ensemble\n", "Cross-validating LightGBM, XGBoost, and CatBoost models across 3 random seeds using 5-fold Stratified K-Fold. Blending out-of-fold predictions using Rank-Average optimization and calibration via Isotonic Regression."]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in p4_lines]
    })
    
    # Phase 5: SHAP & Predictions Export
    p5_lines = lines[791:820]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Phase 5 -- Explainability & Predictions Export\n", "Computing SHAP values for the full dataset to explain global feature importance, generating beeswarm plots, and exporting the predictions."]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in p5_lines]
    })
    
    # Execution Block
    exec_lines = lines[820:]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Pipeline Execution Run\n", "Invoking the full end-to-end process sequence."]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in exec_lines]
    })
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Successfully converted {py_path} to Jupyter Notebook {ipynb_path}")

if __name__ == "__main__":
    convert_py_to_ipynb("pipeline_v5.py", "notebook.ipynb")

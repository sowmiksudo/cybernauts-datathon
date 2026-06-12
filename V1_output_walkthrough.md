# FictiPay Customer Churn Prediction — Walkthrough

This walkthrough summarizes the completedNSUCEC FictiPay Customer Churn Prediction Datathon implementation.

## 1. Summary of Changes

We implemented the complete datathon workload plan from [IMPLEMENT.md](file:///c:/Users/Sowmik/Documents/GitHub/Personal%20AI%20assistant%20folder/cybernauts-datathon/IMPLEMENT.md):
- **Dependencies Setup**: Initialized `.venv` using `uv` and successfully installed all data science and modeling packages specified in [requirements.txt](file:///c:/Users/Sowmik/Documents/GitHub/Personal%20AI%20assistant%20folder/cybernauts-datathon/requirements.txt).
- **Core Pipeline Script**: Wrote [pipeline.py](file:///c:/Users/Sowmik/Documents/GitHub/Personal%20AI%20assistant%20folder/cybernauts-datathon/pipeline.py) which coordinates Polars ingestion, outlier preprocessing, Stratified 5-Fold Cross-Validation modeling, hyperparameter optimization, SHAP explainability analysis, and document compilation.
- **Out-of-Core Processing**: Formulated and executed Polars LazyFrame scanning of ~200M transactions and ~360M daily balances using date partition filtering and streaming collection.
- **Complete Feature Engineering**: Developed the 10 core features ($F_1$ to $F_{10}$) with the exact formulas requested. This includes $F_{10}$ (Inbound Support Frequency) which was missing from the master playbook's code blueprint.
- **Preprocessing (Stage 3)**: Applied 99th percentile Winsorization to clip outliers and $log(x + 1)$ transforms to stabilize right-skewed transaction volumes and balance distributions. Created an explicit dormancy indicator mapping 30-day inactivity.
- **Class Imbalance & Modeling (Stage 4 & 5)**: Configured LightGBM and XGBoost classifiers with cost-sensitive learning (`scale_pos_weight = 6.88`) to accommodate the 12.7% churn class imbalance.
- **Optuna Tuning (Stage 6)**: Optimized LightGBM hyperparameters over 20 Bayesian TPE optimization trials.
- **Model Explainability (Stage 7)**: Computed tree SHAP values on validation records, generating beeswarm feature importance and recency/velocity dependence plots.
- **Compiled Deliverables (Stage 8)**: Exported the final blended predictions, slide deck, Jupyter notebook, feature markdown, and PDF performance report.

---

## 2. Validation & Model Performance

Validation metrics under 5-Fold Stratified Cross-Validation:

| Model Name | AUC-ROC | PR-AUC | Precision @ Top 10% | Recall @ Top 10% |
|---|---|---|---|---|
| **LightGBM (Tuned)** | 0.98349 | 0.91139 | 90.916% | 71.711% |
| **XGBoost (Baseline)** | 0.98324 | 0.91170 | 90.903% | 71.700% |
| **Ensemble Blend (50/50)** | **0.98348** | **0.91193** | **90.939%** | **71.729%** |

> [!TIP]
> The blended ensemble achieves the highest overall Average Precision (PR-AUC = **0.91193**) and captures **71.729%** of all true churners in the top 10% highest-risk segment with an outstanding precision of **90.939%**. This provides FictiPay with a highly reliable target segment for campaign deployment.

---

## 3. Visualizations & Quality Plots

````carousel
![Feature Quality: Before vs After Transformations](file:///C:/Users/Sowmik/.gemini/antigravity-ide/brain/53ccdad5-2c9c-42e4-9401-e953c03e578a/explainability/balance_quality_transforms.png)
<!-- slide -->
![Optuna Hyperparameter Optimization History](file:///C:/Users/Sowmik/.gemini/antigravity-ide/brain/53ccdad5-2c9c-42e4-9401-e953c03e578a/explainability/optuna_history.png)
<!-- slide -->
![SHAP Beeswarm Plot: Feature Importances](file:///C:/Users/Sowmik/.gemini/antigravity-ide/brain/53ccdad5-2c9c-42e4-9401-e953c03e578a/explainability/shap_beeswarm.png)
<!-- slide -->
![SHAP Dependence Plot: Days Since Last Outbound Activity (F1)](file:///C:/Users/Sowmik/.gemini/antigravity-ide/brain/53ccdad5-2c9c-42e4-9401-e953c03e578a/explainability/shap_dependence_F1.png)
<!-- slide -->
![SHAP Dependence Plot: March Activity Momentum Ratio (F2)](file:///C:/Users/Sowmik/.gemini/antigravity-ide/brain/53ccdad5-2c9c-42e4-9401-e953c03e578a/explainability/shap_dependence_F2.png)
````

---

## 4. Submission Deliverables Checklist

The following deliverables have been successfully compiled in the repository root:

1. [predictions.csv](file:///c:/Users/Sowmik/Documents/GitHub/Personal%20AI%20assistant%20folder/cybernauts-datathon/predictions.csv): CSV file containing exactly 255,000 predictions mapping `ACCOUNT_ID,CHURN_PROB`.
2. [features.md](file:///c:/Users/Sowmik/Documents/GitHub/Personal%20AI%20assistant%20folder/cybernauts-datathon/features.md): Documentation of the 10 engineered features with formulas and business rationales.
3. [notebook.ipynb](file:///c:/Users/Sowmik/Documents/GitHub/Personal%20AI%20assistant%20folder/cybernauts-datathon/notebook.ipynb): Jupyter Notebook representing the fully documented, reproducible pipeline.
4. [report.pdf](file:///c:/Users/Sowmik/Documents/GitHub/Personal%20AI%20assistant%20folder/cybernauts-datathon/report.pdf): Structured multi-page PDF performance report with tables and tracking charts.
5. [presentation.pptx](file:///c:/Users/Sowmik/Documents/GitHub/Personal%20AI%20assistant%20folder/cybernauts-datathon/presentation.pptx): Professional 5-slide widescreen business PowerPoint presentation for leadership.
6. [explainability/shap_interpretations.md](file:///c:/Users/Sowmik/Documents/GitHub/Personal%20AI%20assistant%20folder/cybernauts-datathon/explainability/shap_interpretations.md): Narrative explainability document analyzing SHAP results.

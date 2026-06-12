# Model Explainability & SHAP Interpretations

## 1. Key Feature Importance
- **Feature 1 (Days Since Last Outbound Activity - Recency)** is by far the strongest driver of customer churn. High recency days values lead to strongly positive SHAP values, signaling highly increased risk of churn.
- **Feature 2 (March Activity Momentum Ratio)** represents usage velocity. Low values (< 0.8) indicate a decline in usage frequency relative to January/February, driving positive SHAP values (increased churn likelihood).
- **Feature 9 (Absolute Balance Level)** acts as an anchor: users who have significant balance levels on March 31st show highly negative SHAP values, indicating a strong likelihood of staying active (CHURN = 0).
- **Feature 10 (Inbound Support Frequency)**: Having regular incoming transfers acts as an external engagement anchor. Higher inbound counts push SHAP values negative, preventing churn.

## 2. Dependence Plots Insights
- **F1_recency_days**: Customers show a dramatic increase in churn risk starting around day 15 to 20 of inactivity, capping out at day 30.
- **F2_momentum_ratio**: A clear downward trend in velocity (momentum ratio dropping below 1.0) shows a steady, monotonic rise in churn risk.

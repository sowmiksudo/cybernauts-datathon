# FictiPay Customer Churn Prediction — Feature Library Documentation

This document describes the 10 core features implemented for theNSUCEC FictiPay Datathon, including their mathematical formulations and business rationales.

## Feature Library

| ID | Feature Name | Formula | Category | Business Rationale |
|---|---|---|---|---|
| **F1** | Days Since Last Outbound Activity | $T_{end} - \max(\text{Outbound Datetime})$ | Recency | Measures inactivity. A higher value indicates the user has been inactive for longer and is more likely to churn. |
| **F2** | March Activity Momentum Ratio | $\frac{C_{\text{March}} + 1}{\frac{1}{2}(C_{\text{Jan}} + C_{\text{Feb}}) + 1}$ | Velocity | Captures frequency change. Values < 1.0 mean activity is slowing down as the prediction window approaches, indicating churn risk. |
| **F3** | Outbound Volume Concentration Trend | $\frac{\sum \text{Vol}_{\text{March}}}{\sum \text{Vol}_{\text{Jan-Feb}} + 1}$ | Monetary | Tracks spending velocity. Drops in transaction amounts month-over-month capture high-value users scaling back. |
| **F4** | Balance Depletion Rate | $\overline{B}_{\text{March}} - \overline{B}_{\text{Jan}}$ | Trend | Users who are draining their wallets before abandoning the platform will show negative depletion rates. |
| **F5** | Balance Volatility Index | Standard Deviation of Daily Balances | Dispersion | Active users exhibit high wallet volatility. Dormant accounts have volatility close to zero. |
| **F6** | Platform Pass-Through Index | $\frac{\text{Vol}_{\text{CashOut}}}{\text{Vol}_{\text{CashIn}} + 1}$ | Behavioral | Users with ratios close to 1.0 treat the wallet as a pass-through (quick cash-out) and do not retain value on the platform, making them high-risk. |
| **F7** | Digital Ecosystem Interdependence | $\frac{\text{Count}(\text{MerchantPay}) + \text{Count}(\text{BillPay})}{\text{Total Outbound Count} + 1}$ | Utility | Stickier users use utility features like merchant checkout and bills. Basic P2P-only users are easier to lose. |
| **F8** | Vintage Account Factor | $T_{end} - \text{Open Date}$ | Profile | New users are less habituated and have higher churn rates than older vintage accounts. |
| **F9** | Absolute Balance Level | $B_{\text{Mar 31}}$ | Liquidity | Accounts with substantial remaining funds are highly unlikely to go completely inactive in the following month. |
| **F10** | Inbound Support Frequency | Count of inbound transfers | Network | Regular inbound transfers from friends/employers act as an external anchor keeping the account active. |

## Feature Quality Transformations

- **Winsorization**: Extreme outliers in transactions and balance variables capped at the 99th percentile to prevent decision boundary distortion in boosting models.
- **Log-Transforms**: Applied $log(x + 1)$ to right-skewed variables (amounts, balances, inbound frequencies) to stabilize variance and normalize distributions.
- **Dormancy indicator**: Created binary `is_dormant_last_30d` (F1 > 30 days) to explicitly mark inactive accounts.

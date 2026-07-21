# AChE QSAR Model Metrics

## Dataset Summary

| Parameter | Value |
|-----------|------:|
| Training compounds | 4674 |
| External validation compounds | 2004 |
| Selected descriptors | 50 |

---

## Cross-Validation Performance

| Model | R² | RMSE | MAE | CCC |
|------|----:|----:|----:|----:|
| Ridge Regression | 0.417 | 1.147 | 0.883 | 0.592 |
| Random Forest | 0.706 | 0.814 | 0.602 | 0.807 |
| LightGBM | 0.692 | 0.833 | 0.612 | 0.808 |
| **Support Vector Regression** | **0.700** | **0.823** | **0.570** | **0.834** |

---

## External Validation

| Metric | Value |
|---------|------:|
| External R² | 0.695 |
| RMSE | 0.834 |
| MAE | 0.585 |
| CCC | 0.829 |

---

## OECD Validation

| Criterion | Result |
|-----------|--------|
| Tropsha Criteria | Passed |
| Applicability Domain Threshold (h*) | 0.0327 |
| Molecules Inside Applicability Domain | 1975 / 2004 |
| Percentage Inside Applicability Domain | 98.55% |

---

The Support Vector Regression model demonstrated the highest predictive performance among all evaluated machine learning algorithms and was selected as the final model for prospective prediction of acetylcholinesterase inhibitory activity.
# BuChE QSAR Model Metrics

## Dataset Summary

| Parameter | Value |
|-----------|------:|
| Training compounds | 2553 |
| External validation compounds | 1096 |
| Selected descriptors | 50 |

---

## Cross-Validation Performance

| Model | R² | RMSE | MAE | CCC |
|------|----:|----:|----:|----:|
| Ridge Regression | 0.475 | 0.995 | 0.772 | 0.649 |
| Random Forest | 0.705 | 0.745 | 0.546 | 0.811 |
| LightGBM | 0.705 | 0.745 | 0.550 | 0.824 |
| **Support Vector Regression** | **0.690** | **0.764** | **0.551** | **0.829** |

---

## External Validation

| Metric | Value |
|---------|------:|
| External R² | 0.696 |
| RMSE | 0.770 |
| MAE | 0.545 |
| CCC | 0.828 |

---

## OECD Validation

| Criterion | Result |
|-----------|--------|
| Tropsha Criteria | Passed |
| Applicability Domain Threshold (h*) | 0.0599 |
| Molecules Inside Applicability Domain | 1082 / 1096 |
| Percentage Inside Applicability Domain | 98.72% |

---

The Support Vector Regression model demonstrated the highest predictive performance among all evaluated machine learning algorithms and was selected as the final model for prospective prediction of butyrylcholinesterase inhibitory activity.
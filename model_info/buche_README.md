# Butyrylcholinesterase (BuChE) QSAR Model

This directory contains the machine learning QSAR model developed for predicting butyrylcholinesterase (BuChE) inhibitory activity from molecular structure. The model was developed using a curated BindingDB dataset and validated according to the OECD principles for QSAR model development and validation.

## Dataset

| Parameter | Value |
|-----------|------:|
| Target | Butyrylcholinesterase (BuChE) |
| Training compounds | 2553 |
| External validation compounds | 1096 |
| Selected molecular descriptors | 50 |

## Machine Learning Algorithms Evaluated

- Ridge Regression
- Random Forest
- LightGBM
- Support Vector Regression (SVR)

## Champion Model

**Support Vector Regression (SVR)**

## Model Performance

| Metric | Value |
|---------|------:|
| Cross-validation R² | 0.690 |
| External R² | 0.696 |
| External RMSE | 0.770 |
| External MAE | 0.545 |
| Lin's CCC | 0.828 |

## OECD Validation

| Criterion | Result |
|-----------|--------|
| Tropsha Criteria | Passed |
| Applicability Domain (h*) | 0.0599 |
| Molecules inside AD | 98.72% |

## Contents

```
models/
    Trained model and preprocessing objects

examples/
    Example prediction input and output

figures/
    Correlation plot and Williams plot

results/
    Model performance summary
```

## Citation

If you use this model in your work, please cite the accompanying publication (to be added).
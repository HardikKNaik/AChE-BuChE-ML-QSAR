# AChE-BuChE-ML-QSAR

Machine learning-based Quantitative Structure–Activity Relationship (QSAR) models for predicting acetylcholinesterase (AChE) and butyrylcholinesterase (BuChE) inhibitory activity using curated BindingDB datasets.

---

## Overview

This repository contains reproducible machine learning workflows for developing, validating, and deploying QSAR models for cholinesterase inhibition. The pipeline follows OECD principles for QSAR model development and includes descriptor generation, feature selection, model training, applicability domain analysis, and prediction of novel compounds.

---

## Features

- Automated data curation
- Molecular descriptor generation
- Feature selection
- Activity-stratified data splitting
- Multiple machine learning algorithms
  - Ridge Regression
  - Random Forest
  - LightGBM
  - Support Vector Regression (SVR)
- OECD-compliant model validation
- Applicability Domain (Williams Plot)
- Y-randomization analysis
- Prediction pipeline
- Publication-quality visualizations

---

## Repository Structure

```
AChE-BuChE-ML-QSAR/

├── scripts/
├── models/
│   ├── ache/
│   └── buche/
├── examples/
│   ├── ache/
│   └── buche/
├── figures/
│   ├── ache/
│   └── buche/
├── results/
│   ├── ache/
│   └── buche/
├── docs/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
└── CITATION.cff
```

---

## Workflow

```
Dataset Collection
        │
        ▼
Data Cleaning
        │
        ▼
Descriptor Generation
        │
        ▼
Dataset Splitting
        │
        ▼
Feature Selection
        │
        ▼
Model Development
        │
        ▼
Internal Validation
        │
        ▼
External Validation
        │
        ▼
Applicability Domain Analysis
        │
        ▼
Prediction
```

---

## Included Models

| Target | Champion Model |
|---------|----------------|
| AChE | Support Vector Regression (SVR) |
| BuChE | Support Vector Regression (SVR) |

---

## Requirements

- Python ≥3.10
- RDKit
- pandas
- numpy
- scikit-learn
- LightGBM
- matplotlib
- joblib

---

## License

This project is distributed under the MIT License.

---

## Citation

If you use this repository in your research, please cite the accompanying publication (to be added).
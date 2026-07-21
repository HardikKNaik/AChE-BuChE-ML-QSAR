#!/usr/bin/env python3
"""
04_feature_selector.py
High-Performance Feature Selection Pipeline for QSAR.
Uses rapid variance filters, correlation thresholds, and LightGBM ranking.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb

def print_banner(title):
    print("\n" + "="*60)
    print(f" {title.center(58)} ")
    print("="*60)

def main():
    print_banner("Interactive QSAR Feature Selector")
    
    # --- Check for inputs ---
    train_path = "qsar_train.csv"
    val_path = "qsar_val.csv"
    test_path = "qsar_test.csv"
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print("❌ Error: Splitted data files missing! Run 03_data_splitter.py first.")
        return

    print("📖 Loading datasets (Keeping External Test untouched in memory)...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Auto-detect meta columns
    id_col = 'ID' if 'ID' in train_df.columns else train_df.columns[0]
    target_col = 'pIC50' if 'pIC50' in train_df.columns else train_df.columns[1]
    
    # Extract feature names
    all_features = [c for c in train_df.columns if c not in [id_col, target_col]]
    print(f"Initial Feature Count: {len(all_features)}")

    # --- Step 1: Low Variance Filter ---
    # Double check to remove near-constant columns on the training split
    print("\n⚡ Step 1: Removing near-constant features (Variance < 0.002)...")
    train_vars = train_df[all_features].var()
    kept_features_var = train_vars[train_vars >= 0.002].index.tolist()
    print(f"   Features remaining after variance filter: {len(kept_features_var)}")

    # --- Step 2: High Correlation Filter ---
    print("\n⚡ Step 2: Running quick correlation check to remove redundant features...")
    corr_threshold = input("   Enter correlation threshold to drop duplicates (0.80-0.99) [Default: 0.95]: ").strip() or "0.95"
    corr_threshold = float(corr_threshold)
    
    # Compute correlation matrix on Training data only
    corr_matrix = train_df[kept_features_var].corr().abs()
    
    # Fast selection of upper triangle elements to drop
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    
    kept_features_corr = [c for c in kept_features_var if c not in to_drop]
    print(f"   Dropped {len(to_drop)} collinear features.")
    print(f"   Features remaining after collinearity filter: {len(kept_features_corr)}")

    # --- Step 3: Fast Tree-based Importance Ranking ---
    print("\n⚡ Step 3: Training an ultra-fast LightGBM model to rank feature importance...")
    
    X_train = train_df[kept_features_corr]
    y_train = train_df[target_col]
    X_val = val_df[kept_features_corr]
    y_val = val_df[target_col]
    
    # Define hyper-fast, shallow LightGBM parameters to avoid overfitting during selection
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.08,
        'num_leaves': 31,
        'verbosity': -1,
        'n_jobs': -1, # Use all available CPU cores
        'random_state': 42
    }
    
    # Train model
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    # Extract importances
    importances = gbm.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': kept_features_corr,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Filter out features with absolute zero importance
    active_importance_df = importance_df[importance_df['Importance'] > 0]
    print(f"   LightGBM found {len(active_importance_df)} features with useful predictive contribution.")

    # --- User selection of how many features to keep ---
    print(f"\nHow many top features do you want to keep for the final models?")
    print(f"Fewer features = Faster training & lower risk of over-fitting.")
    print(f"Suggested values: 50, 100, 150, or 200.")
    
    num_to_keep = input("Enter number of top features to retain [Default: 100]: ").strip() or "100"
    num_to_keep = min(int(num_to_keep), len(active_importance_df))

    top_features = active_importance_df['Feature'].head(num_to_keep).tolist()
    
    print(f"\nSelected top {len(top_features)} features.")
    print("Top 10 most critical features identified:")
    for idx, row in active_importance_df.head(10).iterrows():
        print(f"  - {row['Feature']}: (Gain score: {row['Importance']:.2f})")

    # --- Save Subset Matrices across ALL splits safely ---
    print("\n💾 Subsetting columns and exporting finalized QSAR-ready matrices...")
    
    cols_to_save = [id_col, target_col] + top_features
    
    train_df[cols_to_save].to_csv("qsar_train_selected.csv", index=False)
    val_df[cols_to_save].to_csv("qsar_val_selected.csv", index=False)
    test_df[cols_to_save].to_csv("qsar_test_selected.csv", index=False)
    
    print_banner("Feature Selection Complete")
    print(f"Final Data Matrices Matrix Shape: ({train_df.shape[0]}, {len(cols_to_save)})")
    print("Files successfully generated:")
    print(" -> qsar_train_selected.csv\n -> qsar_val_selected.csv\n -> qsar_test_selected.csv")
    print("\nNext step: Train final ML/DL architectures on optimized data (05_model_trainer.py)!")

if __name__ == "__main__":
    main()
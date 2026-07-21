#!/usr/bin/env python3
"""
05_model_trainer_oecd.py
OECD-Compliant QSAR Modeling & External Validation Suite.
Implements Activity-Stratified 10-Fold CV on the standalone Training set, 
and evaluates against a combined Validation + Test External Pool.
Includes Lin's CCC, Tropsha Criteria, and Williams Plot Applicability Domain (AD).
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb

def print_banner(title, file=None):
    banner = "\n" + "="*65 + f"\n {title.center(63)} \n" + "="*65
    print(banner)
    if file:
        file.write(banner + "\n")

def calculate_ccc(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient."""
    true_mean, pred_mean = np.mean(y_true), np.mean(y_pred)
    true_var, pred_var = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - true_mean) * (y_pred - pred_mean))
    denominator = true_var + pred_var + (true_mean - pred_mean) ** 2
    return (2 * covariance) / denominator if denominator != 0 else 0.0

def check_tropsha(y_true, y_pred, r2):
    """Verifies Tropsha criteria for QSAR model acceptability."""
    if r2 < 0.60: return False
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    k = np.sum(y_true * y_pred) / np.sum(y_pred ** 2) if np.sum(y_pred ** 2) != 0 else 0
    k_prime = np.sum(y_true * y_pred) / np.sum(y_true ** 2) if np.sum(y_true ** 2) != 0 else 0
    
    r2_0 = 1 - (np.sum((y_true - k * y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    r_prime2_0 = 1 - (np.sum((y_pred - k_prime * y_true) ** 2) / np.sum((y_pred - np.mean(y_pred)) ** 2))
    
    cond1 = (r2 - r2_0) / r2 < 0.1 or (r2 - r_prime2_0) / r2 < 0.1
    cond2 = 0.85 <= k <= 1.15 or 0.85 <= k_prime <= 1.15
    
    return cond1 and cond2

def compute_applicability_domain(X_train, X_external):
    """Calculates Leverage (Hat Matrix) to establish Applicability Domain boundaries."""
    X_tr = X_train.values
    X_ex = X_external.values
    
    try:
        xtx_inv = np.linalg.pinv(np.dot(X_tr.T, X_tr))
    except np.linalg.LinAlgError:
        return np.zeros(len(X_external)), 0.0
        
    n, p = X_tr.shape
    h_star = (3.0 * (p + 1)) / n
    
    leverages = np.zeros(len(X_ex))
    for i in range(len(X_ex)):
        leverages[i] = np.dot(np.dot(X_ex[i], xtx_inv), X_ex[i].T)
        
    return leverages, h_star

def get_model(model_name):
    if model_name == "Ridge":
        return Ridge(alpha=10.0)
    elif model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=42, n_jobs=-1)
    elif model_name == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=250, learning_rate=0.04, num_leaves=31, random_state=42, verbosity=-1, n_jobs=-1)
    elif model_name == "SVR":
        return SVR(C=15.0, epsilon=0.05, kernel='rbf')

def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"qsar_oecd_report_{timestamp}.log"
    log_file = open(log_filename, "w", encoding="utf-8")
    
    print_banner("OECD-Compliant Advanced QSAR Benchmarking Platform", log_file)
    print(f"Tracking run log: {log_filename}\n")
    
    # --- 1. Load Data Elements ---
    try:
        train_df = pd.read_csv("qsar_train_selected.csv")
        val_df = pd.read_csv("qsar_val_selected.csv")
        test_df = pd.read_csv("qsar_test_selected.csv")
    except FileNotFoundError:
        print("❌ Missing base partitions! Run earlier cleaning pipelines first.")
        return

    # Keep Train as is
    id_col, target_col = train_df.columns[0], 'pIC50'
    features = [c for c in train_df.columns if c not in [id_col, target_col]]
    
    X_train = train_df[features]
    y_train = train_df[target_col]
    
    # Club Test and Validation together to form the ultimate external validation pool
    external_df = pd.concat([val_df, test_df], axis=0).reset_index(drop=True)
    X_external = external_df[features]
    y_external = external_df[target_col]
    
    msg = f"-> Isolated Training Dataset Population  : {len(X_train)} molecules\n-> Combined External Test Pool Population: {len(X_external)} molecules\n-> Structural Explanatory Features       : {len(features)}"
    print(msg)
    log_file.write(msg + "\n")

    # --- 2. Scale Features ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_external_scaled = pd.DataFrame(scaler.transform(X_external), columns=features)

    # --- 3. Stratified 10-Fold CV Configuration ---
    print_banner("Executing Stratified 10-Fold Internal Cross-Validation", log_file)
    
    activity_bins = pd.qcut(y_train, q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    model_names = ["Ridge", "Random Forest", "LightGBM", "SVR"]
    cv_summary = {}

    for name in model_names:
        print(f"  ⚡ Cross-validating algorithm: {name}...")
        f_r2, f_rmse, f_mae, f_ccc = [], [], [], []
        
        X_pool = X_train_scaled if name in ["Ridge", "SVR"] else X_train
        
        for train_idx, val_idx in skf.split(X_pool, activity_bins):
            X_tr, X_va = X_pool.iloc[train_idx], X_pool.iloc[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = get_model(name)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            
            r2 = r2_score(y_va, preds)
            f_r2.append(r2)
            f_rmse.append(np.sqrt(mean_squared_error(y_va, preds)))
            f_mae.append(mean_absolute_error(y_va, preds))
            f_ccc.append(calculate_ccc(y_va, preds))
            
        cv_summary[name] = {
            'R2': np.mean(f_r2), 'RMSE': np.mean(f_rmse), 'MAE': np.mean(f_mae), 'CCC': np.mean(f_ccc)
        }

    # Print internal cross-validation matrix
    table_fmt = "{:<15} | {:<8} | {:<8} | {:<8} | {:<8}"
    hdr = table_fmt.format("Architecture", "CV R²", "CV RMSE", "CV MAE", "CV CCC")
    print("\n" + hdr + "\n" + "-"*55)
    log_file.write("\n" + hdr + "\n" + "-"*55 + "\n")
    for name in model_names:
        s = cv_summary[name]
        row = table_fmt.format(name, f"{s['R2']:.3f}", f"{s['RMSE']:.3f}", f"{s['MAE']:.3f}", f"{s['CCC']:.3f}")
        print(row)
        log_file.write(row + "\n")

    # --- 4. Champion Selection & Vault Assessment ---
    champion_name = max(cv_summary, key=lambda k: cv_summary[k]['CCC'])
    print_banner(f"Champion Architecture Selected: {champion_name}", log_file)
    
    # Train champion on complete training partition
    X_final_train = X_train_scaled if champion_name in ["Ridge", "SVR"] else X_train
    X_final_external = X_external_scaled if champion_name in ["Ridge", "SVR"] else X_external
    
    champion_model = get_model(champion_name)
    champion_model.fit(X_final_train, y_train)
    
    # Run Combined External Predictions
    ext_preds = champion_model.predict(X_final_external)
    ext_r2 = r2_score(y_external, ext_preds)
    ext_rmse = np.sqrt(mean_squared_error(y_external, ext_preds))
    ext_mae = mean_absolute_error(y_external, ext_preds)
    ext_ccc = calculate_ccc(y_external, ext_preds)
    
    # --- 5. Applicability Domain Analysis ---
    print("🛡️ Computing Hat Leverage Matrix boundaries (OECD Principle 3)...")
    leverages, h_star = compute_applicability_domain(X_train_scaled, X_external_scaled)
    inside_ad_mask = leverages <= h_star
    pct_inside = (np.sum(inside_ad_mask) / len(leverages)) * 100.0
    
    # Metrics isolated purely inside the applicability domain
    if np.sum(inside_ad_mask) > 0:
        ad_r2 = r2_score(y_external[inside_ad_mask], ext_preds[inside_ad_mask])
        ad_ccc = calculate_ccc(y_external[inside_ad_mask], ext_preds[inside_ad_mask])
    else:
        ad_r2, ad_ccc = 0.0, 0.0

    # --- 6. Print Comprehensive Performance Metrics ---
    tropsha_pass = check_tropsha(y_external, ext_preds, ext_r2)
    
    report = [
        f"Champion Model Architecture    : {champion_name}",
        f"Combined External Pool Size    : {len(y_external)} structures (Val + Test)",
        f"Global External Q² (R² ext)    : {ext_r2:.3f}   (Passes Tropsha threshold > 0.60: {ext_r2 > 0.60})",
        f"Global External Concordance CCC: {ext_ccc:.3f}",
        f"Global External RMSE           : {ext_rmse:.3f}",
        f"Global External MAE            : {ext_mae:.3f}",
        f"Tropsha structural validation   : {'PASSED' if tropsha_pass else 'FAILED'}",
        f"---------------------------------------------------------",
        f"Applicability Domain Limit (h*): {h_star:.4f}",
        f"Molecules inside safe AD zone  : {pct_inside:.2f}% ({np.sum(inside_ad_mask)} / {len(leverages)})",
        f"Predictive Q² inside safe AD   : {ad_r2:.3f}",
        f"Concordance CCC inside safe AD : {ad_ccc:.3f}"
    ]
    
    print("\n")
    for line in report:
        print(f"  {line}")
        log_file.write(line + "\n")

    # --- 7. Serialize Artifact Pipeline ---
    print("\n💾 Archiving project blueprints...")
    joblib.dump(champion_model, "best_qsar_model.joblib")
    joblib.dump(scaler, "qsar_scaler.joblib")
    joblib.dump({'h_star': h_star, 'X_train_scaled': X_train_scaled}, "qsar_ad_domain.joblib")
    
    msg = "Artifacts serialized successfully:\n -> best_qsar_model.joblib\n -> qsar_scaler.joblib\n -> qsar_ad_domain.joblib"
    print(msg)
    log_file.write("\n" + msg + "\n")
    
    log_file.close()
    print_banner("OECD Pipeline Protocol Complete", None)

if __name__ == "__main__":
    main()
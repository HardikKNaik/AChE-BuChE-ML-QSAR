#!/usr/bin/env python3
"""
08_plot_generator.py
Generates publication-quality Williams plots and correlation scatter figures
for the finalized QSAR validation dataset.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean, publication-style visual parameters
sns.set_theme(style="ticks")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def main():
    if not (os.path.exists("best_qsar_model.joblib") and os.path.exists("qsar_train_selected.csv")):
        print("❌ Prerequisites missing. Please execute step 5 first.")
        return

    # --- Load Data Arrays ---
    train_df = pd.read_csv("qsar_train_selected.csv")
    ext_df = pd.concat([pd.read_csv("qsar_val_selected.csv"), pd.read_csv("qsar_test_selected.csv")], axis=0).reset_index(drop=True)
    
    features = [c for c in train_df.columns if c not in [train_df.columns[0], 'pIC50']]
    
    model = joblib.load("best_qsar_model.joblib")
    scaler = joblib.load("qsar_scaler.joblib")
    ad_data = joblib.load("qsar_ad_domain.joblib")
    h_star = ad_data['h_star']
    
    # Process Coordinates
    X_tr_s = ad_data['X_train_scaled'].values
    X_ex_s = scaler.transform(ext_df[features])
    
    # --- Generate Predictions ---
    train_preds = model.predict(X_tr_s)
    ext_preds = model.predict(X_ex_s)
    
    # Calculate Residuals & Leverages
    xtx_inv = np.linalg.pinv(np.dot(X_tr_s.T, X_tr_s))
    
    train_leverages = np.array([np.dot(np.dot(row, xtx_inv), row.T) for row in X_tr_s])
    ext_leverages = np.array([np.dot(np.dot(row, xtx_inv), row.T) for row in X_ex_s])
    
    train_residuals = train_df['pIC50'].values - train_preds
    ext_residuals = ext_df['pIC50'].values - ext_preds
    
    # Standardize Residuals using residual standard deviation
    res_std = np.std(train_residuals)
    train_std_res = train_residuals / res_std
    ext_std_res = ext_residuals / res_std

    # ==========================================
    # PLOT 1: PREDICTED VS EXPERIMENTAL
    # ==========================================
    plt.figure(figsize=(6, 5.5))
    plt.scatter(train_df['pIC50'], train_preds, alpha=0.5, color='#4A90E2', label=f'Training Set (n={len(train_df)})', edgecolors='w', s=25)
    plt.scatter(ext_df['pIC50'], ext_preds, alpha=0.7, color='#E056FD', label=f'External Test Pool (n={len(ext_df)})', edgecolors='w', s=25)
    
    # 45-degree reference line
    min_val = min(train_df['pIC50'].min(), ext_df['pIC50'].min()) - 0.5
    max_val = max(train_df['pIC50'].max(), ext_df['pIC50'].max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Ideal Alignment')
    
    plt.xlabel('Experimental $pIC_{50}$')
    plt.ylabel('Predicted $pIC_{50}$')
    plt.title('QSAR Activity Prediction Correlation (SVR Champion)', fontweight='bold', pad=12)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig('qsar_plot_correlation.png', dpi=300)
    plt.close()
    print("✅ Figure generated successfully: qsar_plot_correlation.png")

    # ==========================================
    # PLOT 2: WILLIAMS PLOT (APPLICABILITY DOMAIN)
    # ==========================================
    plt.figure(figsize=(7, 5.5))
    plt.scatter(train_leverages, train_std_res, alpha=0.4, color='#4A90E2', label='Training Set', s=20)
    plt.scatter(ext_leverages, ext_std_res, alpha=0.7, color='#E056FD', label='External Test Pool', edgecolors='k', linewidths=0.3, s=25)
    
    # Horizontal outlier thresholds
    plt.axhline(y=3.0, color='#FF7675', linestyle=':', linewidth=1.5)
    plt.axhline(y=-3.0, color='#FF7675', linestyle=':', linewidth=1.5)
    # Vertical leverage warning threshold
    plt.axvline(x=h_star, color='#D63031', linestyle='--', linewidth=1.5)
    
    # Add textual boundary labels on the canvas frame
    plt.text(h_star + (max(ext_leverages.max(), train_leverages.max())*0.02), 2.5, f'$h^*$ threshold\n({h_star:.4f})', color='#D63031', fontweight='bold')
    
    plt.xlabel('Leverage Value ($h$)')
    plt.ylabel('Standardized Residual ($\delta$)')
    plt.title('Williams Plot & Applicability Domain Distribution Profile', fontweight='bold', pad=12)
    plt.xlim(-0.002, max(ext_leverages.max(), train_leverages.max()) + 0.01)
    plt.ylim(min(train_std_res.min(), ext_std_res.min()) - 0.5, max(train_std_res.max(), ext_std_res.max()) + 0.5)
    plt.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    plt.savefig('qsar_plot_williams.png', dpi=300)
    plt.close()
    print("✅ Figure generated successfully: qsar_plot_williams.png")

if __name__ == "__main__":
    main()
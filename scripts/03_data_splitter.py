#!/usr/bin/env python3
"""
03_data_splitter.py
Activity-Stratified Train/Validation/Test Splitter for QSAR
Creates representative datasets based on target variable distribution.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def print_banner(title):
    print("\n" + "="*60)
    print(f" {title.center(58)} ")
    print("="*60)

def main():
    print_banner("Activity-Stratified QSAR Data Splitter")
    
    # --- Interactive Input File ---
    default_input = "clean_descriptors.csv"
    input_file = input(f"Enter your descriptor matrix path [Default: {default_input}]: ").strip() or default_input
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found.")
        return

    print("📖 Loading dataset...")
    df = pd.read_csv(input_file)
    print(f"Loaded matrix shape: {df.shape}")

    # Auto-detect key columns
    id_col = 'ID' if 'ID' in df.columns else df.columns[0]
    target_col = 'pIC50' if 'pIC50' in df.columns else df.columns[1]

    # --- Interactive Split Configuration ---
    print("\nSelect a split ratio profile (Train / Validation / Test):")
    print("1) 70 / 15 / 15")
    print("2) 60 / 20 / 20")
    print("3) 80 / 10 / 10")
    print("4) Custom profile")
    
    choice = input("\nEnter choice (1-4) [Default: 1]: ").strip() or "1"
    
    if choice == "1":
        train_p, val_p, test_p = 0.70, 0.15, 0.15
    elif choice == "2":
        train_p, val_p, test_p = 0.60, 0.20, 0.20
    elif choice == "3":
        train_p, val_p, test_p = 0.80, 0.10, 0.10
    elif choice == "4":
        print("\nEnter custom percentages (must sum up to 100):")
        try:
            train_p = float(input("Train %: ")) / 100.0
            val_p = float(input("Validation %: ")) / 100.0
            test_p = float(input("Test %: ")) / 100.0
        except ValueError:
            print("❌ Invalid numeric inputs. Defaulting to 70/15/15.")
            train_p, val_p, test_p = 0.70, 0.15, 0.15
    else:
        train_p, val_p, test_p = 0.70, 0.15, 0.15

    # Simple rounding error check
    if not np.isclose(train_p + val_p + test_p, 1.0):
        print(f"⚠️ Warning: Ratios sum to {train_p+val_p+test_p:.2f} instead of 1.0. Normalizing...")
        total = train_p + val_p + test_p
        train_p, val_p, test_p = train_p/total, val_p/total, test_p/total

    print(f"\nFinal Split Configuration Target: Train={train_p*100:.1f}%, Val={val_p*100:.1f}%, Test={test_p*100:.1f}%")

    # --- Stratification via Quantile Binning ---
    print("\n📊 Binning activity values via quantiles to handle sparse groups...")
    
    # We will use 10 quantiles (deciles) to capture smooth shape distributions
    n_bins = 10
    
    # pd.qcut segments data into equal-sized bins based on sample count rank
    # duplicate values can compress bin boundaries, duplicate='drop' automatically merges them
    df['activity_bin'] = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates='drop')
    
    # --- Safe Handling of Underpopulated Bins ---
    # To split into Train/Val/Test seamlessly using stratified approaches, 
    # each bin needs a baseline population (at least 3 samples).
    bin_counts = df['activity_bin'].value_counts()
    print("Initial bin allocations:")
    for b, c in sorted(bin_counts.items()):
        print(f"  Bin {b}: {c} molecules")
        
    sparse_bins = bin_counts[bin_counts < 3].index.tolist()
    if sparse_bins:
        print(f"\n⚠️ Handling sparse activity windows: Bins {sparse_bins} have fewer than 3 structures.")
        # Re-assign elements belonging to isolated bins down to the nearest available neighbor
        for sb in sparse_bins:
            if sb > 0:
                df.loc[df['activity_bin'] == sb, 'activity_bin'] = sb - 1
            else:
                df.loc[df['activity_bin'] == sb, 'activity_bin'] = sb + 1
        print("✅ Consolidated underpopulated bins.")

    # --- Split Step 1: Extract Test Partition ---
    # To yield exactly the correct Test ratio, test size is set directly to test_p
    # Stratified split ensures the ratio is pulled proportionally from all bins
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_p, 
        stratify=df['activity_bin'], 
        random_state=42
    )
    
    # --- Split Step 2: Separate Train and Validation ---
    # Calculate the remaining relative fraction needed for the Validation set
    relative_val_p = val_p / (train_p + val_p)
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=relative_val_p, 
        stratify=train_val_df['activity_bin'], 
        random_state=42
    )

    # --- Clean Temporary Bins from Final Matrices ---
    for dataset in [train_df, val_df, test_df]:
        dataset.drop(columns=['activity_bin'], inplace=True, errors='ignore')

    # --- Verification & Distribution Checks ---
    print("\n📋 Split Summary Verification:")
    print(f"  🟢 Train Set Size      : {len(train_df)} compounds ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  🟡 Validation Set Size : {len(val_df)} compounds ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  🔵 Test (External) Size: {len(test_df)} compounds ({len(test_df)/len(df)*100:.1f}%)")

    print("\n📈 Activity Distribution Alignment (Mean ± SD pIC50):")
    print(f"  Original Data  : {df[target_col].mean():.3f} ± {df[target_col].std():.3f}")
    print(f"  Train Set      : {train_df[target_col].mean():.3f} ± {train_df[target_col].std():.3f}")
    print(f"  Validation Set : {val_df[target_col].mean():.3f} ± {val_df[target_col].std():.3f}")
    print(f"  Test Set       : {test_df[target_col].mean():.3f} ± {test_df[target_col].std():.3f}")

    # --- Save Matrices ---
    print("\n💾 Exporting split CSV files...")
    train_df.to_csv("qsar_train.csv", index=False)
    val_df.to_csv("qsar_val.csv", index=False)
    test_df.to_csv("qsar_test.csv", index=False)
    
    print_banner("Partitioning Complete")
    print("Files successfully written to disk: \n -> qsar_train.csv\n -> qsar_val.csv\n -> qsar_test.csv")
    print("\nNext step: Build the machine learning modeler script (04_model_trainer.py)!")

if __name__ == "__main__":
    main()
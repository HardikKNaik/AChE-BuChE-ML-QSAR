#!/usr/bin/env python3
"""
01_data_cleaner.py
QSAR Data Preparation Script
Cleans SMILES, removes salts, neutralizes molecules, handles duplicates via median pIC50,
and filters out entries with invalid structures.
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover

def print_banner(title):
    print("\n" + "="*60)
    print(f" {title.center(58)} ")
    print("="*60)

def neutralize_atoms(mol):
    """Neutralizes simple charged functional groups to standardize structures."""
    pattern = Chem.MolFromSmarts("[+1,-1]")
    if mol.HasSubstructMatch(pattern):
        # Explicitly handling common QSAR functional group charges if needed, 
        # but a simple uncharge or keeping it standardized works for most descriptors.
        # For advanced neutralization, RDKit's Uncharger can be used:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        uncharger = rdMolStandardize.Uncharger()
        return uncharger.uncharge(mol)
    return mol

def clean_smiles(smiles_str, remover):
    """Validates SMILES, strips salts, and returns a canonicalized SMILES."""
    if not isinstance(smiles_str, str) or smiles_str.strip() == "":
        return None
    
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None
    
    try:
        # 1. Remove salts/mixtures
        mol_stripped = remover.StripMol(mol, dontRemoveEverything=True)
        
        # 2. Neutralize charges
        mol_neutral = neutralize_atoms(mol_stripped)
        
        # 3. Generate canonical SMILES (without stereochemistry if preferred for basic 2D QSAR)
        canonical_smiles = Chem.MolToSmiles(mol_neutral, canonical=True, isomericSmiles=False)
        return canonical_smiles
    except Exception:
        return None

def main():
    print_banner("Interactive QSAR Data Cleaner")
    
    # --- Interactive File Input ---
    default_input = "bindingdb_raw.csv"
    input_file = input(f"Enter the path to your raw CSV file [Default: {default_input}]: ").strip()
    if not input_file:
        input_file = default_input
        
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found. Please check the path and run again.")
        return

    # --- Load Data ---
    print(f"\n📖 Loading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Initial Dataset Shape: {df.shape}")
    
    # --- Interactive Column Mapping ---
    print("\nColumns available:", list(df.columns))
    id_col = input("Enter the ID column name [Default: ID]: ").strip() or "ID"
    smiles_col = input("Enter the SMILES column name [Default: SMILES]: ").strip() or "SMILES"
    pic50_col = input("Enter the pIC50 column name [Default: pIC50]: ").strip() or "pIC50"

    # Quick validation of columns
    missing_cols = [c for c in [id_col, smiles_col, pic50_col] if c not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing columns in CSV: {missing_cols}")
        return

    # Drop missing target values or SMILES right off the bat
    df = df.dropna(subset=[smiles_col, pic50_col])
    # Ensure pIC50 is numeric
    df[pic50_col] = pd.to_numeric(df[pic50_col], errors='coerce')
    df = df.dropna(subset=[pic50_col])
    
    print(f"Shape after removing NaN SMILES/pIC50: {df.shape}")

    # --- Chemical Cleaning Pipeline ---
    print("\n🧪 Initializing RDKit SaltRemover & Standardizer...")
    remover = SaltRemover.SaltRemover() # Uses RDKit's default salt list
    
    print("Running curation (salts removal, neutralization, canonicalization)...")
    df['Cleaned_SMILES'] = df[smiles_col].apply(lambda x: clean_smiles(x, remover))
    
    # Drop failed structures
    failed_count = df['Cleaned_SMILES'].isna().sum()
    df = df.dropna(subset=['Cleaned_SMILES'])
    print(f"Successfully processed structures. (Removed {failed_count} invalid SMILES)")

    # --- Duplicate Merging via Median ---
    print("\n🔄 Checking for duplicate structures (based on canonicalized SMILES)...")
    
    # Count how many unique vs total rows we have now
    total_processed = len(df)
    unique_smiles_count = df['Cleaned_SMILES'].nunique()
    print(f"Total entries: {total_processed}")
    print(f"Unique chemical structures found: {unique_smiles_count}")
    
    if total_processed > unique_smiles_count:
        print("Averaging activity values for duplicate entries using the **median**...")
        
        # Group by the newly cleaned SMILES
        # Aggregate: pIC50 gets median, ID gets joined or first kept
        clean_df = df.groupby('Cleaned_SMILES').agg({
            id_col: 'first',          # Retain the first ID encountered
            pic50_col: 'median'       # Take the median of the pIC50 values
        }).reset_index()
    else:
        print("No duplicates detected!")
        clean_df = df[['Cleaned_SMILES', id_col, pic50_col]].copy()

    # Reordering columns gracefully
    clean_df = clean_df[[id_col, 'Cleaned_SMILES', pic50_col]].rename(columns={'Cleaned_SMILES': 'SMILES'})

    # --- Save Cleaned Data ---
    default_output = "01_cleaned_qsar_data.csv"
    output_file = input(f"\nEnter the output filename [Default: {default_output}]: ").strip() or default_output
    
    clean_df.to_csv(output_file, index=False)
    
    print_banner("Data Cleaning Complete")
    print(f"Final Dataset Shape: {clean_df.shape}")
    print(f"Cleaned dataset successfully saved to: {output_file}")
    print("Ready for descriptor generation (02_descriptors.py)!")

if __name__ == "__main__":
    main()
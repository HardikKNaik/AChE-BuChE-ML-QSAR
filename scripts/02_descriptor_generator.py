#!/usr/bin/env python3
"""
02_descriptor_generator.py
QSAR Feature Generation Pipeline
Generates 1D/2D RDKit Descriptors, Morgan, RDKit, Avalon, and MACCS Fingerprints.
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import MACCSkeys

def print_banner(title):
    print("\n" + "="*60)
    print(f" {title.center(58)} ")
    print("="*60)

def generate_fingerprints(mols, fp_type, radius=2, n_bits=2048):
    """Generates specific fingerprint arrays for a list of molecules."""
    fps = []
    for mol in mols:
        try:
            if fp_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            elif fp_type == 'rdkit':
                fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
            elif fp_type == 'avalon':
                fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
            elif fp_type == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
            
            # Convert bit vector to a numpy array of integers
            arr = np.zeros((1,), dtype=int)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        except Exception:
            # Fallback if a specific FP generation fails for a weird structure
            if fp_type == 'maccs':
                fps.append(np.zeros((167,), dtype=int))
            else:
                fps.append(np.zeros((n_bits,), dtype=int))
                
    return np.array(fps)

def main():
    print_banner("Interactive QSAR Descriptor Generator")
    
    # --- Interactive File Input ---
    default_input = "clean.csv"
    input_file = input(f"Enter your cleaned data path [Default: {default_input}]: ").strip() or default_input
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found.")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded dataset containing {len(df)} compounds.")

    # Auto-detect columns based on output of step 1
    id_col = 'ID' if 'ID' in df.columns else df.columns[0]
    smiles_col = 'SMILES' if 'SMILES' in df.columns else df.columns[1]
    target_col = 'pIC50' if 'pIC50' in df.columns else df.columns[2]

    print(f"Detected mapping -> ID: {id_col} | SMILES: {smiles_col} | Target: {target_col}")

    # Generate RDKit Mol objects
    print("\nParsing SMILES strings into RDKit molecules...")
    mols = [Chem.MolFromSmiles(s) for s in df[smiles_col]]
    
    # --- Interactive Choice ---
    print("\nSelect the type of features you want to generate:")
    print("1) RDKit Physicochemical Descriptors (~200 continuous properties)")
    print("2) Morgan Fingerprints (Circular / ECFP4-like bits)")
    print("3) RDKit Topological Fingerprints (Path-based bits)")
    print("4) Avalon Fingerprints (Substructure path bits)")
    print("5) MACCS Keys (166 structural keys)")
    print("6) ALL OF THE ABOVE (Merged structural + physical properties)")
    
    choice = input("\nEnter your choice (1-6) [Default: 6]: ").strip() or "6"

    # Container for out dataframe features
    feature_dfs = [df[[id_col, target_col]].copy()]

    # 1. RDKit continuous descriptors
    if choice in ['1', '6']:
        print("\n🧮 Computing RDKit Global Descriptors (LogP, MW, Polar Surface Area, etc.)...")
        calc_list = Descriptors._descList
        desc_data = []
        
        for mol in mols:
            desc_row = {}
            for name, func in calc_list:
                try:
                    desc_row[name] = func(mol)
                except Exception:
                    desc_row[name] = np.nan
            desc_data.append(desc_row)
            
        desc_df = pd.DataFrame(desc_data)
        # Drop columns that have too many NaN values due to rare structural exceptions
        desc_df = desc_df.dropna(axis=1, thresh=int(0.95 * len(desc_df)))
        # Fill remaining isolated NaNs with column mean
        desc_df = desc_df.fillna(desc_df.mean())
        
        feature_dfs.append(desc_df)
        print(f"Generated {desc_df.shape[1]} global descriptors.")

    # Bit settings for fingerprints
    nbits = 2048

    # 2. Morgan
    if choice in ['2', '6']:
        print(f"\n🧬 Generating Morgan Fingerprints (Radius=2, Bits={nbits})...")
        fp_matrix = generate_fingerprints(mols, 'morgan', radius=2, n_bits=nbits)
        fp_df = pd.DataFrame(fp_matrix, columns=[f"Morgan_{i}" for i in range(fp_matrix.shape[1])])
        feature_dfs.append(fp_df)

    # 3. RDKit
    if choice in ['3', '6']:
        print(f"\n📈 Generating RDKit Topological Fingerprints (Bits={nbits})...")
        fp_matrix = generate_fingerprints(mols, 'rdkit', n_bits=nbits)
        fp_df = pd.DataFrame(fp_matrix, columns=[f"RDKitFP_{i}" for i in range(fp_matrix.shape[1])])
        feature_dfs.append(fp_df)

    # 4. Avalon
    if choice in ['4', '6']:
        print(f"\n💎 Generating Avalon Fingerprints (Bits={nbits})...")
        fp_matrix = generate_fingerprints(mols, 'avalon', n_bits=nbits)
        fp_df = pd.DataFrame(fp_matrix, columns=[f"Avalon_{i}" for i in range(fp_matrix.shape[1])])
        feature_dfs.append(fp_df)

    # 5. MACCS Keys
    if choice in ['5', '6']:
        print(f"\n🗝️ Generating MACCS Structural Keys (166 keys)...")
        fp_matrix = generate_fingerprints(mols, 'maccs')
        # MACCS array length is usually 167 (0 index is ignored by convention but captured)
        fp_df = pd.DataFrame(fp_matrix, columns=[f"MACCS_{i}" for i in range(fp_matrix.shape[1])])
        feature_dfs.append(fp_df)

    # --- Combine data matrices ---
    print("\nConcatenating features into final layout...")
    final_df = pd.concat(feature_dfs, axis=1)

    # --- Handle Low Variance / Zero Columns for Fingerprints ---
    # Removes bits that are all 0 or all 1 across the dataset to keep data clean
    if choice != '1':
        print("Filtering out completely invariant features/bits...")
        # Ignore ID and target columns while checking variance
        feature_cols = [c for c in final_df.columns if c not in [id_col, target_col]]
        variances = final_df[feature_cols].var()
        non_zero_var_cols = variances[variances > 0.001].index.tolist()
        final_df = final_df[[id_col, target_col] + non_zero_var_cols]

    # --- Save Features ---
    default_output = "02_descriptors_output.csv"
    output_file = input(f"\nEnter output descriptor file name [Default: {default_output}]: ").strip() or default_output
    
    final_df.to_csv(output_file, index=False)
    
    print_banner("Feature Generation Complete")
    print(f"Final Feature Matrix Shape: {final_df.shape}")
    print(f"Saved dataset successfully to: {output_file}")
    print("\nNext step: split data, train, and validate ML models (03_qsar_modeler.py)!")

if __name__ == "__main__":
    main()

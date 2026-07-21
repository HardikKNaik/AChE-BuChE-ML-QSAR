#!/usr/bin/env python3
"""
07_predictor.py
QSAR Production Prediction Engine
Loads the serialized champion SVR QSAR model and scaler pipelines to project bioactivity 
(pIC50 and nM IC50) for custom structural libraries, calculating Applicability Domain boundaries on-the-fly.
"""

import os
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import MACCSkeys

def print_banner(title):
    print("\n" + "="*65 + f"\n {title.center(63)} \n" + "="*65)

def generate_single_mol_features(mol, features, n_bits=2048):
    """Extracts exactly the specific features required by the trained model pipeline."""
    desc_row = {}
    calc_dict = {name: func for name, func in Descriptors._descList}
    
    for f in features:
        if f in calc_dict:
            try:
                desc_row[f] = calc_dict[f](mol)
            except Exception:
                desc_row[f] = 0.0

    morgan_needed = [f for f in features if f.startswith("Morgan_")]
    if morgan_needed:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((1,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        for f in morgan_needed:
            bit_idx = int(f.split("_")[1])
            desc_row[f] = arr[bit_idx]

    rdkit_needed = [f for f in features if f.startswith("RDKitFP_")]
    if rdkit_needed:
        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        arr = np.zeros((1,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        for f in rdkit_needed:
            bit_idx = int(f.split("_")[1])
            desc_row[f] = arr[bit_idx]

    avalon_needed = [f for f in features if f.startswith("Avalon_")]
    if avalon_needed:
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        arr = np.zeros((1,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        for f in avalon_needed:
            bit_idx = int(f.split("_")[1])
            desc_row[f] = arr[bit_idx]

    maccs_needed = [f for f in features if f.startswith("MACCS_")]
    if maccs_needed:
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((1,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        for f in maccs_needed:
            bit_idx = int(f.split("_")[1])
            desc_row[f] = arr[bit_idx]
            
    return desc_row

def pic50_to_nm(pic50_value):
    """Converts logarithmic pIC50 back to absolute concentration units (nM)."""
    try:
        molar_conc = 10**(-pic50_value)
        nanomolar_conc = molar_conc * 1e9
        return nanomolar_conc
    except Exception:
        return np.nan

def main():
    print_banner("QSAR Bioactivity Prediction Platform")
    
    # --- Check for Saved Pipeline Blueprints ---
    if not (os.path.exists("best_qsar_model.joblib") and os.path.exists("qsar_scaler.joblib") and os.path.exists("qsar_ad_domain.joblib")):
        print("❌ Error: Production artifacts missing! Run 05_model_trainer_oecd.py first.")
        return

    # Load artifacts
    print("🧠 Loading validated model artifacts & domain descriptors...")
    model = joblib.load("best_qsar_model.joblib")
    scaler = joblib.load("qsar_scaler.joblib")
    ad_data = joblib.load("qsar_ad_domain.joblib")
    
    h_star = ad_data['h_star']
    X_tr_values = ad_data['X_train_scaled'].values
    xtx_inv = np.linalg.pinv(np.dot(X_tr_values.T, X_tr_values))
    feature_names = scaler.feature_names_in_
    
    # --- Interactive File Processing ---
    print("\n📊 Awaiting candidate structure mapping spreadsheet...")
    input_file = input("Enter path to your input CSV file containing ID and SMILES: ").strip()
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found. Verify path parameters and retry.")
        return

    df = pd.read_csv(input_file)
    if len(df.columns) < 2:
        print("❌ Error: The CSV sheet must contain at least two designated mapping columns (ID, SMILES).")
        return

    id_col = df.columns[0]
    smiles_col = df.columns[1]
    print(f"Detected Column Mapping -> Compound IDs: '{id_col}' | Molecular SMILES: '{smiles_col}'")
    print(f"Loaded {len(df)} target molecules for prediction screening.")

    # --- Feature Generation Pipeline ---
    processed_features = []
    valid_rows = []
    
    print("\n🧪 Executing characterization and feature extraction protocols...")
    for idx, row in df.iterrows():
        smiles_str = str(row[smiles_col]).strip()
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            print(f"  ⚠️ Row {idx+1} ({row[id_col]}): Invalid SMILES notation. Skipping.")
            continue
            
        # Standardize representation to match 2D training configuration
        Chem.RemoveStereochemistry(mol)
        
        # Calculate exactly the required structural features
        mol_feat = generate_single_mol_features(mol, feature_names)
        processed_features.append(mol_feat)
        valid_rows.append(row)
        
    if not processed_features:
        print("❌ Failure: Zero valid chemical identities could be resolved into descriptor records.")
        return

    # Assemble and scale target test coordinates
    X_query = pd.DataFrame(processed_features, columns=feature_names).fillna(0.0)
    X_query_scaled = scaler.transform(X_query)
    
    # --- Engine Predictions & Leverage Checks ---
    print("🔮 Projecting binding affinities and analyzing spatial leverage spaces...")
    predictions_pic50 = model.predict(X_query_scaled)
    
    output_records = []
    for i in range(len(valid_rows)):
        row_data = valid_rows[i]
        leverage = np.dot(np.dot(X_query_scaled[i], xtx_inv), X_query_scaled[i].T)
        
        pic50_pred = predictions_pic50[i]
        ic50_nm_pred = pic50_to_nm(pic50_pred)
        
        ad_status = "INSIDE" if leverage <= h_star else "OUTSIDE (Unreliable)"
        
        output_records.append({
            'ID': row_data[id_col],
            'SMILES': row_data[smiles_col],
            'Predicted_pIC50': round(pic50_pred, 3),
            'Predicted_IC50_nM': round(ic50_nm_pred, 2) if not np.isnan(ic50_nm_pred) else "NaN",
            'Leverage_h': round(leverage, 4),
            'Applicability_Domain': ad_status
        })
        
    # Build final prioritized output matrix
    output_df = pd.DataFrame(output_records)
    # Sort from highest affinity down to weakest
    output_df = output_df.sort_values(by='Predicted_pIC50', ascending=False).reset_index(drop=True)
    
    # Save output spreadsheet
    output_filename = "qsar_predictions_output.csv"
    output_df.to_csv(output_filename, index=False)
    
    print_banner("Prioritized Lead Predictions (Top Candidates)")
    display_cols = ['ID', 'Predicted_pIC50', 'Predicted_IC50_nM', 'Applicability_Domain']
    print(output_df[display_cols].head(15).to_string(index=False))
    
    print("\n" + "="*65)
    print(f"💾 Full mapped library sheet successfully written to: {output_filename}")
    print("Calculations complete. You can now use this sheet to prioritize wet lab synthesis targets!")

if __name__ == "__main__":
    main()
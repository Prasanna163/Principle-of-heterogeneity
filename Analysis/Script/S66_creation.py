#!/usr/bin/env python3
"""
Data Preparation and S66x8 Extraction
=====================================
Extract S66x8 data from the combined dataset for validation analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime

def extract_s66x8_data():
    """Extract S66x8 data (rows 2649-2848) from the combined dataset"""
    
    print("🔥 EXTRACTING S66x8 DATA FOR VALIDATION")
    print("=" * 50)
    
    # Load the complete dataset
    df_complete = pd.read_csv('pan_cahemical_raw_nci.csv')
    print(f"Complete dataset: {len(df_complete):,} rows × {len(df_complete.columns)} columns")
    
    # Extract S66x8 portion (from row 2649 onwards)
    s66x8_start_idx = 2649
    df_s66x8 = df_complete.iloc[s66x8_start_idx:].copy()
    
    # Extract base complex names (remove distance factors)
    df_s66x8['base_complex'] = df_s66x8['Complex'].apply(
        lambda x: '_'.join(x.split('_')[:-1])
    )
    
    df_s66x8['distance_factor'] = df_s66x8['Complex'].apply(
        lambda x: float(x.split('_')[-1]) if x.split('_')[-1].replace('.', '').isdigit() else 1.0
    )
    
    # Classify interaction types
    def classify_interaction_type(complex_name):
        base_name = complex_name.lower()
        
        if any(keyword in base_name for keyword in ['water', 'peptide', 'uracil-uracil_bp', 'acnh2', 'menh2']):
            return 'hydrogen_bond'
        elif any(keyword in base_name for keyword in ['benzene-benzene_pi', 'pyridine-pyridine_pi', 'uracil-uracil_pi', 'benzene-pyridine_pi']):
            return 'pi_pi_stacking'
        elif any(keyword in base_name for keyword in ['cyclopentane', 'pentane', 'ethene']):
            return 'dispersion'
        elif any(keyword in base_name for keyword in ['benzene-benzene_ts', 'pyridine-pyridine_ts', 'benzene-pyridine_ts', 'ethyne-ethyne_ts']):
            return 'edge_to_face'
        elif any(keyword in base_name for keyword in ['oh-pi', 'nh-pi', 'ch-o', 'ch-n']):
            return 'mixed_interactions'
        else:
            return 'other'
    
    df_s66x8['interaction_type'] = df_s66x8['base_complex'].apply(classify_interaction_type)
    
    # Save datasets
    df_s66x8.to_csv('KNF_Validation_Study_2025/01_Raw_Data/s66x8_extracted.csv', index=False)
    df_complete.iloc[:s66x8_start_idx].to_csv('KNF_Validation_Study_2025/01_Raw_Data/des_extracted.csv', index=False)
    df_complete.to_csv('KNF_Validation_Study_2025/01_Raw_Data/pan_cahemical_raw_nci.csv', index=False)
    
    print(f"✅ S66x8 data extracted: {len(df_s66x8)} rows")
    print(f"✅ Files saved to KNF_Validation_Study_2025/01_Raw_Data/")
    
    return df_s66x8

s66x8_data = extract_s66x8_data()

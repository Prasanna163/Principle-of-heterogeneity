import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Load merged dataset
merged_df = pd.read_csv('merged_des_dataset.csv')

print("🔍 STABILITY METRIC INTER-CORRELATIONS:")
print("="*50)

targets = {
    'SNCI': 'reference_snci',
    'Binding_Energy': 'Binding_Energy (kcal/mol)', 
    'SCDI': 'SCDI'
}

# Calculate all pairwise correlations
for target1_name, target1_col in targets.items():
    for target2_name, target2_col in targets.items():
        if target1_name != target2_name:
            corr, p_val = pearsonr(merged_df[target1_col], merged_df[target2_col])
            print(f"{target1_name:15} vs {target2_name:15} | r = {corr:6.3f} | p = {p_val:.2e}")

print("\nCRITICAL VALUES:")
snci_binding_corr = pearsonr(merged_df['reference_snci'], merged_df['Binding_Energy (kcal/mol)'])[0]
snci_scdi_corr = pearsonr(merged_df['reference_snci'], merged_df['SCDI'])[0]
binding_scdi_corr = pearsonr(merged_df['Binding_Energy (kcal/mol)'], merged_df['SCDI'])[0]

print(f"SNCI vs Binding Energy: {snci_binding_corr:.3f}")
print(f"SNCI vs SCDI: {snci_scdi_corr:.3f}")  
print(f"Binding Energy vs SCDI: {binding_scdi_corr:.3f}")

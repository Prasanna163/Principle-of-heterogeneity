import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# Load data
merged_df = pd.read_csv('merged_des_dataset.csv')

print("🧪 COMPOSITE STABILITY METRIC TESTING:")
print("="*60)

# Normalize metrics for fair combination
scaler = MinMaxScaler()
binding_energy_positive = -merged_df['Binding_Energy (kcal/mol)']  # More negative = more stable

normalized_data = scaler.fit_transform(np.column_stack([
    merged_df['reference_snci'],
    merged_df['SCDI'], 
    binding_energy_positive
]))

# Define composite formulations to test
composite_formulas = {
    'SNCI_Only': [1.0, 0.0, 0.0],
    'SNCI_SCDI_Equal': [0.5, 0.5, 0.0],
    'SNCI_SCDI_Heavy': [0.7, 0.3, 0.0],
    'All_Equal': [1/3, 1/3, 1/3],
    'SNCI_Heavy': [0.6, 0.2, 0.2],
    'Binding_Heavy': [0.2, 0.2, 0.6],
}

knf_features = [
    'f1_com_distance', 'f2_dha_angle', 'f3_max_inter_wbo', 'f4_total_dipole_moment',
    'f5_iso_polarizability', 'f6_nci_attractive_points', 'f7_nci_mean', 
    'f8_nci_std_dev', 'f9_nci_skewness'
]

print(f"{'Formula':20} | {'f8_corr':^8} | {'f8_rank':^8} | {'Top_Feature':^20}")
print("-" * 70)

results = []
for formula_name, weights in composite_formulas.items():
    # Create composite stability metric
    composite = (weights[0] * normalized_data[:, 0] + 
                weights[1] * normalized_data[:, 1] + 
                weights[2] * normalized_data[:, 2])
    
    # Calculate all feature correlations with composite
    feature_correlations = []
    for feature in knf_features:
        corr, _ = pearsonr(merged_df[feature], composite)
        feature_correlations.append((feature, corr, abs(corr)))
    
    # Sort by absolute correlation  
    feature_correlations.sort(key=lambda x: x[2], reverse=True)
    
    # Find f8 details
    f8_corr = next(corr for feat, corr, abs_corr in feature_correlations if feat == 'f8_nci_std_dev')
    f8_rank = next(i+1 for i, (feat, _, _) in enumerate(feature_correlations) if feat == 'f8_nci_std_dev')
    top_feature = feature_correlations[0][0]
    
    print(f"{formula_name:20} | {f8_corr:7.3f} | {f8_rank:^8} | {top_feature:^20}")
    
    results.append({
        'formula': formula_name,
        'f8_correlation': f8_corr,
        'f8_rank': f8_rank,
        'top_feature': top_feature,
        'all_correlations': feature_correlations
    })

# Analyze the sign cancellation effect
print(f"\n🔍 SIGN CANCELLATION ANALYSIS:")
f8_snci_corr = pearsonr(merged_df['f8_nci_std_dev'], merged_df['reference_snci'])[0]
f8_scdi_corr = pearsonr(merged_df['f8_nci_std_dev'], merged_df['SCDI'])[0]
f8_binding_corr = pearsonr(merged_df['f8_nci_std_dev'], binding_energy_positive)[0]

print(f"f8 with SNCI: {f8_snci_corr:+.3f}")
print(f"f8 with SCDI: {f8_scdi_corr:+.3f}")  
print(f"f8 with Binding: {f8_binding_corr:+.3f}")
print(f"\nSNCI+SCDI composite effect: 0.5×({f8_snci_corr:+.3f}) + 0.5×({f8_scdi_corr:+.3f}) = {0.5*f8_snci_corr + 0.5*f8_scdi_corr:+.3f}")
print("👆 This shows why f8 loses dominance in SNCI+SCDI composite!")

# Test sign-corrected composite
scdi_inverted = 1 - merged_df['SCDI']
composite_corrected = 0.5 * merged_df['reference_snci'] + 0.5 * scdi_inverted
f8_corrected_corr = pearsonr(merged_df['f8_nci_std_dev'], composite_corrected)[0]

print(f"\nWith SCDI inverted (1-SCDI):")
print(f"f8 correlation with SNCI + (1-SCDI): {f8_corrected_corr:.3f}")

# Test all features with corrected composite
corrected_correlations = []
for feature in knf_features:
    corr, _ = pearsonr(merged_df[feature], composite_corrected)
    corrected_correlations.append((feature, corr, abs(corr)))

corrected_correlations.sort(key=lambda x: x[2], reverse=True)
f8_corrected_rank = next(i+1 for i, (feat, _, _) in enumerate(corrected_correlations) if feat == 'f8_nci_std_dev')

print(f"f8 rank with corrected composite: #{f8_corrected_rank}")
print(f"Top 3: {[feat for feat, _, _ in corrected_correlations[:3]]}")

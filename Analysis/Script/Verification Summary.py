import pandas as pd
from scipy.stats import pearsonr

# Load data
merged_df = pd.read_csv('merged_des_dataset.csv')

print("✅ FINAL VERIFICATION SUMMARY:")
print("="*50)

# Key correlations to verify
key_correlations = [
    ("f8 with SNCI", 'f8_nci_std_dev', 'reference_snci'),
    ("f8 with SCDI", 'f8_nci_std_dev', 'SCDI'), 
    ("f8 with Binding Energy", 'f8_nci_std_dev', 'Binding_Energy (kcal/mol)'),
    ("SNCI with Binding Energy", 'reference_snci', 'Binding_Energy (kcal/mol)'),
    ("SNCI with SCDI", 'reference_snci', 'SCDI'),
    ("Binding Energy with SCDI", 'Binding_Energy (kcal/mol)', 'SCDI'),
]

print("Key correlations:")
for desc, col1, col2 in key_correlations:
    corr, p_val = pearsonr(merged_df[col1], merged_df[col2])
    print(f"{desc:25}: r = {corr:6.3f}, p = {p_val:.2e}")

# Feature ranking for SNCI
knf_features = [
    'f1_com_distance', 'f2_dha_angle', 'f3_max_inter_wbo', 'f4_total_dipole_moment',
    'f5_iso_polarizability', 'f6_nci_attractive_points', 'f7_nci_mean', 
    'f8_nci_std_dev', 'f9_nci_skewness'
]

snci_correlations = []
for feature in knf_features:
    corr, _ = pearsonr(merged_df[feature], merged_df['reference_snci'])
    snci_correlations.append((feature, abs(corr)))

snci_correlations.sort(key=lambda x: x[1], reverse=True)

print(f"\nFeature ranking for SNCI prediction:")
for i, (feature, abs_corr) in enumerate(snci_correlations):
    status = " 🎯" if feature == 'f8_nci_std_dev' else ""
    print(f"{i+1:2}. {feature:25} | |r| = {abs_corr:.3f}{status}")

f8_rank = next(i+1 for i, (feat, _) in enumerate(snci_correlations) if feat == 'f8_nci_std_dev')
print(f"\n🎯 f8_nci_std_dev ranks #{f8_rank} for SNCI prediction")

print(f"\nDataset info:")
print(f"Total complexes: {len(merged_df)}")
print(f"SNCI range: {merged_df['reference_snci'].min():.6f} to {merged_df['reference_snci'].max():.6f}")
print(f"Binding Energy range: {merged_df['Binding_Energy (kcal/mol)'].min():.2f} to {merged_df['Binding_Energy (kcal/mol)'].max():.2f}")
print(f"SCDI range: {merged_df['SCDI'].min():.3f} to {merged_df['SCDI'].max():.3f}")

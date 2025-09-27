import pandas as pd
from scipy.stats import pearsonr

# Load data
merged_df = pd.read_csv('merged_des_dataset.csv')

# Define KNF features
knf_features = [
    'f1_com_distance', 'f2_dha_angle', 'f3_max_inter_wbo', 'f4_total_dipole_moment',
    'f5_iso_polarizability', 'f6_nci_attractive_points', 'f7_nci_mean', 
    'f8_nci_std_dev', 'f9_nci_skewness'
]

# Define target variables
targets = {
    'SNCI': 'reference_snci',
    'Binding_Energy': 'Binding_Energy (kcal/mol)', 
    'SCDI': 'SCDI'
}

# Calculate correlations for all targets
print("🎯 FEATURE CORRELATIONS WITH EACH TARGET:")
print("="*70)

all_results = {}
for target_name, target_col in targets.items():
    print(f"\n📊 {target_name.upper()} CORRELATIONS:")
    print("-" * 40)
    
    correlations = []
    for feature in knf_features:
        corr, p_val = pearsonr(merged_df[feature], merged_df[target_col])
        correlations.append({
            'Feature': feature,
            'Correlation': corr,
            'P_Value': p_val,
            'Abs_Correlation': abs(corr)
        })
        
        print(f"{feature:25} | r = {corr:7.3f} | p = {p_val:.2e}")
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x['Abs_Correlation'], reverse=True)
    all_results[target_name] = correlations
    
    print(f"\n🏆 TOP 5 FOR {target_name}:")
    for i, corr in enumerate(correlations[:5]):
        print(f"  {i+1}. {corr['Feature']:25} | r = {corr['Correlation']:7.3f}")

# Create ranking comparison table
print(f"\n📈 FEATURE RANKING COMPARISON:")
print("="*80)
print(f"{'Feature':25} | {'SNCI':^8} | {'Binding':^8} | {'SCDI':^8} | {'f8 Status':^15}")
print("-" * 80)

for feature in knf_features:
    rankings = {}
    for target_name in targets.keys():
        rank = next(i+1 for i, corr in enumerate(all_results[target_name]) if corr['Feature'] == feature)
        rankings[target_name] = rank
    
    f8_status = "🎯 KEY!" if feature == 'f8_nci_std_dev' else ""
    
    print(f"{feature:25} | {rankings['SNCI']:^8} | {rankings['Binding_Energy']:^8} | {rankings['SCDI']:^8} | {f8_status:^15}")

# Show f8 specific results
print(f"\n🎯 f8_nci_std_dev SUMMARY:")
print(f"  SNCI correlation: {all_results['SNCI'][3]['Correlation']:.3f} (rank #{next(i+1 for i, c in enumerate(all_results['SNCI']) if c['Feature'] == 'f8_nci_std_dev')})")
print(f"  Binding correlation: {next(c['Correlation'] for c in all_results['Binding_Energy'] if c['Feature'] == 'f8_nci_std_dev'):.3f}")
print(f"  SCDI correlation: {next(c['Correlation'] for c in all_results['SCDI'] if c['Feature'] == 'f8_nci_std_dev'):.3f}")

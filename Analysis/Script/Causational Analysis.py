
"""
KULKARNI HETEROGENEITY PRINCIPLE - CAUSATION vs CORRELATION ANALYSIS
=====================================================================
Complete framework for validating causal relationships in supramolecular stability

Author: Prasanna P. Kulkarni
Purpose: Establish causation evidence for "Interaction heterogeneity as primary 
         determinant of supramolecular stability" discovery

Usage: Update CSV_PATH and run all 4 analysis modules
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION
# ================================================================

# UPDATE THIS PATH TO YOUR ACTUAL DATA
CSV_PATH = r"D:/COMP RESEARCH/KNF-VALIDATION/merged_des_dataset.csv"

# Expected columns in your dataset
EXPECTED_COLUMNS = [
    'Complex', 'reference_snci', 'f1_com_distance', 'f2_dha_angle',
    'f3_max_inter_wbo', 'f4_total_dipole_moment', 'f5_iso_polarizability',
    'f6_nci_attractive_points', 'f7_nci_mean', 'f8_nci_std_dev',
    'f9_nci_skewness', 'Binding_Energy (kcal/mol)', 'SCDI'
]

# Analysis parameters
HETEROGENEITY_COL = 'f8_nci_std_dev'  # Your key discovery variable
STABILITY_COL = 'reference_snci'      # Primary stability metric
TEST_SIZE = 0.3                       # Train/test split
RANDOM_STATE = 42                     # Reproducibility

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def load_and_validate_data(filepath):
    """Load data and validate structure"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded: {len(df)} complexes")

        # Check columns
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")

        # Basic statistics
        print(f"📊 Heterogeneity range: {df[HETEROGENEITY_COL].min():.3f} to {df[HETEROGENEITY_COL].max():.3f}")
        print(f"📊 Stability range: {df[STABILITY_COL].min():.3f} to {df[STABILITY_COL].max():.3f}")

        return df

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def partial_correlation(df, x, y, control_vars):
    """Calculate partial correlation controlling for variables"""
    if not control_vars:
        return pearsonr(df[x], df[y])

    # Residualize x with respect to control variables
    reg_x = LinearRegression()
    reg_x.fit(df[control_vars], df[x])
    residuals_x = df[x] - reg_x.predict(df[control_vars])

    # Residualize y with respect to control variables  
    reg_y = LinearRegression()
    reg_y.fit(df[control_vars], df[y])
    residuals_y = df[y] - reg_y.predict(df[control_vars])

    # Correlation of residuals
    return pearsonr(residuals_x, residuals_y)

def granger_causality_test(x, y, max_lag=5):
    """Simplified Granger causality test"""
    n = len(x)
    results = {}

    for lag in range(1, min(max_lag + 1, n // 4)):
        # Create lagged variables
        X_restricted = []  # Only y lags
        X_full = []        # Both x and y lags
        y_target = y[lag:]

        for i in range(lag, n):
            # Restricted model (only y lags)
            y_lags = [y[i-j-1] for j in range(lag)]
            X_restricted.append(y_lags)

            # Full model (x and y lags)
            x_lags = [x[i-j-1] for j in range(lag)]
            X_full.append(y_lags + x_lags)

        if len(X_restricted) < 10:  # Need minimum data points
            continue

        try:
            # Fit models
            reg_restricted = LinearRegression().fit(X_restricted, y_target)
            reg_full = LinearRegression().fit(X_full, y_target)

            # Calculate improvement
            r2_restricted = reg_restricted.score(X_restricted, y_target)
            r2_full = reg_full.score(X_full, y_target)

            results[lag] = {
                'r2_improvement': r2_full - r2_restricted,
                'r2_full': r2_full,
                'r2_restricted': r2_restricted
            }
        except:
            continue

    return results

# ================================================================
# ANALYSIS MODULES
# ================================================================

def analysis_1_confounding_variables(df):
    """Test if correlation persists after controlling for potential confounders"""

    print("\n" + "="*60)
    print("📊 ANALYSIS 1: CONFOUNDING VARIABLE ANALYSIS")
    print("="*60)

    # Define potential confounding variables
    confounding_scenarios = [
        [],  # Base case
        ['f1_com_distance'],  # Molecular size
        ['f4_total_dipole_moment'],  # Electronic effects
        ['f5_iso_polarizability'],  # System complexity
        ['f6_nci_attractive_points'],  # Interaction count
        ['f1_com_distance', 'f5_iso_polarizability'],  # Size + complexity
        ['f4_total_dipole_moment', 'f6_nci_attractive_points'],  # Electronic + structural
        ['f1_com_distance', 'f4_total_dipole_moment', 'f5_iso_polarizability']  # Multi-factor
    ]

    print("Testing partial correlations controlling for confounders:")
    print("-" * 70)

    results = []
    for i, control_vars in enumerate(confounding_scenarios):
        try:
            corr, p_val = partial_correlation(df, HETEROGENEITY_COL, STABILITY_COL, control_vars)

            if i == 0:
                control_desc = "None (base correlation)"
                base_corr = corr
            else:
                control_desc = " + ".join(control_vars)

            print(f"Controlling for {control_desc:<35}: r = {corr:.3f}, p = {p_val:.2e}")
            results.append({'controls': control_vars, 'correlation': corr, 'p_value': p_val})

        except Exception as e:
            print(f"Error with controls {control_vars}: {e}")

    # Assess robustness
    if len(results) > 1:
        correlation_changes = [abs(base_corr - r['correlation']) for r in results[1:]]
        avg_change = np.mean(correlation_changes)
        max_change = max(correlation_changes)

        print(f"\nRobustness Assessment:")
        print(f"Average change in correlation: {avg_change:.3f}")
        print(f"Maximum change in correlation: {max_change:.3f}")

        if avg_change < 0.05:
            print("✅ VERY ROBUST: Minimal confounding effect")
            robustness = "very_robust"
        elif avg_change < 0.1:
            print("✅ ROBUST: Correlation persists after controlling")
            robustness = "robust"
        elif avg_change < 0.2:
            print("⚠️  MODERATE: Some confounding influence detected")
            robustness = "moderate"
        else:
            print("❌ WEAK: Strong confounding effects present")
            robustness = "weak"
    else:
        robustness = "unable_to_assess"

    return {
        'base_correlation': base_corr if 'base_corr' in locals() else None,
        'partial_correlations': results,
        'robustness': robustness
    }

def analysis_2_predictive_validation(df):
    """Test predictive power and directionality"""

    print("\n" + "="*60)
    print("🎯 ANALYSIS 2: MECHANISTIC PREDICTION VALIDATION")
    print("="*60)

    # Forward prediction: Heterogeneity → Stability
    X_forward = df[[HETEROGENEITY_COL]]
    y_forward = df[STABILITY_COL]

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_forward, y_forward, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Reverse prediction: Stability → Heterogeneity  
    X_reverse = df[[STABILITY_COL]]
    y_reverse = df[HETEROGENEITY_COL]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reverse, y_reverse, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Test multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    }

    print("Forward Prediction (Heterogeneity → Stability):")
    print("-" * 50)

    forward_results = {}
    for name, model in models.items():
        model.fit(X_train_f, y_train_f)
        y_pred_f = model.predict(X_test_f)

        r2_f = r2_score(y_test_f, y_pred_f)
        rmse_f = np.sqrt(mean_squared_error(y_test_f, y_pred_f))
        corr_f, p_f = pearsonr(y_test_f, y_pred_f)

        forward_results[name] = {'r2': r2_f, 'rmse': rmse_f, 'correlation': corr_f}
        print(f"{name:<18}: R² = {r2_f:.3f}, RMSE = {rmse_f:.3f}, r = {corr_f:.3f}")

    print("\nReverse Prediction (Stability → Heterogeneity):")
    print("-" * 50)

    reverse_results = {}
    for name, model in models.items():
        model_copy = LinearRegression() if name == 'Linear Regression' else RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        model_copy.fit(X_train_r, y_train_r)
        y_pred_r = model_copy.predict(X_test_r)

        r2_r = r2_score(y_test_r, y_pred_r)
        rmse_r = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
        corr_r, p_r = pearsonr(y_test_r, y_pred_r)

        reverse_results[name] = {'r2': r2_r, 'rmse': rmse_r, 'correlation': corr_r}
        print(f"{name:<18}: R² = {r2_r:.3f}, RMSE = {rmse_r:.3f}, r = {corr_r:.3f}")

    # Directionality analysis
    forward_r2 = forward_results['Linear Regression']['r2']
    reverse_r2 = reverse_results['Linear Regression']['r2']

    print(f"\n📊 DIRECTIONALITY ANALYSIS:")
    print(f"Forward prediction strength: R² = {forward_r2:.3f}")
    print(f"Reverse prediction strength: R² = {reverse_r2:.3f}")
    print(f"Directionality ratio: {forward_r2/reverse_r2 if reverse_r2 != 0 else 'inf':.2f}")

    if forward_r2 > reverse_r2 * 1.5:
        directionality = "forward_favored"
        print("✅ EVIDENCE: Heterogeneity → Stability direction favored")
    elif reverse_r2 > forward_r2 * 1.5:
        directionality = "reverse_favored"
        print("⚠️  CAUTION: Reverse causation (Stability → Heterogeneity) favored")
    else:
        directionality = "bidirectional"
        print("🤔 UNCLEAR: Bidirectional or weak directional relationship")

    return {
        'forward_results': forward_results,
        'reverse_results': reverse_results,
        'directionality': directionality,
        'forward_r2': forward_r2,
        'reverse_r2': reverse_r2
    }

def analysis_3_temporal_framework(df):
    """Framework for temporal/dynamic causation analysis"""

    print("\n" + "="*60)
    print("⏱️  ANALYSIS 3: TEMPORAL/DYNAMIC ANALYSIS FRAMEWORK")
    print("="*60)

    print("This analysis requires time-series data from MD simulations.")
    print("Framework provided for implementation with temporal data.")
    print("-" * 60)

    # For static data, we can test lead-lag relationships using ordering
    # Sort by heterogeneity and test if high heterogeneity precedes high stability

    df_sorted = df.sort_values(HETEROGENEITY_COL)
    n = len(df_sorted)

    # Split into thirds based on heterogeneity
    low_het = df_sorted.iloc[:n//3]
    med_het = df_sorted.iloc[n//3:2*n//3] 
    high_het = df_sorted.iloc[2*n//3:]

    stability_by_het = {
        'Low Heterogeneity': low_het[STABILITY_COL].mean(),
        'Medium Heterogeneity': med_het[STABILITY_COL].mean(),
        'High Heterogeneity': high_het[STABILITY_COL].mean()
    }

    print("Stability by Heterogeneity Level (Static Analysis):")
    for level, stab in stability_by_het.items():
        print(f"{level:<20}: {stab:.4f}")

    # Test for monotonic relationship
    het_levels = [stability_by_het['Low Heterogeneity'], 
                  stability_by_het['Medium Heterogeneity'],
                  stability_by_het['High Heterogeneity']]

    if het_levels[2] > het_levels[1] > het_levels[0]:
        monotonic = "increasing"
        print("✅ EVIDENCE: Monotonic increase in stability with heterogeneity")
    elif het_levels[0] > het_levels[1] > het_levels[2]:
        monotonic = "decreasing" 
        print("⚠️  EVIDENCE: Monotonic decrease in stability with heterogeneity")
    else:
        monotonic = "non_monotonic"
        print("🤔 MIXED: Non-monotonic relationship")

    print("\n⚡ FOR FULL TEMPORAL ANALYSIS:")
    print("1. Implement MD simulations with time-series output")
    print("2. Track heterogeneity and stability evolution over time")
    print("3. Apply Granger causality tests")
    print("4. Look for lead-lag relationships")

    return {
        'static_analysis': stability_by_het,
        'monotonic_trend': monotonic,
        'framework_status': 'static_only'
    }

def analysis_4_intervention_framework(df):
    """Framework for computational intervention experiments"""

    print("\n" + "="*60)
    print("🧪 ANALYSIS 4: INTERVENTION/PERTURBATION ANALYSIS FRAMEWORK")
    print("="*60)

    print("Framework for designing computational intervention experiments:")
    print("-" * 65)

    # Simulate what intervention experiments would look like
    base_correlation, _ = pearsonr(df[HETEROGENEITY_COL], df[STABILITY_COL])

    print(f"Current correlation strength: r = {base_correlation:.3f}")
    print("\nProposed Intervention Experiments:")
    print("1. COMPUTATIONAL PERTURBATIONS:")
    print("   - Artificially modify f8 values in subset of complexes")
    print("   - Recompute stability using quantum methods")
    print("   - Compare observed vs predicted stability changes")

    print("\n2. SYSTEMATIC DESIGN EXPERIMENTS:")
    print("   - Design new complexes with controlled heterogeneity")
    print("   - High het. group: f8 > 90th percentile of current data")
    print("   - Low het. group: f8 < 10th percentile of current data")
    print("   - Synthesize/compute and test stability predictions")

    # Calculate thresholds for design experiments
    het_90th = np.percentile(df[HETEROGENEITY_COL], 90)
    het_10th = np.percentile(df[HETEROGENEITY_COL], 10)

    print(f"\nDesign Experiment Thresholds:")
    print(f"High heterogeneity target: f8 > {het_90th:.3f}")
    print(f"Low heterogeneity target:  f8 < {het_10th:.3f}")

    # Predict expected outcomes
    high_het_expected = df[df[HETEROGENEITY_COL] > het_90th][STABILITY_COL].mean()
    low_het_expected = df[df[HETEROGENEITY_COL] < het_10th][STABILITY_COL].mean()

    print(f"\nPredicted Outcomes (based on current correlation):")
    print(f"High heterogeneity complexes: SNCI ≈ {high_het_expected:.3f}")
    print(f"Low heterogeneity complexes:  SNCI ≈ {low_het_expected:.3f}")
    print(f"Expected difference: Δ = {high_het_expected - low_het_expected:.3f}")

    print("\n3. MECHANISM INTERVENTION TESTS:")
    print("   - Test specific heterogeneity mechanisms:")
    print("     • Load distribution hypothesis")
    print("     • Redundancy hypothesis") 
    print("     • Cooperativity hypothesis")

    confidence_level = "high" if abs(base_correlation) > 0.5 else "moderate" if abs(base_correlation) > 0.3 else "low"

    return {
        'base_correlation': base_correlation,
        'design_thresholds': {'high': het_90th, 'low': het_10th},
        'predicted_outcomes': {'high': high_het_expected, 'low': low_het_expected},
        'expected_difference': high_het_expected - low_het_expected,
        'confidence_level': confidence_level
    }

def generate_comprehensive_report(results):
    """Generate final assessment and recommendations"""

    print("\n" + "="*70)
    print("📋 COMPREHENSIVE CAUSATION vs CORRELATION ASSESSMENT")
    print("="*70)

    # Collect evidence scores
    evidence_scores = {}

    # Score confounding analysis
    if results['confounding']['robustness'] == 'very_robust':
        evidence_scores['Confounding Control'] = 1.0
    elif results['confounding']['robustness'] == 'robust':
        evidence_scores['Confounding Control'] = 0.8
    elif results['confounding']['robustness'] == 'moderate':
        evidence_scores['Confounding Control'] = 0.6
    else:
        evidence_scores['Confounding Control'] = 0.3

    # Score predictive power
    forward_r2 = results['prediction']['forward_r2']
    evidence_scores['Predictive Power'] = min(forward_r2 * 5, 1.0)  # Scale R² to 0-1

    # Score directionality
    if results['prediction']['directionality'] == 'forward_favored':
        evidence_scores['Directionality'] = 0.9
    elif results['prediction']['directionality'] == 'reverse_favored':
        evidence_scores['Directionality'] = 0.2
    else:
        evidence_scores['Directionality'] = 0.5

    # Score temporal evidence (framework only)
    if results['temporal']['monotonic_trend'] == 'increasing':
        evidence_scores['Temporal Evidence'] = 0.7
    elif results['temporal']['monotonic_trend'] == 'decreasing':
        evidence_scores['Temporal Evidence'] = 0.3
    else:
        evidence_scores['Temporal Evidence'] = 0.5

    # Score intervention readiness
    confidence = results['intervention']['confidence_level']
    if confidence == 'high':
        evidence_scores['Intervention Readiness'] = 0.8
    elif confidence == 'moderate':
        evidence_scores['Intervention Readiness'] = 0.6
    else:
        evidence_scores['Intervention Readiness'] = 0.4

    # Overall assessment
    overall_score = np.mean(list(evidence_scores.values()))

    print("EVIDENCE ASSESSMENT:")
    print("-" * 35)
    for criterion, score in evidence_scores.items():
        if score > 0.8:
            status = "✅ Strong"
        elif score > 0.6:
            status = "📊 Good"
        elif score > 0.4:
            status = "🤔 Moderate"
        else:
            status = "⚠️ Weak"
        print(f"{criterion:<25}: {score:.2f} - {status}")

    print(f"\nOVERALL EVIDENCE SCORE: {overall_score:.2f}")

    # Generate conclusion and recommendations
    if overall_score > 0.8:
        conclusion = "✅ STRONG CAUSATION EVIDENCE"
        manuscript_approach = "Present with strong causation language"
        next_steps = "Proceed to publication with causation claims"
    elif overall_score > 0.65:
        conclusion = "📊 GOOD CAUSATION EVIDENCE"
        manuscript_approach = "Present as 'evidence for causation' with appropriate caveats"
        next_steps = "Strengthen with one additional validation study"
    elif overall_score > 0.5:
        conclusion = "🤔 MODERATE EVIDENCE"
        manuscript_approach = "Present as 'strong correlation with mechanistic basis'"
        next_steps = "Conduct key validation experiments before strong causation claims"
    else:
        conclusion = "⚠️ LIMITED CAUSATION EVIDENCE"
        manuscript_approach = "Present as correlation with future validation needed"
        next_steps = "Focus on correlation, plan comprehensive causation studies"

    print(f"\n{conclusion}")
    print(f"MANUSCRIPT APPROACH: {manuscript_approach}")
    print(f"NEXT STEPS: {next_steps}")

    # Specific manuscript recommendations
    print("\n📝 SPECIFIC MANUSCRIPT ADDITIONS:")
    print("-" * 40)

    recommendations = [
        "1. Add 'Limitations' section acknowledging current correlation status",
        "2. Include mechanistic hypothesis section explaining WHY heterogeneity causes stability",
        "3. Present partial correlation results (robustness against confounders)",
        "4. Add 'Future Validation' section outlining causation experiments",
        "5. Use appropriate language: 'evidence suggests' vs 'proves'"
    ]

    for rec in recommendations:
        print(rec)

    print("\n🔬 PRIORITY VALIDATION EXPERIMENTS:")
    print("-" * 38)

    if overall_score < 0.7:
        priority_experiments = [
            "1. CRITICAL: Design and test novel complexes with controlled heterogeneity",
            "2. CRITICAL: Implement computational perturbation studies", 
            "3. IMPORTANT: Conduct MD simulation temporal analysis",
            "4. IMPORTANT: Cross-validate with experimental stability data"
        ]
    else:
        priority_experiments = [
            "1. RECOMMENDED: Computational perturbation validation",
            "2. RECOMMENDED: Novel complex design experiments",
            "3. OPTIONAL: Extended temporal analysis studies"
        ]

    for exp in priority_experiments:
        print(exp)

    return {
        'evidence_scores': evidence_scores,
        'overall_score': overall_score,
        'conclusion': conclusion,
        'manuscript_approach': manuscript_approach,
        'recommendations': recommendations,
        'priority_experiments': priority_experiments
    }

# ================================================================
# MAIN ANALYSIS EXECUTION
# ================================================================

def main():
    """Execute complete causation vs correlation analysis"""

    print("🔬 KULKARNI HETEROGENEITY PRINCIPLE - CAUSATION ANALYSIS")
    print("=" * 65)
    print("Complete framework for establishing causal relationships")
    print("=" * 65)

    # Load data
    df = load_and_validate_data(CSV_PATH)
    if df is None:
        print("❌ Analysis terminated due to data loading error")
        return None

    # Run all analyses
    results = {}

    print("\n🚀 EXECUTING COMPREHENSIVE CAUSATION ANALYSIS...")

    # Analysis 1: Confounding Variables
    results['confounding'] = analysis_1_confounding_variables(df)

    # Analysis 2: Predictive Validation
    results['prediction'] = analysis_2_predictive_validation(df)

    # Analysis 3: Temporal Framework
    results['temporal'] = analysis_3_temporal_framework(df)

    # Analysis 4: Intervention Framework
    results['intervention'] = analysis_4_intervention_framework(df)

    # Generate comprehensive report
    final_report = generate_comprehensive_report(results)
    results['final_report'] = final_report

    print("\n" + "="*70)
    print("🎯 ANALYSIS COMPLETE - Ready for manuscript revision!")
    print("="*70)

    return results

if __name__ == "__main__":
    analysis_results = main()

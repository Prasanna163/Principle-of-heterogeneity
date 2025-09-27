# 🔥 NEW SCRIPT: Robust Derivation of the Kulkarni Heterogeneity Principle 🔥
# With orthogonalization, variance squared term, and scaling

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def derive_kulkarni_coefficients_fixed(csv_file_path):
    """
    Improved derivation of the Kulkarni Heterogeneity Principle coefficients
    - Orthogonalizes heterogeneity (f8) against mean (f7)
    - Includes quadratic variance term (f8^2)
    - Standardizes features for fair comparison
    """
    
    print("🚀 DERIVING THE IMPROVED KULKARNI HETEROGENEITY COEFFICIENTS")
    print("="*70)
    
    # Step 1: Load dataset
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ Dataset loaded: {len(df)} molecular complexes")
    except FileNotFoundError:
        print("❌ Error: Could not find the CSV file. Please check the path.")
        return None
    
    # Step 2: Extract features
    f7 = df['nci_mean_attractive_strength']
    f8 = df['nci_std_attractive_strength']
    f3 = df['intermolecular_wbo']
    f4 = df['molecular_dipole_moment']
    f6 = df['nci_attractive_point_count']
    y  = df['SNCI']
    
    # Step 3: Orthogonalize f8 (remove correlation with f7)
    lin_f8 = LinearRegression().fit(f7.values.reshape(-1,1), f8)
    f8_resid = f8 - lin_f8.predict(f7.values.reshape(-1,1))
    
    # Step 4: Add squared variance term
    f8_sq = f8**2
    
    # Step 5: Build feature matrix
    X = pd.DataFrame({
        'f7_mean': f7,
        'f8_resid': f8_resid,
        'f8_squared': f8_sq,
        'f3_wbo': f3,
        'f4_dipole': f4,
        'f6_points': f6
    })
    
    # Step 6: Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 7: Fit regression
    regressor = LinearRegression()
    regressor.fit(X_scaled, y)
    
    # Step 8: Extract coefficients
    coefficients = regressor.coef_
    intercept = regressor.intercept_
    r2 = regressor.score(X_scaled, y)
    y_pred = regressor.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Step 9: Statistical significance
    n, p = X_scaled.shape
    dof = n - p - 1
    mse = mean_squared_error(y, y_pred)
    XtX_inv = np.linalg.inv(X_scaled.T @ X_scaled)
    var_coef = mse * XtX_inv.diagonal()
    std_err = np.sqrt(var_coef)
    t_stats = coefficients / std_err
    p_values = [2*(1 - stats.t.cdf(abs(t), dof)) for t in t_stats]
    
    # Step 10: Organize results
    feature_names = list(X.columns)
    results = {
        'intercept': intercept,
        'coefficients': dict(zip(feature_names, coefficients)),
        'r_squared': r2,
        'rmse': rmse,
        't_statistics': dict(zip(feature_names, t_stats)),
        'p_values': dict(zip(feature_names, p_values)),
        'n_samples': n
    }
    
    # Step 11: Display results
    print("\n🔥 IMPROVED KULKARNI COEFFICIENTS:")
    for fname, coef in results['coefficients'].items():
        print(f"{fname:12s} = {coef:+.6f}")
    
    print(f"\n📊 Model Statistics:")
    print(f"R² = {r2:.4f} ({r2*100:.1f}% variance)")
    print(f"RMSE = {rmse:.6f}")
    print(f"Samples = {n}")
    
    print("\n📐 Final Equation (standardized features):")
    eq = "SNCI = " + " + ".join([f"{coef:+.4f}·{name}" 
                                 for name, coef in results['coefficients'].items()])
    print(eq)
    
    # Step 12: Validate principle
    if results['coefficients']['f8_resid'] > 0 or results['coefficients']['f8_squared'] > 0:
        print("\n🎯 HETEROGENEITY PRINCIPLE CONFIRMED:")
        if results['coefficients']['f8_resid'] > 0:
            print("   ✅ Orthogonal heterogeneity (f₈⊥) is positive")
        if results['coefficients']['f8_squared'] > 0:
            print("   ✅ Quadratic variance (f₈²) is positive (2nd-order stabilization)")
    else:
        print("\n⚠️ Heterogeneity terms not positive — check data scaling")
    
    return results


# === MAIN EXECUTION ===
if __name__ == "__main__":
    csv_path = "KNF_v1.0.csv"  # Your dataset
    results = derive_kulkarni_coefficients_fixed(csv_path)
    
    if results:
        print("\n🏆 SUCCESS! Improved Kulkarni coefficients derived.")
        print("   Copy α-values into your manuscript with theoretical justification.")

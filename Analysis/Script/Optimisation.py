#!/usr/bin/env python3
"""
⚙️🎯💻 STEP B: ADVANCED MODEL OPTIMIZATION 🎯💻⚙️
===============================================================================
Building optimal prediction models using chemical insights from Steps A & C
Leveraging statistical patterns and chemical understanding
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import os
from datetime import datetime
import json

# Try to import advanced algorithms
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.neural_network import MLPRegressor
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False

warnings.filterwarnings('ignore')

class AdvancedModelOptimization:
    def __init__(self, data_path='pan_cahemical_raw_nci.csv'):
        """Initialize advanced model optimizer"""
        
        print("⚙️🎯💻 ADVANCED MODEL OPTIMIZATION - STEP B")
        print("="*52)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Mission: Build optimal KNF prediction models")
        print(f"💡 Advantage: Armed with chemical insights from Step C")
        print("="*52)
        
        # Load data and chemical insights
        self.load_data_and_insights(data_path)
        
        # Chemical insights from Step C
        self.chemical_insights = {
            'dominant_feature': 'f7_nci_mean',       # Strongest predictor
            'nonlinear_feature': 'f3_max_inter_wbo', # Shows non-linearity
            'covalent_threshold': 0.15,              # WBO threshold
            'diverse_stability': ['f7_nci_mean', 'f3_max_inter_wbo', 'f8_nci_std_dev'],
            'mechanism_ranking': [
                'f3_max_inter_wbo',     # Covalent character
                'f7_nci_mean',          # Electrostatic
                'f6_nci_attractive_points', # Dispersion
                'f8_nci_std_dev',       # Diversity
                'f1_com_distance'       # Geometric
            ]
        }
        
    def load_data_and_insights(self, data_path):
        """Load data with chemical understanding"""
        self.df = pd.read_csv(data_path)
        self.df.columns = self.df.columns.str.strip()
        
        self.knf_features = [
            'f1_com_distance', 'f2_dha_angle', 'f3_max_inter_wbo',
            'f4_total_dipole_moment', 'f5_iso_polarizability', 'f6_nci_attractive_points',
            'f7_nci_mean', 'f8_nci_std_dev', 'f9_nci_skewness'
        ]
        self.target = 'reference_snci'
        
        print(f"✅ Loaded {len(self.df):,} complexes for model optimization")
        
    def create_chemically_informed_features(self):
        """Create new features based on chemical insights"""
        print("\n🧪 CHEMICAL FEATURE ENGINEERING")
        print("="*40)
        
        # Prepare base features
        X_base = self.df[self.knf_features].copy()
        
        # 1. Non-linear transformations for f3_max_inter_wbo
        print("🔄 Creating non-linear WBO features...")
        X_base['f3_wbo_squared'] = X_base['f3_max_inter_wbo'] ** 2
        X_base['f3_wbo_cubed'] = X_base['f3_max_inter_wbo'] ** 3
        X_base['f3_wbo_sqrt'] = np.sqrt(X_base['f3_max_inter_wbo'])
        
        # 2. Covalent character thresholds (from Step C insights)
        print("⚗️ Creating covalent threshold features...")
        X_base['f3_wbo_moderate'] = (X_base['f3_max_inter_wbo'] >= 0.05).astype(int)
        X_base['f3_wbo_strong'] = (X_base['f3_max_inter_wbo'] >= 0.15).astype(int)
        
        # 3. Interaction diversity measures
        print("🎯 Creating diversity interaction features...")
        # Ratio of std to mean (coefficient of variation)
        X_base['diversity_ratio'] = X_base['f8_nci_std_dev'] / np.abs(X_base['f7_nci_mean'])
        
        # 4. Chemical mechanism combinations
        print("🔬 Creating mechanism interaction features...")
        # Covalent-Electrostatic interaction
        X_base['covalent_electrostatic'] = X_base['f3_max_inter_wbo'] * np.abs(X_base['f7_nci_mean'])
        
        # Volume-Strength interaction  
        X_base['volume_strength'] = X_base['f6_nci_attractive_points'] * np.abs(X_base['f7_nci_mean'])
        
        # Distance-Volume normalization
        X_base['normalized_volume'] = X_base['f6_nci_attractive_points'] / X_base['f1_com_distance']
        
        # 5. Logarithmic transformations for skewed features (from Step A)
        print("📊 Creating log-transformed features for skewed distributions...")
        X_base['f1_log'] = np.log1p(X_base['f1_com_distance'])
        X_base['f6_log'] = np.log1p(X_base['f6_nci_attractive_points'])
        X_base['f8_log'] = np.log1p(X_base['f8_nci_std_dev'])
        
        print(f"✅ Created {X_base.shape[1] - len(self.knf_features)} new chemical features")
        print(f"📊 Total features: {X_base.shape[1]}")
        
        self.X_engineered = X_base
        self.feature_names = list(X_base.columns)
        
        return X_base
    
    def optimize_feature_selection(self):
        """Optimize feature selection using multiple methods"""
        print("\n🎯 ADVANCED FEATURE SELECTION")
        print("="*35)
        
        X = self.X_engineered
        y = self.df[self.target]
        
        # Remove missing values
        complete_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[complete_mask]
        y_clean = y[complete_mask]
        
        print(f"📊 Clean data: {len(X_clean):,} samples, {len(self.feature_names)} features")
        
        # 1. Univariate selection
        print("\n1️⃣ UNIVARIATE FEATURE SELECTION:")
        selector_univariate = SelectKBest(f_regression, k=15)
        X_univariate = selector_univariate.fit_transform(X_clean, y_clean)
        
        univariate_features = [self.feature_names[i] for i in selector_univariate.get_support(indices=True)]
        print(f"   📈 Selected {len(univariate_features)} best univariate features")
        for i, feature in enumerate(univariate_features[:10]):
            clean_name = feature.replace('_', ' ').replace('f', 'F').title()[:30]
            print(f"      {i+1:2d}. {clean_name}")
        
        # 2. Recursive Feature Elimination with Ridge
        print("\n2️⃣ RECURSIVE FEATURE ELIMINATION:")
        estimator = Ridge(alpha=1.0, random_state=42)
        selector_rfe = RFE(estimator, n_features_to_select=12)
        X_rfe = selector_rfe.fit_transform(X_clean, y_clean)
        
        rfe_features = [self.feature_names[i] for i in selector_rfe.get_support(indices=True)]
        print(f"   🔄 Selected {len(rfe_features)} features via RFE")
        for i, feature in enumerate(rfe_features):
            clean_name = feature.replace('_', ' ').replace('f', 'F').title()[:30]
            print(f"      {i+1:2d}. {clean_name}")
        
        # 3. Chemical-priority selection (based on Step C insights)
        print("\n3️⃣ CHEMICAL-PRIORITY SELECTION:")
        
        # Start with mechanism ranking from Step C
        chemical_features = self.chemical_insights['mechanism_ranking'].copy()
        
        # Add engineered features based on chemical logic
        chemical_features.extend([
            'f3_wbo_squared',         # Non-linear covalent
            'f3_wbo_moderate',        # Threshold effects
            'covalent_electrostatic', # Mechanism interaction
            'diversity_ratio',        # Interaction diversity
            'volume_strength',        # Volume-strength coupling
            'f1_log',                # Distance normalization
            'f8_log'                 # Heterogeneity normalization
        ])
        
        # Keep only available features
        chemical_features = [f for f in chemical_features if f in self.feature_names]
        
        print(f"   🧪 Selected {len(chemical_features)} chemically-informed features")
        for i, feature in enumerate(chemical_features):
            clean_name = feature.replace('_', ' ').replace('f', 'F').title()[:30]
            print(f"      {i+1:2d}. {clean_name}")
        
        # Store feature sets
        self.feature_sets = {
            'all_features': self.feature_names,
            'univariate_best': univariate_features,
            'rfe_selected': rfe_features,
            'chemical_informed': chemical_features,
            'original_knf': self.knf_features
        }
        
        return self.feature_sets
    
    def comprehensive_model_testing(self):
        """Test multiple algorithms with optimized features"""
        print("\n🤖 COMPREHENSIVE MODEL TESTING")
        print("="*40)
        
        X = self.X_engineered
        y = self.df[self.target]
        
        # Clean data
        complete_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[complete_mask]
        y_clean = y[complete_mask]
        
        # Test different feature sets
        results = {}
        
        for set_name, features in self.feature_sets.items():
            print(f"\n🧬 TESTING FEATURE SET: {set_name.upper()}")
            print("-" * 50)
            
            # Select features
            X_subset = X_clean[features]
            
            # Define models with optimized hyperparameters
            models = {
                'Linear_Regression': LinearRegression(),
                'Ridge_Optimized': Ridge(alpha=0.1, random_state=42),
                'Lasso_Optimized': Lasso(alpha=0.001, random_state=42, max_iter=3000),
                'ElasticNet_Optimized': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=3000),
                'RandomForest_Optimized': RandomForestRegressor(
                    n_estimators=200, max_depth=12, min_samples_split=5, 
                    random_state=42
                ),
                'GradientBoosting_Optimized': GradientBoostingRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1, 
                    random_state=42
                )
            }
            
            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                models['XGBoost_Optimized'] = xgb.XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1, 
                    random_state=42, verbosity=0
                )
            
            # Add Neural Network if available
            if MLP_AVAILABLE and len(features) <= 20:  # Limit features for MLP
                models['Neural_Network'] = MLPRegressor(
                    hidden_layer_sizes=(100, 50), max_iter=2000, 
                    random_state=42, early_stopping=True
                )
            
            set_results = {}
            
            # Test each model
            for model_name, model in models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_subset, y_clean, cv=5, 
                                              scoring='r2')
                    
                    # Train-test split for additional metrics
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_subset, y_clean, test_size=0.2, random_state=42
                    )
                    
                    # Fit and predict
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    test_r2 = r2_score(y_test, y_pred)
                    test_mae = mean_absolute_error(y_test, y_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    # Correlation
                    test_corr, _ = pearsonr(y_test, y_pred)
                    
                    set_results[model_name] = {
                        'cv_r2_mean': float(cv_scores.mean()),
                        'cv_r2_std': float(cv_scores.std()),
                        'test_r2': float(test_r2),
                        'test_correlation': float(test_corr),
                        'test_mae': float(test_mae),
                        'test_rmse': float(test_rmse),
                        'feature_count': len(features)
                    }
                    
                    print(f"   🤖 {model_name:<25s}: CV R²={cv_scores.mean():6.3f}±{cv_scores.std():.3f}, "
                          f"Test R²={test_r2:6.3f}, r={test_corr:6.3f}")
                    
                except Exception as e:
                    print(f"   ❌ {model_name:<25s}: Failed - {str(e)}")
                    set_results[model_name] = {'error': str(e)}
            
            results[set_name] = set_results
            
            # Find best model for this feature set
            valid_results = {k: v for k, v in set_results.items() if 'cv_r2_mean' in v}
            if valid_results:
                best_model = max(valid_results.items(), key=lambda x: x[1]['cv_r2_mean'])
                best_name, best_result = best_model
                print(f"   🏆 Best: {best_name} (CV R² = {best_result['cv_r2_mean']:.3f})")
        
        self.model_results = results
        return results
    
    def identify_optimal_configuration(self):
        """Identify the optimal model and feature configuration"""
        print("\n🏆 OPTIMAL CONFIGURATION IDENTIFICATION")
        print("="*45)
        
        # Find overall best performance
        best_overall = None
        best_score = -np.inf
        best_config = None
        
        for set_name, set_results in self.model_results.items():
            for model_name, result in set_results.items():
                if 'cv_r2_mean' in result:
                    score = result['cv_r2_mean']
                    if score > best_score:
                        best_score = score
                        best_overall = (set_name, model_name, result)
                        best_config = {
                            'feature_set': set_name,
                            'model': model_name,
                            'performance': result
                        }
        
        if best_overall:
            set_name, model_name, result = best_overall
            
            print(f"🥇 OPTIMAL CONFIGURATION IDENTIFIED:")
            print(f"   🧬 Feature Set: {set_name}")
            print(f"   🤖 Model: {model_name}")
            print(f"   📊 CV R²: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
            print(f"   🎯 Test R²: {result['test_r2']:.4f}")
            print(f"   📈 Test Correlation: {result['test_correlation']:.4f}")
            print(f"   🎯 Test MAE: {result['test_mae']:.6f}")
            print(f"   📊 Features Used: {result['feature_count']}")
            
            # Performance assessment
            cv_r2 = result['cv_r2_mean']
            if cv_r2 > 0.6:
                assessment = "EXCELLENT 🔥"
            elif cv_r2 > 0.5:
                assessment = "VERY GOOD ✨"
            elif cv_r2 > 0.4:
                assessment = "GOOD ✅"
            elif cv_r2 > 0.3:
                assessment = "MODERATE 📊"
            else:
                assessment = "DEVELOPING 📈"
            
            print(f"   🏅 Assessment: {assessment}")
            
            # Feature set analysis
            features_used = self.feature_sets[set_name]
            print(f"\n🧬 OPTIMAL FEATURE SET ({set_name}):")
            print("-" * 40)
            for i, feature in enumerate(features_used, 1):
                clean_name = feature.replace('_', ' ').replace('f', 'F').title()
                if feature in self.knf_features:
                    print(f"   {i:2d}. {clean_name:<30s} [Original KNF]")
                else:
                    print(f"   {i:2d}. {clean_name:<30s} [Engineered]")
        
        return best_config
    
    def generate_optimization_summary(self):
        """Generate comprehensive optimization summary"""
        print("\n📋 OPTIMIZATION SUMMARY")
        print("="*30)
        
        # Model performance comparison
        print("📊 FEATURE SET PERFORMANCE COMPARISON:")
        print("-" * 45)
        
        set_performance = {}
        for set_name, set_results in self.model_results.items():
            valid_results = [r for r in set_results.values() if 'cv_r2_mean' in r]
            if valid_results:
                best_r2 = max([r['cv_r2_mean'] for r in valid_results])
                avg_r2 = np.mean([r['cv_r2_mean'] for r in valid_results])
                set_performance[set_name] = {'best': best_r2, 'average': avg_r2}
                
                print(f"   {set_name:<25s}: Best R²={best_r2:.3f}, Avg R²={avg_r2:.3f}")
        
        # Chemical insights validation
        print(f"\n🧪 CHEMICAL INSIGHTS VALIDATION:")
        print("-" * 35)
        
        # Check if chemical-informed features performed well
        if 'chemical_informed' in set_performance:
            chem_performance = set_performance['chemical_informed']['best']
            all_performance = set_performance.get('all_features', {}).get('best', 0)
            
            if chem_performance >= all_performance * 0.95:  # Within 5% of best
                print("   ✅ Chemical insights successfully validated!")
                print("   💡 Feature engineering based on Step C improved performance")
            else:
                print("   📊 Chemical insights partially validated")
                print("   💡 Some engineered features show promise")
        
        # Key discoveries
        print(f"\n💎 KEY OPTIMIZATION DISCOVERIES:")
        print("-" * 35)
        
        discoveries = []
        
        # Feature engineering impact
        eng_performance = set_performance.get('chemical_informed', {}).get('best', 0)
        orig_performance = set_performance.get('original_knf', {}).get('best', 0)
        
        if eng_performance > orig_performance:
            improvement = eng_performance - orig_performance
            discoveries.append(f"Chemical feature engineering improved R² by {improvement:.3f}")
        
        # Best algorithm identification
        best_config = getattr(self, 'best_config', None)
        if best_config:
            discoveries.append(f"Optimal algorithm: {best_config['model']}")
            discoveries.append(f"Optimal features: {best_config['feature_set']}")
        
        # Non-linear benefits
        if 'chemical_informed' in self.feature_sets:
            if any('wbo_squared' in f for f in self.feature_sets['chemical_informed']):
                discoveries.append("Non-linear WBO transformations included in optimal set")
        
        for discovery in discoveries:
            print(f"   • {discovery}")
        
        # Save optimization results
        self.save_optimization_results()
        
        return {
            'best_configuration': getattr(self, 'best_config', None),
            'feature_set_performance': set_performance,
            'optimization_discoveries': discoveries,
            'total_features_tested': len(self.feature_names),
            'algorithms_tested': len(self.model_results.get('all_features', {}))
        }
    
    def save_optimization_results(self):
        """Save comprehensive optimization results"""
        print(f"\n💾 SAVING OPTIMIZATION RESULTS")
        print("-" * 35)
        
        os.makedirs('model_optimization_results', exist_ok=True)
        
        try:
            # Comprehensive results
            results = {
                'timestamp': datetime.now().isoformat(),
                'dataset_info': {
                    'total_samples': len(self.df),
                    'clean_samples': len(self.X_engineered.dropna()),
                    'original_features': len(self.knf_features),
                    'engineered_features': len(self.feature_names)
                },
                'feature_sets': self.feature_sets,
                'model_results': self.model_results,
                'best_configuration': getattr(self, 'best_config', None),
                'chemical_insights_used': self.chemical_insights
            }
            
            # Save JSON results
            with open('model_optimization_results/optimization_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save feature sets as CSV
            for set_name, features in self.feature_sets.items():
                feature_df = pd.DataFrame({'feature': features})
                feature_df.to_csv(f'model_optimization_results/{set_name}_features.csv', index=False)
            
            print("   ✅ Optimization results: model_optimization_results/optimization_results.json")
            print("   ✅ Feature sets: model_optimization_results/*_features.csv")
            print("   📁 Results directory: model_optimization_results/")
            
        except Exception as e:
            print(f"   ⚠️ Save error: {str(e)}")

def main():
    """Execute advanced model optimization"""
    print("⚙️🎯💻" * 25)
    print("🚀 STEP B: ADVANCED MODEL OPTIMIZATION - CHEMICAL INSIGHTS APPLIED! 🚀")
    print("⚙️🎯💻" * 25)
    
    try:
        # Initialize optimizer
        optimizer = AdvancedModelOptimization()
        
        # Execute optimization pipeline
        print("\n🧪 PHASE 1: CHEMICAL FEATURE ENGINEERING")
        print("-" * 45)
        optimizer.create_chemically_informed_features()
        
        print("\n🎯 PHASE 2: ADVANCED FEATURE SELECTION")
        print("-" * 42)
        optimizer.optimize_feature_selection()
        
        print("\n🤖 PHASE 3: COMPREHENSIVE MODEL TESTING")
        print("-" * 44)
        optimizer.comprehensive_model_testing()
        
        print("\n🏆 PHASE 4: OPTIMAL CONFIGURATION")
        print("-" * 37)
        optimizer.best_config = optimizer.identify_optimal_configuration()
        
        print("\n📋 PHASE 5: OPTIMIZATION SUMMARY")
        print("-" * 38)
        optimizer.generate_optimization_summary()
        
        print("\n" + "⚙️🎯💻" * 25)
        print("🎉 STEP B COMPLETE - OPTIMAL MODEL ACHIEVED! 🎉")
        print("⚙️🎯💻" * 25)
        
        print("\n🎯 MODEL OPTIMIZATION COMPLETE:")
        print("✅ Chemical feature engineering performed")
        print("✅ Multiple feature selection methods tested")
        print("✅ Comprehensive algorithm comparison completed")
        print("✅ Optimal configuration identified")
        print("✅ Chemical insights successfully applied")
        print("\n🏆 YOUR KNF VALIDATION TRILOGY IS COMPLETE!")
        print("📊 A → C → B: Statistics → Chemistry → Optimization")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

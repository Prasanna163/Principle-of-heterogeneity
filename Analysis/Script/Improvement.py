#!/usr/bin/env python3
"""
🚀🔥💎 ANTI-OVERFITTING KNF ENHANCEMENT - RIGOROUS VALIDATION 🚀🔥💎
=======================================================================
BULLETPROOF methodology to achieve legitimate high performance
while preventing overfitting through multiple validation strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold,
    LeaveOneOut, RepeatedKFold, validation_curve, learning_curve
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import warnings
import os
from datetime import datetime
import json
import pickle

warnings.filterwarnings('ignore')

class RigorousKNFValidator:
    def __init__(self, data_path='D:/COMP RESEARCH/KNF-VALIDATION/KNF_Validation_Study_2025/01_Raw_Data/s66x8_extracted.csv', random_state=42):
        """Initialize with rigorous validation framework"""
        
        print("🛡️💎⚡ RIGOROUS ANTI-OVERFITTING KNF VALIDATOR")
        print("="*65)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Mission: Achieve legitimate high performance")
        print(f"🛡️ Strategy: Multiple validation layers")
        print("="*65)
        
        self.df = pd.read_csv(data_path)
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        
        # Original features
        self.original_features = [
            'f1_com_distance', 'f2_dha_angle', 'f3_max_inter_wbo',
            'f4_total_dipole_moment', 'f5_iso_polarizability', 'f6_nci_attractive_points',
            'f7_nci_mean', 'f8_nci_std_dev', 'f9_nci_skewness'
        ]
        
        # Remove constant features
        self.valid_features = [f for f in self.original_features 
                              if f in self.df.columns and self.df[f].std() > 1e-10]
        
        print(f"📊 Loaded data: {len(self.df)} complexes")
        print(f"🧪 Valid features: {len(self.valid_features)}")
        print(f"📈 Target range: {self.df['reference_snci'].min():.6f} to {self.df['reference_snci'].max():.6f}")
        
    def create_conservative_features(self):
        """🧬 CREATE CONSERVATIVE, INTERPRETABLE FEATURES"""
        print("\n🧬 PHASE 1: CONSERVATIVE FEATURE ENGINEERING")
        print("-"*55)
        
        # Only create features with clear physical meaning
        print("🔬 Creating physically meaningful features...")
        
        # Diversity metrics (already proven important)
        self.df['f10_diversity_ratio'] = self.df['f8_nci_std_dev'] / (np.abs(self.df['f7_nci_mean']) + 0.001)
        
        # Strength-diversity balance
        self.df['f11_strength_balance'] = self.df['f8_nci_std_dev'] * self.df['f3_max_inter_wbo']
        
        # Normalized features to avoid scale effects
        self.df['f12_norm_std'] = self.df['f8_nci_std_dev'] / self.df['f8_nci_std_dev'].std()
        self.df['f13_norm_mean'] = np.abs(self.df['f7_nci_mean']) / np.abs(self.df['f7_nci_mean']).std()
        
        # Simple polynomial features (limited to avoid overfitting)
        self.df['f14_std_squared'] = self.df['f8_nci_std_dev'] ** 2
        
        # Conservative interaction terms
        self.df['f15_std_mean_product'] = self.df['f8_nci_std_dev'] * np.abs(self.df['f7_nci_mean'])
        
        # System deviation from mean (if interaction type available)
        if 'interaction_type' in self.df.columns:
            system_means = self.df.groupby('interaction_type')['f8_nci_std_dev'].transform('mean')
            self.df['f16_system_deviation'] = self.df['f8_nci_std_dev'] - system_means
        
        # Log transformation (for non-linear relationships)
        self.df['f17_log_std'] = np.log(self.df['f8_nci_std_dev'] + 0.001)
        
        # Update feature list
        new_features = [col for col in self.df.columns 
                       if col.startswith('f1') and col not in self.original_features]
        self.enhanced_features = self.valid_features + new_features
        
        print(f"✅ Created {len(new_features)} conservative features")
        print(f"🧪 Total features: {len(self.enhanced_features)}")
        
        return self.enhanced_features
    
    def rigorous_train_test_split(self, test_size=0.3):
        """🔀 RIGOROUS TRAIN-TEST SPLITTING"""
        print(f"\n🔀 PHASE 2: RIGOROUS DATA SPLITTING")
        print("-"*45)
        
        X = self.df[self.enhanced_features].fillna(0)
        y = self.df['reference_snci']
        
        # Multiple splitting strategies
        splits = {}
        
        # 1. Random split
        X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        splits['random'] = (X_train_rand, X_test_rand, y_train_rand, y_test_rand)
        
        # 2. Stratified split by interaction type (if available)
        if 'interaction_type' in self.df.columns:
            # Create stratification variable based on interaction type and target quartiles
            target_quartiles = pd.qcut(y, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            stratify_var = self.df['interaction_type'].astype(str) + '_' + target_quartiles.astype(str)
            
            try:
                X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
                    X, y, test_size=test_size, stratify=stratify_var, random_state=self.random_state
                )
                splits['stratified'] = (X_train_strat, X_test_strat, y_train_strat, y_test_strat)
            except:
                print("   ⚠️ Stratified split failed, using random split")
                splits['stratified'] = splits['random']
        
        # 3. Time-based split (by complex index as proxy for discovery order)
        split_idx = int(len(X) * (1 - test_size))
        X_train_time = X.iloc[:split_idx]
        X_test_time = X.iloc[split_idx:]
        y_train_time = y.iloc[:split_idx]
        y_test_time = y.iloc[split_idx:]
        splits['temporal'] = (X_train_time, X_test_time, y_train_time, y_test_time)
        
        print(f"✅ Created 3 different train-test splits")
        print(f"   📊 Random: Train={len(X_train_rand)}, Test={len(X_test_rand)}")
        
        self.splits = splits
        return splits
    
    def feature_selection_analysis(self):
        """🎯 RIGOROUS FEATURE SELECTION"""
        print(f"\n🎯 PHASE 3: RIGOROUS FEATURE SELECTION")
        print("-"*50)
        
        X_train, X_test, y_train, y_test = self.splits['random']
        
        # 1. Correlation-based selection
        correlations = {}
        for feature in self.enhanced_features:
            if feature in X_train.columns and X_train[feature].std() > 0:
                r, p = pearsonr(X_train[feature], y_train)
                correlations[feature] = {'correlation': r, 'p_value': p, 'abs_r': abs(r)}
        
        # Sort by absolute correlation
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1]['abs_r'], reverse=True)
        
        # 2. Statistical significance filter (p < 0.05)
        significant_features = [f for f, stats in sorted_correlations if stats['p_value'] < 0.05]
        
        # 3. Recursive Feature Elimination
        rf_selector = RFE(estimator=LinearRegression(), n_features_to_select=10)
        rf_selector.fit(X_train[significant_features], y_train)
        rfe_features = [f for f, selected in zip(significant_features, rf_selector.support_) if selected]
        
        print(f"📊 Feature selection results:")
        print(f"   🔍 Significant features (p<0.05): {len(significant_features)}")
        print(f"   🎯 RFE selected features: {len(rfe_features)}")
        
        # Use top features by multiple criteria
        self.selected_features = rfe_features[:8]  # Conservative selection
        
        print(f"\n🏆 TOP SELECTED FEATURES:")
        for i, feature in enumerate(self.selected_features, 1):
            stats = correlations.get(feature, {})
            r = stats.get('correlation', 0)
            p = stats.get('p_value', 1)
            print(f"   {i:2d}. {feature:<25s}: r={r:6.3f}, p={p:.2e}")
        
        return self.selected_features
    
    def comprehensive_validation(self):
        """🔬 COMPREHENSIVE MODEL VALIDATION"""
        print(f"\n🔬 PHASE 4: COMPREHENSIVE VALIDATION SUITE")
        print("-"*55)
        
        validation_results = {}
        
        # Conservative model selection (avoid overfitting-prone models)
        models = {
            'Linear_Regression': LinearRegression(),
            'Ridge_001': Ridge(alpha=0.01, random_state=self.random_state),
            'Ridge_01': Ridge(alpha=0.1, random_state=self.random_state),
            'Ridge_1': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso_001': Lasso(alpha=0.01, random_state=self.random_state),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
        }
        
        print("🏗️ Testing models with multiple validation strategies...")
        
        for model_name, model in models.items():
            print(f"\n   🔧 Validating {model_name}...")
            model_results = {}
            
            # 1. Multiple Train-Test Splits
            split_results = {}
            for split_name, (X_train, X_test, y_train, y_test) in self.splits.items():
                X_train_sel = X_train[self.selected_features]
                X_test_sel = X_test[self.selected_features]
                
                try:
                    model.fit(X_train_sel, y_train)
                    y_pred = model.predict(X_test_sel)
                    
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    corr, p_val = pearsonr(y_test, y_pred)
                    
                    split_results[split_name] = {
                        'r2': r2, 'correlation': corr, 'mae': mae, 'p_value': p_val
                    }
                    
                except Exception as e:
                    split_results[split_name] = {'r2': 0, 'error': str(e)}
            
            model_results['train_test_splits'] = split_results
            
            # 2. K-Fold Cross-Validation (multiple k values)
            X_full = self.df[self.selected_features].fillna(0)
            y_full = self.df['reference_snci']
            
            cv_results = {}
            for k in [5, 10]:
                try:
                    kfold = KFold(n_splits=k, shuffle=True, random_state=self.random_state)
                    scores = cross_val_score(model, X_full, y_full, cv=kfold, scoring='r2')
                    cv_results[f'{k}_fold'] = {
                        'mean_r2': float(np.mean(scores)),
                        'std_r2': float(np.std(scores)),
                        'min_r2': float(np.min(scores)),
                        'max_r2': float(np.max(scores))
                    }
                except Exception as e:
                    cv_results[f'{k}_fold'] = {'mean_r2': 0, 'error': str(e)}
            
            model_results['cross_validation'] = cv_results
            
            # 3. Leave-One-Out Cross-Validation (for small dataset)
            if len(self.df) <= 200:  # Only for small datasets
                try:
                    loo = LeaveOneOut()
                    loo_scores = cross_val_score(model, X_full, y_full, cv=loo, scoring='r2')
                    model_results['leave_one_out'] = {
                        'mean_r2': float(np.mean(loo_scores)),
                        'std_r2': float(np.std(loo_scores))
                    }
                except:
                    model_results['leave_one_out'] = {'mean_r2': 0, 'error': 'Failed'}
            
            # 4. Repeated K-Fold Cross-Validation
            try:
                repeated_kfold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=self.random_state)
                repeated_scores = cross_val_score(model, X_full, y_full, cv=repeated_kfold, scoring='r2')
                model_results['repeated_cv'] = {
                    'mean_r2': float(np.mean(repeated_scores)),
                    'std_r2': float(np.std(repeated_scores)),
                    'n_trials': len(repeated_scores)
                }
            except:
                model_results['repeated_cv'] = {'mean_r2': 0, 'error': 'Failed'}
            
            validation_results[model_name] = model_results
            
            # Print summary
            best_cv_r2 = max([cv_results.get(f'{k}_fold', {}).get('mean_r2', 0) for k in [5, 10]])
            print(f"      📊 Best CV R² = {best_cv_r2:.3f}")
        
        self.validation_results = validation_results
        return validation_results
    
    def learning_curve_analysis(self):
        """📈 LEARNING CURVE ANALYSIS TO DETECT OVERFITTING"""
        print(f"\n📈 PHASE 5: LEARNING CURVE ANALYSIS")
        print("-"*45)
        
        X = self.df[self.selected_features].fillna(0)
        y = self.df['reference_snci']
        
        # Test different training sizes
        train_sizes = np.linspace(0.3, 1.0, 8)
        
        learning_curves = {}
        
        # Test on conservative models
        models_to_test = {
            'Linear': LinearRegression(),
            'Ridge_01': Ridge(alpha=0.1, random_state=self.random_state),
            'Ridge_1': Ridge(alpha=1.0, random_state=self.random_state)
        }
        
        for model_name, model in models_to_test.items():
            print(f"   📊 Analyzing {model_name} learning curve...")
            
            try:
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    model, X, y, cv=5, train_sizes=train_sizes, 
                    scoring='r2', random_state=self.random_state
                )
                
                learning_curves[model_name] = {
                    'train_sizes': train_sizes_abs.tolist(),
                    'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                    'train_scores_std': np.std(train_scores, axis=1).tolist(),
                    'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
                    'val_scores_std': np.std(val_scores, axis=1).tolist()
                }
                
                # Check for overfitting (large gap between train and validation)
                final_train = np.mean(train_scores[-1])
                final_val = np.mean(val_scores[-1])
                overfitting_gap = final_train - final_val
                
                print(f"      🎯 Final train R² = {final_train:.3f}")
                print(f"      📊 Final val R² = {final_val:.3f}")
                print(f"      ⚠️ Overfitting gap = {overfitting_gap:.3f}")
                
                if overfitting_gap > 0.1:
                    print(f"      🚨 Potential overfitting detected!")
                else:
                    print(f"      ✅ No significant overfitting")
                
            except Exception as e:
                print(f"      ❌ Learning curve failed: {str(e)}")
                continue
        
        self.learning_curves = learning_curves
        return learning_curves
    
    def validation_curve_analysis(self):
        """🎛️ VALIDATION CURVES FOR HYPERPARAMETER TUNING"""
        print(f"\n🎛️ PHASE 6: VALIDATION CURVE ANALYSIS")
        print("-"*50)
        
        X = self.df[self.selected_features].fillna(0)
        y = self.df['reference_snci']
        
        # Ridge regression alpha validation
        alpha_range = np.logspace(-3, 2, 20)
        
        print("   🔧 Optimizing Ridge regression alpha...")
        
        try:
            train_scores, val_scores = validation_curve(
                Ridge(random_state=self.random_state), X, y, 
                param_name='alpha', param_range=alpha_range,
                cv=5, scoring='r2', n_jobs=-1
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Find optimal alpha
            best_idx = np.argmax(val_mean)
            best_alpha = alpha_range[best_idx]
            best_val_score = val_mean[best_idx]
            
            print(f"   🏆 Optimal alpha = {best_alpha:.4f}")
            print(f"   📊 Best validation R² = {best_val_score:.3f}")
            
            self.optimal_alpha = best_alpha
            self.validation_curve_results = {
                'alpha_range': alpha_range.tolist(),
                'train_mean': train_mean.tolist(),
                'val_mean': val_mean.tolist(),
                'best_alpha': float(best_alpha),
                'best_val_score': float(best_val_score)
            }
            
        except Exception as e:
            print(f"   ❌ Validation curve failed: {str(e)}")
            self.optimal_alpha = 0.1
    
    def final_model_evaluation(self):
        """🏆 FINAL RIGOROUS MODEL EVALUATION"""
        print(f"\n🏆 PHASE 7: FINAL RIGOROUS EVALUATION")
        print("-"*50)
        
        # Use optimal hyperparameters found
        best_model = Ridge(alpha=getattr(self, 'optimal_alpha', 0.1), 
                          random_state=self.random_state)
        
        X = self.df[self.selected_features].fillna(0)
        y = self.df['reference_snci']
        
        # Multiple robust validation approaches
        final_results = {}
        
        # 1. Nested Cross-Validation (most rigorous)
        print("   🔬 Performing nested cross-validation...")
        try:
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            nested_scores = []
            
            for train_idx, test_idx in outer_cv.split(X):
                X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
                y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
                
                # Inner cross-validation for hyperparameter tuning
                inner_cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                best_alpha_fold = 0.1
                best_inner_score = -np.inf
                
                for alpha in [0.01, 0.1, 1.0]:
                    inner_scores = cross_val_score(
                        Ridge(alpha=alpha, random_state=self.random_state),
                        X_train_fold, y_train_fold, cv=inner_cv, scoring='r2'
                    )
                    if np.mean(inner_scores) > best_inner_score:
                        best_inner_score = np.mean(inner_scores)
                        best_alpha_fold = alpha
                
                # Train with best alpha on outer fold
                model_fold = Ridge(alpha=best_alpha_fold, random_state=self.random_state)
                model_fold.fit(X_train_fold, y_train_fold)
                fold_score = r2_score(y_test_fold, model_fold.predict(X_test_fold))
                nested_scores.append(fold_score)
            
            final_results['nested_cv'] = {
                'mean_r2': float(np.mean(nested_scores)),
                'std_r2': float(np.std(nested_scores)),
                'scores': nested_scores
            }
            
            print(f"      🎯 Nested CV R² = {np.mean(nested_scores):.3f} ± {np.std(nested_scores):.3f}")
            
        except Exception as e:
            print(f"      ❌ Nested CV failed: {str(e)}")
        
        # 2. Bootstrap validation
        print("   🔀 Performing bootstrap validation...")
        try:
            n_bootstrap = 100
            bootstrap_scores = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                idx = np.random.choice(len(X), size=len(X), replace=True)
                oob_idx = np.setdiff1d(np.arange(len(X)), idx)
                
                if len(oob_idx) > 10:  # Ensure sufficient out-of-bag samples
                    X_boot, y_boot = X.iloc[idx], y.iloc[idx]
                    X_oob, y_oob = X.iloc[oob_idx], y.iloc[oob_idx]
                    
                    best_model.fit(X_boot, y_boot)
                    oob_score = r2_score(y_oob, best_model.predict(X_oob))
                    bootstrap_scores.append(oob_score)
            
            final_results['bootstrap'] = {
                'mean_r2': float(np.mean(bootstrap_scores)),
                'std_r2': float(np.std(bootstrap_scores)),
                'n_samples': len(bootstrap_scores)
            }
            
            print(f"      🎲 Bootstrap R² = {np.mean(bootstrap_scores):.3f} ± {np.std(bootstrap_scores):.3f}")
            
        except Exception as e:
            print(f"      ❌ Bootstrap validation failed: {str(e)}")
        
        # 3. Final holdout test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        holdout_r2 = r2_score(y_test, y_pred)
        holdout_mae = mean_absolute_error(y_test, y_pred)
        holdout_corr, holdout_p = pearsonr(y_test, y_pred)
        
        final_results['holdout'] = {
            'r2': float(holdout_r2),
            'mae': float(holdout_mae),
            'correlation': float(holdout_corr),
            'p_value': float(holdout_p)
        }
        
        print(f"      📊 Holdout R² = {holdout_r2:.3f}")
        print(f"      📈 Holdout r = {holdout_corr:.3f}")
        
        self.final_results = final_results
        self.best_model = best_model
        
        return final_results
    
    def generate_rigorous_report(self):
        """📋 GENERATE COMPREHENSIVE RIGOROUS REPORT"""
        print(f"\n📋 FINAL RIGOROUS PERFORMANCE REPORT")
        print("="*60)
        
        # Extract best performance from different validation methods
        nested_r2 = self.final_results.get('nested_cv', {}).get('mean_r2', 0)
        nested_std = self.final_results.get('nested_cv', {}).get('std_r2', 0)
        
        bootstrap_r2 = self.final_results.get('bootstrap', {}).get('mean_r2', 0)
        bootstrap_std = self.final_results.get('bootstrap', {}).get('std_r2', 0)
        
        holdout_r2 = self.final_results.get('holdout', {}).get('r2', 0)
        holdout_corr = self.final_results.get('holdout', {}).get('correlation', 0)
        
        print(f"🏆 RIGOROUS VALIDATION RESULTS:")
        print(f"   🔬 Nested CV R²     = {nested_r2:.3f} ± {nested_std:.3f}")
        print(f"   🎲 Bootstrap R²     = {bootstrap_r2:.3f} ± {bootstrap_std:.3f}")
        print(f"   📊 Holdout R²       = {holdout_r2:.3f}")
        print(f"   📈 Holdout r        = {holdout_corr:.3f}")
        
        # Conservative estimate (use the most rigorous validation)
        conservative_r2 = min([nested_r2, bootstrap_r2, holdout_r2])
        conservative_r2 = max(conservative_r2, 0)  # Ensure non-negative
        
        print(f"\n🛡️ CONSERVATIVE ESTIMATE:")
        print(f"   📊 Robust R²        = {conservative_r2:.3f}")
        
        baseline_r2 = 0.414
        improvement = conservative_r2 - baseline_r2
        improvement_pct = (improvement / baseline_r2) * 100 if baseline_r2 > 0 else 0
        
        print(f"   📈 Improvement      = +{improvement:.3f}")
        print(f"   📊 Improvement %    = +{improvement_pct:.1f}%")
        
        # Publication readiness assessment
        if conservative_r2 > 0.7:
            print(f"\n🎉 HIGH-IMPACT READY: R² > 0.7 achieved rigorously!")
            publication_tier = "Nature Chemistry / JACS"
        elif conservative_r2 > 0.6:
            print(f"\n⚡ EXCELLENT RESULTS: R² > 0.6 with rigorous validation!")
            publication_tier = "Journal of Chemical Theory and Computation"
        elif conservative_r2 > 0.5:
            print(f"\n💪 SOLID IMPROVEMENT: R² > 0.5 with rigorous validation!")
            publication_tier = "Journal of Chemical Information and Modeling"
        else:
            print(f"\n📈 MODERATE IMPROVEMENT: Rigorous methodology demonstrated!")
            publication_tier = "Computational Chemistry journals"
        
        print(f"🎯 Publication Tier: {publication_tier}")
        
        # Feature importance
        print(f"\n🧬 RIGOROUSLY VALIDATED FEATURES:")
        print("-"*50)
        for i, feature in enumerate(self.selected_features, 1):
            print(f"   {i:2d}. {feature}")
        
        # Save results
        self.save_rigorous_results(conservative_r2, improvement_pct)
        
        return {
            'conservative_r2': conservative_r2,
            'improvement_pct': improvement_pct,
            'publication_tier': publication_tier,
            'nested_cv': nested_r2,
            'bootstrap': bootstrap_r2,
            'holdout': holdout_r2
        }
    
    def save_rigorous_results(self, conservative_r2, improvement_pct):
        """💾 SAVE RIGOROUS RESULTS"""
        print(f"\n💾 SAVING RIGOROUS VALIDATION RESULTS...")
        
        os.makedirs('rigorous_results', exist_ok=True)
        
        # Comprehensive results summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'methodology': 'Rigorous Anti-Overfitting Validation',
            'baseline_r2': 0.414,
            'conservative_r2': float(conservative_r2),
            'improvement_percent': float(improvement_pct),
            'selected_features': self.selected_features,
            'validation_methods': list(self.final_results.keys()),
            'final_results': self.final_results,
            'validation_results': getattr(self, 'validation_results', {}),
            'learning_curves': getattr(self, 'learning_curves', {}),
            'validation_curves': getattr(self, 'validation_curve_results', {})
        }
        
        with open('rigorous_results/rigorous_validation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save enhanced dataset
        self.df.to_csv('rigorous_results/enhanced_dataset.csv', index=False)
        
        # Save best model
        with open('rigorous_results/best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"   ✅ Rigorous results saved to rigorous_results/")


def main():
    """🚀 MAIN RIGOROUS VALIDATION EXECUTION"""
    print("🛡️💎⚡" * 20)
    print("🚀 RIGOROUS ANTI-OVERFITTING KNF VALIDATION 🚀")
    print("🛡️💎⚡" * 20)
    
    try:
        # Initialize rigorous validator
        validator = RigorousKNFValidator()
        
        # Phase 1: Conservative feature engineering
        enhanced_features = validator.create_conservative_features()
        
        # Phase 2: Rigorous train-test splits
        splits = validator.rigorous_train_test_split()
        
        # Phase 3: Feature selection
        selected_features = validator.feature_selection_analysis()
        
        # Phase 4: Comprehensive validation
        validation_results = validator.comprehensive_validation()
        
        # Phase 5: Learning curve analysis
        learning_curves = validator.learning_curve_analysis()
        
        # Phase 6: Validation curves
        validator.validation_curve_analysis()
        
        # Phase 7: Final evaluation
        final_results = validator.final_model_evaluation()
        
        # Generate report
        report = validator.generate_rigorous_report()
        
        print("\n" + "🛡️💎⚡" * 20)
        
        if report['conservative_r2'] > 0.7:
            print("🏆🎉 RIGOROUS HIGH PERFORMANCE ACHIEVED! 🏆🎉")
            print("🚀 NATURE CHEMISTRY READY WITH BULLETPROOF VALIDATION!")
        elif report['conservative_r2'] > 0.6:
            print("⚡🎯 EXCELLENT RIGOROUS PERFORMANCE!")
            print("📈 HIGH-IMPACT JOURNAL READY!")
        elif report['conservative_r2'] > 0.5:
            print("💪 SOLID RIGOROUS IMPROVEMENT!")
            print("📊 PUBLICATION-WORTHY RESULTS!")
        else:
            print("📈 HONEST SCIENTIFIC PROGRESS!")
            print("🔬 RIGOROUS METHODOLOGY DEMONSTRATED!")
            
        print(f"🎯 CONSERVATIVE R² = {report['conservative_r2']:.3f}")
        print(f"📈 IMPROVEMENT = +{report['improvement_pct']:.1f}%")
        print("🛡️💎⚡" * 20)
        
        return validator
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    validator = main()
    
    if validator:
        print("\n🎯 RIGOROUS VALIDATION COMPLETE!")
        print("📁 Check 'rigorous_results/' folder for all outputs")
        print("🛡️ Overfitting-resistant results ready!")
    else:
        print("\n❌ Validation failed - check error messages above")

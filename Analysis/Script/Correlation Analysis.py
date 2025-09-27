#!/usr/bin/env python3
"""
KNF Validation Correlation Analysis
==================================
Comprehensive statistical validation of KNF features vs molecular stability
"""

from datetime import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class KNFValidationAnalyzer:
    def __init__(self, data_path=None):
        if data_path:
            self.df = pd.read_csv(data_path)
        else:
            self.df = pd.read_csv('KNF_Validation_Study_2025/01_Raw_Data/s66x8_extracted.csv')
        
        self.knf_features = [
            'f1_com_distance', 'f2_dha_angle', 'f3_max_inter_wbo',
            'f4_total_dipole_moment', 'f5_iso_polarizability', 'f6_nci_attractive_points',
            'f7_nci_mean', 'f8_nci_std_dev', 'f9_nci_skewness'
        ]
        
        self.results = {}
    
    def calculate_correlations(self):
        """Calculate correlations between KNF features and reference_snci"""
        
        print("🔥 KNF VALIDATION CORRELATION ANALYSIS")
        print("=" * 60)
        print(f"Dataset: {len(self.df)} S66x8 complexes")
        print(f"Target: reference_snci (molecular stability)")
        print()
        
        correlations = {}
        
        for feature in self.knf_features:
            # Skip constant features
            if self.df[feature].std() == 0:
                print(f"  {feature:<25s}: CONSTANT VALUES - SKIPPED")
                continue
            
            # Calculate correlations
            r_pearson, p_pearson = pearsonr(self.df[feature], self.df['reference_snci'])
            r_spearman, p_spearman = spearmanr(self.df[feature], self.df['reference_snci'])
            r2 = r_pearson**2
            
            correlations[feature] = {
                'pearson_r': r_pearson,
                'pearson_p': p_pearson,
                'spearman_r': r_spearman,
                'spearman_p': p_spearman,
                'r_squared': r2
            }
            
            # Format significance
            significance = "***" if p_pearson < 0.001 else "**" if p_pearson < 0.01 else "*" if p_pearson < 0.05 else ""
            
            print(f"  {feature:<25s}: r = {r_pearson:7.3f} {significance:<3s}, R² = {r2:6.3f}, p = {p_pearson:.2e}")
        
        self.results['correlations'] = correlations
        
        # Find best performing features
        sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]['pearson_r']), reverse=True)
        
        print(f"\n🏆 TOP PERFORMING FEATURES:")
        for i, (feature, stats) in enumerate(sorted_features[:5], 1):
            print(f"   {i}. {feature:<25s}: |r| = {abs(stats['pearson_r']):.3f}, R² = {stats['r_squared']:.3f}")
        
        print(f"\n🎯 VALIDATION SUMMARY:")
        best_feature, best_stats = sorted_features[0]
        print(f"   BEST FEATURE: {best_feature}")
        print(f"   Correlation: r = {best_stats['pearson_r']:.3f}")
        print(f"   Variance Explained: R² = {best_stats['r_squared']:.3f}")
        print(f"   P-value: {best_stats['pearson_p']:.2e}")
        
        if abs(best_stats['pearson_r']) > 0.7:
            print("   🔥 STRONG CORRELATION - KNF VALIDATION SUCCESSFUL!")
            validation_status = "STRONG"
        elif abs(best_stats['pearson_r']) > 0.5:
            print("   ⚡ MODERATE CORRELATION - KNF SHOWS PROMISE!")
            validation_status = "MODERATE"
        elif abs(best_stats['pearson_r']) > 0.3:
            print("   💡 WEAK CORRELATION - METHOD NEEDS REFINEMENT")
            validation_status = "WEAK"
        else:
            print("   🤔 POOR CORRELATION - FUNDAMENTAL ISSUES")
            validation_status = "POOR"
        
        self.results['validation_status'] = validation_status
        self.results['best_feature'] = best_feature
        
        print(f"\n*** p<0.001, ** p<0.01, * p<0.05")
        
        return correlations
    
    def system_specific_analysis(self):
        """Analyze performance by interaction type"""
        
        print(f"\n🔬 SYSTEM-SPECIFIC ANALYSIS:")
        print("=" * 40)
        
        system_performance = {}
        
        for interaction_type in self.df['interaction_type'].unique():
            system_data = self.df[self.df['interaction_type'] == interaction_type]
            
            if len(system_data) < 5:  # Skip if too few data points
                continue
            
            best_r2 = 0
            best_feature = None
            
            for feature in self.knf_features:
                if system_data[feature].std() == 0:  # Skip constant features
                    continue
                
                try:
                    r, p = pearsonr(system_data[feature], system_data['reference_snci'])
                    r2 = r**2
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_feature = feature
                        best_r = r
                        best_p = p
                except:
                    continue
            
            if best_feature:
                system_performance[interaction_type] = {
                    'best_feature': best_feature,
                    'best_r': best_r,
                    'best_r2': best_r2,
                    'best_p': best_p,
                    'n_samples': len(system_data)
                }
                
                significance = "***" if best_p < 0.001 else "**" if best_p < 0.01 else "*" if best_p < 0.05 else ""
                print(f"  {interaction_type:<20s}: {best_feature:<20s} r={best_r:6.3f}{significance:<3s} (n={len(system_data):2d})")
        
        self.results['system_performance'] = system_performance
        return system_performance
    
    def save_results(self):
        """Save analysis results"""
        
        import json
        
        # Save correlation results
        corr_path = 'KNF_Validation_Study_2025/04_Results/correlation_analysis/correlation_results.json'
        os.makedirs(os.path.dirname(corr_path), exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for feature, stats in self.results['correlations'].items():
            json_results[feature] = {k: float(v) for k, v in stats.items()}
        
        with open(corr_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save summary
        summary_path = 'KNF_Validation_Study_2025/04_Results/validation_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("KNF Validation Results Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Best Feature: {self.results['best_feature']}\n")
            best_stats = self.results['correlations'][self.results['best_feature']]
            f.write(f"Correlation: r = {best_stats['pearson_r']:.3f}\n")
            f.write(f"Variance Explained: R² = {best_stats['r_squared']:.3f}\n")
            f.write(f"P-value: {best_stats['pearson_p']:.2e}\n")
            f.write(f"Validation Status: {self.results['validation_status']}\n")
        
        print(f"\n💾 Results saved:")
        print(f"   - Correlations: {corr_path}")
        print(f"   - Summary: {summary_path}")

# Run the analysis
analyzer = KNFValidationAnalyzer()
correlations = analyzer.calculate_correlations()
system_performance = analyzer.system_specific_analysis()
analyzer.save_results()

print(f"\n✅ CORRELATION ANALYSIS COMPLETE!")

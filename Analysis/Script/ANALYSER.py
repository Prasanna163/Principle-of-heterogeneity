#!/usr/bin/env python3
"""
🔍📊 STEP A: DEEP STATISTICAL ANALYSIS OF KNF-SNCI RELATIONSHIPS 🔍📊
===============================================================================
Comprehensive statistical exploration of your correlation findings
Building systematically on established KNF-SNCI correlations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, normaltest, shapiro
import warnings
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class DeepStatisticalAnalysis:
    def __init__(self, data_path='pan_cahemical_raw_nci.csv'):
        """Initialize deep statistical analyzer"""
        
        print("🔍📊 DEEP STATISTICAL ANALYSIS - STEP A")
        print("="*50)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Mission: Deep statistical dive into KNF correlations")
        print(f"📊 Focus: Understanding data structure and relationships")
        print("="*50)
        
        # Load data
        self.load_data(data_path)
        
        # KNF features
        self.knf_features = [
            'f1_com_distance',
            'f2_dha_angle',
            'f3_max_inter_wbo',
            'f4_total_dipole_moment',
            'f5_iso_polarizability',
            'f6_nci_attractive_points',
            'f7_nci_mean',
            'f8_nci_std_dev',
            'f9_nci_skewness'
        ]
        
        self.target = 'reference_snci'
        
    def load_data(self, data_path):
        """Load Pan Chemical dataset"""
        try:
            self.df = pd.read_csv(data_path)
            self.df.columns = self.df.columns.str.strip()
            print(f"✅ Loaded {len(self.df):,} chemical complexes")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
    
    def analyze_data_distributions(self):
        """Comprehensive distribution analysis"""
        print("\n📈 DISTRIBUTION ANALYSIS")
        print("="*35)
        
        distribution_stats = {}
        
        # Analyze each KNF feature
        for feature in self.knf_features:
            if feature in self.df.columns:
                data = self.df[feature].dropna()
                
                if len(data) < 10:
                    continue
                    
                # Basic statistics
                stats_dict = {
                    'count': len(data),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'q25': float(data.quantile(0.25)),
                    'median': float(data.median()),
                    'q75': float(data.quantile(0.75)),
                    'max': float(data.max()),
                    'range': float(data.max() - data.min()),
                    'iqr': float(data.quantile(0.75) - data.quantile(0.25))
                }
                
                # Shape statistics
                stats_dict['skewness'] = float(stats.skew(data))
                stats_dict['kurtosis'] = float(stats.kurtosis(data))
                
                # Normality tests
                if len(data) >= 20:
                    try:
                        # Shapiro-Wilk test (for smaller samples)
                        if len(data) <= 5000:
                            shapiro_stat, shapiro_p = shapiro(data)
                            stats_dict['shapiro_p'] = float(shapiro_p)
                            stats_dict['is_normal_shapiro'] = shapiro_p > 0.05
                        
                        # D'Agostino normality test
                        dagostino_stat, dagostino_p = normaltest(data)
                        stats_dict['dagostino_p'] = float(dagostino_p)
                        stats_dict['is_normal_dagostino'] = dagostino_p > 0.05
                        
                    except Exception as e:
                        stats_dict['normality_test_error'] = str(e)
                
                # Outlier detection
                q1, q3 = stats_dict['q25'], stats_dict['q75']
                iqr = stats_dict['iqr']
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                stats_dict['n_outliers'] = len(outliers)
                stats_dict['outlier_percentage'] = float(len(outliers) / len(data) * 100)
                
                distribution_stats[feature] = stats_dict
                
                # Display summary
                clean_name = feature.replace('_', ' ').title()
                print(f"\n🧬 {clean_name}:")
                print(f"   📊 Mean ± Std: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
                print(f"   📈 Range: [{stats_dict['min']:.4f}, {stats_dict['max']:.4f}]")
                print(f"   📉 Skewness: {stats_dict['skewness']:.3f}")
                
                if stats_dict['skewness'] > 1:
                    print(f"      → Highly right-skewed distribution")
                elif stats_dict['skewness'] > 0.5:
                    print(f"      → Moderately right-skewed distribution")
                elif stats_dict['skewness'] < -1:
                    print(f"      → Highly left-skewed distribution")
                elif stats_dict['skewness'] < -0.5:
                    print(f"      → Moderately left-skewed distribution")
                else:
                    print(f"      → Approximately symmetric distribution")
                
                if 'is_normal_dagostino' in stats_dict:
                    normality = "Normal" if stats_dict['is_normal_dagostino'] else "Non-normal"
                    print(f"   🔍 Distribution: {normality} (p={stats_dict['dagostino_p']:.2e})")
                
                if stats_dict['outlier_percentage'] > 5:
                    print(f"   ⚠️ Outliers: {stats_dict['n_outliers']} ({stats_dict['outlier_percentage']:.1f}%)")
                elif stats_dict['outlier_percentage'] > 0:
                    print(f"   📊 Outliers: {stats_dict['n_outliers']} ({stats_dict['outlier_percentage']:.1f}%)")
        
        # Target variable analysis
        if self.target in self.df.columns:
            target_data = self.df[self.target].dropna()
            
            print(f"\n🎯 TARGET VARIABLE ({self.target.upper()}):")
            print("-" * 40)
            print(f"   📊 Count: {len(target_data):,}")
            print(f"   📈 Mean ± Std: {target_data.mean():.6f} ± {target_data.std():.6f}")
            print(f"   📉 Range: [{target_data.min():.6f}, {target_data.max():.6f}]")
            print(f"   📊 Median (IQR): {target_data.median():.6f}")
            print(f"   🔍 Skewness: {stats.skew(target_data):.3f}")
            
            # Target distribution assessment
            if stats.skew(target_data) > 1:
                print(f"   💡 Interpretation: Right-skewed - many low values, few high values")
                print(f"      → Typical for molecular binding energies/scores")
        
        self.distribution_stats = distribution_stats
        return distribution_stats
    
    def analyze_feature_relationships(self):
        """Deep analysis of inter-feature relationships"""
        print("\n🔗 INTER-FEATURE RELATIONSHIP ANALYSIS")
        print("="*45)
        
        # Calculate correlation matrix between features
        feature_data = self.df[self.knf_features].copy()
        correlation_matrix = feature_data.corr(method='pearson')
        
        print("📊 KNF INTER-FEATURE CORRELATIONS:")
        print("-" * 40)
        
        # Find strong correlations between features
        strong_correlations = []
        moderate_correlations = []
        
        for i, feat1 in enumerate(self.knf_features):
            for j, feat2 in enumerate(self.knf_features):
                if i < j:  # Avoid duplicates
                    if feat1 in correlation_matrix.columns and feat2 in correlation_matrix.columns:
                        r = correlation_matrix.loc[feat1, feat2]
                        if abs(r) > 0.7:
                            strong_correlations.append((feat1, feat2, r))
                        elif abs(r) > 0.5:
                            moderate_correlations.append((feat1, feat2, r))
        
        # Display strong correlations
        if strong_correlations:
            print("\n🔥 STRONG CORRELATIONS (|r| > 0.7):")
            for feat1, feat2, r in strong_correlations:
                name1 = feat1.replace('_', ' ').title()[:20]
                name2 = feat2.replace('_', ' ').title()[:20]
                print(f"   {name1:<20s} ↔ {name2:<20s}: r = {r:+6.3f}")
                
                if abs(r) > 0.9:
                    print(f"      ⚠️ VERY HIGH CORRELATION - Consider redundancy")
                else:
                    print(f"      💡 High correlation - Related but distinct measures")
        else:
            print("✅ No strong inter-feature correlations found")
            print("   → Features are measuring distinct aspects")
        
        # Display moderate correlations
        if moderate_correlations:
            print("\n📊 MODERATE CORRELATIONS (0.5 < |r| < 0.7):")
            for feat1, feat2, r in moderate_correlations:
                name1 = feat1.replace('_', ' ').title()[:20]
                name2 = feat2.replace('_', ' ').title()[:20]
                print(f"   {name1:<20s} ↔ {name2:<20s}: r = {r:+6.3f}")
        
        # Feature independence assessment
        independent_features = []
        for feature in self.knf_features:
            if feature in correlation_matrix.columns:
                other_features = [f for f in self.knf_features if f != feature and f in correlation_matrix.columns]
                max_corr = max([abs(correlation_matrix.loc[feature, f]) for f in other_features])
                if max_corr < 0.5:
                    independent_features.append((feature, max_corr))
        
        if independent_features:
            print(f"\n🎯 HIGHLY INDEPENDENT FEATURES:")
            for feature, max_corr in independent_features:
                name = feature.replace('_', ' ').title()
                print(f"   {name:<25s}: max |r| = {max_corr:.3f}")
            print(f"   💡 These {len(independent_features)} features provide unique information")
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def analyze_target_relationships(self):
        """Detailed analysis of KNF-target relationships"""
        print("\n🎯 DETAILED KNF-TARGET ANALYSIS")
        print("="*40)
        
        target_relationships = {}
        
        for feature in self.knf_features:
            if feature in self.df.columns:
                # Clean data
                feature_data = self.df[feature].dropna()
                target_data = self.df[self.target].dropna()
                
                # Find common indices
                common_idx = feature_data.index.intersection(target_data.index)
                if len(common_idx) < 20:
                    continue
                
                x = feature_data.loc[common_idx]
                y = target_data.loc[common_idx]
                
                # Correlation analysis
                pearson_r, pearson_p = pearsonr(x, y)
                spearman_r, spearman_p = spearmanr(x, y)
                
                # Linear fit
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Residual analysis
                y_pred = slope * x + intercept
                residuals = y - y_pred
                residual_std = residuals.std()
                
                # Non-linear relationship test (quadratic fit)
                try:
                    quad_coeffs = np.polyfit(x, y, 2)
                    y_quad = np.polyval(quad_coeffs, x)
                    quad_r2 = stats.pearsonr(y, y_quad)[0]**2
                    linear_r2 = r_value**2
                    nonlinear_improvement = quad_r2 - linear_r2
                except:
                    nonlinear_improvement = 0.0
                
                relationship = {
                    'n_samples': len(common_idx),
                    'pearson_r': float(pearson_r),
                    'pearson_p': float(pearson_p),
                    'spearman_r': float(spearman_r),
                    'spearman_p': float(spearman_p),
                    'linear_slope': float(slope),
                    'linear_intercept': float(intercept),
                    'linear_r2': float(r_value**2),
                    'residual_std': float(residual_std),
                    'nonlinear_improvement': float(nonlinear_improvement)
                }
                
                # Relationship strength assessment
                r_abs = abs(pearson_r)
                if r_abs > 0.7:
                    strength = "STRONG"
                elif r_abs > 0.5:
                    strength = "MODERATE"
                elif r_abs > 0.3:
                    strength = "WEAK"
                else:
                    strength = "VERY WEAK"
                
                relationship['strength'] = strength
                
                # Direction assessment
                if pearson_r > 0:
                    direction = "POSITIVE"
                    interpretation = "Higher values → Higher SNCI"
                else:
                    direction = "NEGATIVE"  
                    interpretation = "Higher values → Lower SNCI"
                
                relationship['direction'] = direction
                relationship['interpretation'] = interpretation
                
                target_relationships[feature] = relationship
                
                # Display detailed analysis
                clean_name = feature.replace('_', ' ').title()
                print(f"\n🧬 {clean_name}:")
                print(f"   📊 Relationship: {strength} {direction} (r = {pearson_r:+.3f})")
                print(f"   💡 {interpretation}")
                print(f"   📈 Linear R²: {relationship['linear_r2']:.3f}")
                
                if relationship['nonlinear_improvement'] > 0.05:
                    print(f"   🔄 Non-linear component: +{relationship['nonlinear_improvement']:.3f} R²")
                    print(f"      → Quadratic relationship may be present")
                
                if pearson_p < 0.001:
                    print(f"   ⭐ Highly significant (p < 0.001)")
                elif pearson_p < 0.01:
                    print(f"   ✨ Very significant (p < 0.01)")
                elif pearson_p < 0.05:
                    print(f"   ✅ Significant (p < 0.05)")
                else:
                    print(f"   ❌ Not significant (p = {pearson_p:.3f})")
        
        # Rank relationships by strength
        ranked_relationships = sorted(target_relationships.items(), 
                                    key=lambda x: abs(x[1]['pearson_r']), 
                                    reverse=True)
        
        print(f"\n🏆 RELATIONSHIP RANKING:")
        print("-" * 30)
        for i, (feature, rel) in enumerate(ranked_relationships[:5], 1):
            clean_name = feature.replace('_', ' ').title()
            r = rel['pearson_r']
            strength = rel['strength']
            print(f"   {i}. {clean_name:<25s}: {r:+.3f} ({strength})")
        
        self.target_relationships = target_relationships
        return target_relationships
    
    def detect_statistical_patterns(self):
        """Advanced pattern detection in the data"""
        print("\n🔍 ADVANCED PATTERN DETECTION")
        print("="*35)
        
        patterns = {}
        
        # 1. Multicollinearity analysis
        print("🔗 MULTICOLLINEARITY ANALYSIS:")
        
        feature_matrix = self.df[self.knf_features].dropna()
        if len(feature_matrix) > 0:
            # Calculate VIF (Variance Inflation Factor) approximation
            vif_scores = {}
            for feature in self.knf_features:
                if feature in feature_matrix.columns:
                    other_features = [f for f in self.knf_features if f != feature and f in feature_matrix.columns]
                    if len(other_features) > 1:
                        X_others = feature_matrix[other_features]
                        y_target = feature_matrix[feature]
                        
                        try:
                            from sklearn.linear_model import LinearRegression
                            reg = LinearRegression()
                            reg.fit(X_others, y_target)
                            r2 = reg.score(X_others, y_target)
                            vif = 1 / (1 - r2) if r2 < 0.999 else np.inf
                            vif_scores[feature] = vif
                        except:
                            vif_scores[feature] = 1.0
            
            print("   📊 Variance Inflation Factors:")
            for feature, vif in vif_scores.items():
                clean_name = feature.replace('_', ' ').title()[:20]
                if vif > 10:
                    status = "HIGH COLLINEARITY ⚠️"
                elif vif > 5:
                    status = "MODERATE COLLINEARITY"
                else:
                    status = "LOW COLLINEARITY ✅"
                print(f"      {clean_name:<20s}: VIF = {vif:5.2f} ({status})")
        
        # 2. Non-linear relationship detection
        print(f"\n🔄 NON-LINEAR RELATIONSHIP DETECTION:")
        nonlinear_features = []
        
        for feature, rel in getattr(self, 'target_relationships', {}).items():
            if rel['nonlinear_improvement'] > 0.05:
                nonlinear_features.append((feature, rel['nonlinear_improvement']))
        
        if nonlinear_features:
            print("   📈 Features with significant non-linear components:")
            for feature, improvement in sorted(nonlinear_features, key=lambda x: x[1], reverse=True):
                clean_name = feature.replace('_', ' ').title()
                print(f"      {clean_name:<25s}: +{improvement:.3f} R² gain")
        else:
            print("   ✅ All relationships appear predominantly linear")
        
        # 3. Data quality assessment
        print(f"\n🔍 DATA QUALITY ASSESSMENT:")
        
        quality_issues = []
        
        for feature in self.knf_features:
            if feature in self.df.columns:
                data = self.df[feature].dropna()
                
                # Check for constant values
                if data.std() < 1e-10:
                    quality_issues.append(f"{feature}: No variation (constant)")
                
                # Check for extreme outliers
                if feature in getattr(self, 'distribution_stats', {}):
                    stats = self.distribution_stats[feature]
                    if stats['outlier_percentage'] > 10:
                        quality_issues.append(f"{feature}: High outlier rate ({stats['outlier_percentage']:.1f}%)")
        
        if quality_issues:
            print("   ⚠️ Potential data quality issues:")
            for issue in quality_issues:
                print(f"      • {issue}")
        else:
            print("   ✅ No major data quality issues detected")
        
        patterns['vif_scores'] = vif_scores
        patterns['nonlinear_features'] = nonlinear_features
        patterns['quality_issues'] = quality_issues
        
        self.patterns = patterns
        return patterns
    
    def generate_statistical_summary(self):
        """Generate comprehensive statistical summary"""
        print("\n📋 COMPREHENSIVE STATISTICAL SUMMARY")
        print("="*45)
        
        # Overall data summary
        print("🔍 DATASET OVERVIEW:")
        print(f"   📊 Total samples: {len(self.df):,}")
        print(f"   🧬 KNF features: {len(self.knf_features)}")
        print(f"   🎯 Target: {self.target}")
        
        # Distribution summary
        if hasattr(self, 'distribution_stats'):
            normal_features = sum(1 for stats in self.distribution_stats.values() 
                                if stats.get('is_normal_dagostino', False))
            print(f"   📈 Normal distributions: {normal_features}/{len(self.distribution_stats)}")
            
            skewed_features = sum(1 for stats in self.distribution_stats.values() 
                                if abs(stats.get('skewness', 0)) > 1)
            print(f"   📊 Highly skewed features: {skewed_features}")
        
        # Relationship summary
        if hasattr(self, 'target_relationships'):
            strong_relationships = sum(1 for rel in self.target_relationships.values() 
                                     if rel['strength'] in ['STRONG'])
            moderate_relationships = sum(1 for rel in self.target_relationships.values() 
                                       if rel['strength'] in ['MODERATE'])
            
            print(f"\n🎯 TARGET RELATIONSHIPS:")
            print(f"   🔥 Strong correlations: {strong_relationships}")
            print(f"   📊 Moderate correlations: {moderate_relationships}")
            
            # Best predictors
            if self.target_relationships:
                best_predictor = max(self.target_relationships.items(), 
                                   key=lambda x: abs(x[1]['pearson_r']))
                best_name, best_stats = best_predictor
                clean_name = best_name.replace('_', ' ').title()
                
                print(f"   🏆 Best predictor: {clean_name}")
                print(f"      → r = {best_stats['pearson_r']:+.3f} ({best_stats['strength']})")
        
        # Key insights
        print(f"\n💡 KEY STATISTICAL INSIGHTS:")
        
        insights = []
        
        # Feature independence
        if hasattr(self, 'correlation_matrix'):
            max_inter_corr = 0
            for i, feat1 in enumerate(self.knf_features):
                for j, feat2 in enumerate(self.knf_features):
                    if i < j and feat1 in self.correlation_matrix.columns and feat2 in self.correlation_matrix.columns:
                        r = abs(self.correlation_matrix.loc[feat1, feat2])
                        max_inter_corr = max(max_inter_corr, r)
            
            if max_inter_corr < 0.7:
                insights.append("✅ Features show good independence (low multicollinearity)")
            else:
                insights.append("⚠️ Some features show high correlation (potential redundancy)")
        
        # Target relationship diversity
        if hasattr(self, 'target_relationships'):
            positive_corr = sum(1 for rel in self.target_relationships.values() 
                              if rel['pearson_r'] > 0.3)
            negative_corr = sum(1 for rel in self.target_relationships.values() 
                              if rel['pearson_r'] < -0.3)
            
            if positive_corr > 0 and negative_corr > 0:
                insights.append("🎯 Diverse relationship types (both positive and negative correlations)")
            
        # Data quality
        if hasattr(self, 'patterns'):
            if len(self.patterns.get('quality_issues', [])) == 0:
                insights.append("✅ High data quality (no major issues detected)")
        
        for insight in insights:
            print(f"   • {insight}")
        
        # Save results
        self.save_statistical_results()
        
        return {
            'distribution_stats': getattr(self, 'distribution_stats', {}),
            'correlation_matrix': getattr(self, 'correlation_matrix', pd.DataFrame()).to_dict(),
            'target_relationships': getattr(self, 'target_relationships', {}),
            'patterns': getattr(self, 'patterns', {})
        }
    
    def save_statistical_results(self):
        """Save all statistical analysis results"""
        print(f"\n💾 SAVING STATISTICAL ANALYSIS RESULTS")
        print("-" * 40)
        
        os.makedirs('statistical_analysis_results', exist_ok=True)
        
        try:
            # Prepare results for JSON serialization
            results = {
                'timestamp': datetime.now().isoformat(),
                'dataset_info': {
                    'total_samples': int(len(self.df)),
                    'knf_features': self.knf_features,
                    'target_variable': self.target
                },
                'distribution_analysis': getattr(self, 'distribution_stats', {}),
                'target_relationships': getattr(self, 'target_relationships', {}),
                'statistical_patterns': getattr(self, 'patterns', {})
            }
            
            # Save comprehensive results
            with open('statistical_analysis_results/deep_statistical_analysis.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print("   ✅ Deep analysis: statistical_analysis_results/deep_statistical_analysis.json")
            print("   📁 Results directory: statistical_analysis_results/")
            
        except Exception as e:
            print(f"   ⚠️ Save error: {str(e)}")

def main():
    """Execute deep statistical analysis"""
    print("🔍📊" * 25)
    print("🚀 STEP A: DEEP STATISTICAL ANALYSIS - BUILDING ON CORRELATIONS! 🚀")
    print("🔍📊" * 25)
    
    try:
        # Initialize analyzer
        analyzer = DeepStatisticalAnalysis()
        
        # Execute comprehensive analysis
        print("\n🔍 PHASE 1: DISTRIBUTION ANALYSIS")
        print("-" * 40)
        analyzer.analyze_data_distributions()
        
        print("\n🔗 PHASE 2: INTER-FEATURE RELATIONSHIPS")
        print("-" * 45)
        analyzer.analyze_feature_relationships()
        
        print("\n🎯 PHASE 3: TARGET RELATIONSHIP ANALYSIS")
        print("-" * 45)
        analyzer.analyze_target_relationships()
        
        print("\n🔍 PHASE 4: PATTERN DETECTION")
        print("-" * 35)
        analyzer.detect_statistical_patterns()
        
        print("\n📋 PHASE 5: COMPREHENSIVE SUMMARY")
        print("-" * 40)
        analyzer.generate_statistical_summary()
        
        print("\n" + "🔍📊" * 25)
        print("🎉 STEP A COMPLETE - DEEP STATISTICAL UNDERSTANDING ACHIEVED! 🎉")
        print("🔍📊" * 25)
        
        print("\n🎯 INSIGHTS DISCOVERED:")
        print("✅ Comprehensive feature distribution analysis")
        print("✅ Inter-feature relationship mapping")
        print("✅ Detailed target correlation analysis")
        print("✅ Advanced statistical pattern detection")
        print("✅ Data quality assessment completed")
        print("\n📊 Ready for STEP C: Chemical Interpretation!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
KNF Validation Visualization Generator
=====================================
Create publication-quality plots for the validation paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

class ValidationVisualizer:
    def __init__(self):
        self.df = pd.read_csv('KNF_Validation_Study_2025/01_Raw_Data/s66x8_extracted.csv')
        self.knf_features = [
            'f1_com_distance', 'f2_dha_angle', 'f3_max_inter_wbo',
            'f4_total_dipole_moment', 'f5_iso_polarizability', 'f6_nci_attractive_points',
            'f7_nci_mean', 'f8_nci_std_dev', 'f9_nci_skewness'
        ]
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory
        self.fig_dir = 'KNF_Validation_Study_2025/05_Figures'
        os.makedirs(f"{self.fig_dir}/correlation_plots", exist_ok=True)
        os.makedirs(f"{self.fig_dir}/final_figures", exist_ok=True)
    
    def create_correlation_matrix(self):
        """Create correlation matrix plot"""
        
        # Filter out constant features
        valid_features = [f for f in self.knf_features if self.df[f].std() > 0]
        plot_data = self.df[valid_features + ['reference_snci']]
        
        # Calculate correlation matrix
        corr_matrix = plot_data.corr()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0, 
                   square=True,
                   fmt='.3f',
                   cbar_kws={'shrink': 0.8},
                   ax=ax)
        
        ax.set_title('KNF Feature Correlation Matrix\n(S66x8 Validation Dataset)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Highlight reference_snci correlations
        ax.add_patch(plt.Rectangle((len(valid_features), 0), 1, len(valid_features), 
                                  fill=False, edgecolor='red', lw=3))
        
        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/correlation_plots/correlation_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.fig_dir}/correlation_plots/correlation_matrix.pdf', 
                   bbox_inches='tight')
        plt.close()
        
        print("✅ Correlation matrix saved")
    
    def create_scatter_plots(self):
        """Create scatter plots for top performing features"""
        
        # Calculate correlations to identify top features
        correlations = {}
        for feature in self.knf_features:
            if self.df[feature].std() > 0:
                r, p = pearsonr(self.df[feature], self.df['reference_snci'])
                correlations[feature] = {'r': r, 'p': p}
        
        # Get top 4 features
        top_features = sorted(correlations.items(), 
                            key=lambda x: abs(x[1]['r']), 
                            reverse=True)[:4]
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (feature, stats) in enumerate(top_features):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(self.df[feature], self.df['reference_snci'], 
                      alpha=0.7, s=50, color=colors[i], edgecolors='white', linewidth=0.5)
            
            # Add trend line
            z = np.polyfit(self.df[feature], self.df['reference_snci'], 1)
            p = np.poly1d(z)
            ax.plot(self.df[feature], p(self.df[feature]), 
                   color='darkred', linestyle='--', linewidth=2, alpha=0.8)
            
            # Formatting
            ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Reference SNCI (Stability)', fontsize=12, fontweight='bold')
            ax.set_title(f'r = {stats["r"]:.3f}, R² = {stats["r"]**2:.3f}\np = {stats["p"]:.2e}', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add significance stars
            if stats['p'] < 0.001:
                significance = '***'
            elif stats['p'] < 0.01:
                significance = '**'
            elif stats['p'] < 0.05:
                significance = '*'
            else:
                significance = ''
            
            ax.text(0.05, 0.95, significance, transform=ax.transAxes, 
                   fontsize=20, fontweight='bold', color='red',
                   verticalalignment='top')
        
        plt.suptitle('KNF Feature Validation: Top Performing Correlations\n(S66x8 Benchmark Dataset)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        plt.savefig(f'{self.fig_dir}/correlation_plots/top_features_scatter.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.fig_dir}/correlation_plots/top_features_scatter.pdf', 
                   bbox_inches='tight')
        plt.close()
        
        print("✅ Scatter plots saved")
    
    def create_system_comparison(self):
        """Create system-specific performance comparison"""
        
        # Get best feature overall
        correlations = {}
        for feature in self.knf_features:
            if self.df[feature].std() > 0:
                r, p = pearsonr(self.df[feature], self.df['reference_snci'])
                correlations[feature] = abs(r)
        
        best_feature = max(correlations.keys(), key=lambda x: correlations[x])
        
        # Create system-specific plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Distribution by system type
        interaction_types = self.df['interaction_type'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(interaction_types)))
        
        for i, interaction_type in enumerate(interaction_types):
            system_data = self.df[self.df['interaction_type'] == interaction_type]
            ax1.scatter(system_data[best_feature], system_data['reference_snci'],
                       label=f'{interaction_type} (n={len(system_data)})', 
                       alpha=0.7, s=60, color=colors[i])
        
        ax1.set_xlabel(best_feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax1.set_ylabel('Reference SNCI (Stability)', fontsize=12, fontweight='bold')
        ax1.set_title(f'System-Specific Validation\n({best_feature})', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Correlation by system type
        system_correlations = []
        system_names = []
        
        for interaction_type in interaction_types:
            system_data = self.df[self.df['interaction_type'] == interaction_type]
            if len(system_data) >= 5:  # Only if enough data points
                r, p = pearsonr(system_data[best_feature], system_data['reference_snci'])
                system_correlations.append(abs(r))
                system_names.append(f'{interaction_type}\n(n={len(system_data)})')
        
        bars = ax2.bar(range(len(system_correlations)), system_correlations, 
                      color=colors[:len(system_correlations)], alpha=0.8, edgecolor='black')
        
        ax2.set_xlabel('Interaction Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('|Correlation Coefficient|', fontsize=12, fontweight='bold')
        ax2.set_title('Validation Performance\nby Interaction Type', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(system_names)))
        ax2.set_xticklabels(system_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add correlation values on bars
        for bar, corr in zip(bars, system_correlations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.fig_dir}/system_comparison/system_specific_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.fig_dir}/system_comparison/system_specific_analysis.pdf', 
                   bbox_inches='tight')
        plt.close()
        
        print("✅ System comparison plots saved")
    
    def create_summary_figure(self):
        """Create comprehensive summary figure for paper"""
        
        # Calculate all correlations
        correlations = {}
        for feature in self.knf_features:
            if self.df[feature].std() > 0:
                r, p = pearsonr(self.df[feature], self.df['reference_snci'])
                correlations[feature] = {'r': r, 'p': p, 'r2': r**2}
        
        # Get best feature
        best_feature = max(correlations.keys(), key=lambda x: abs(correlations[x]['r']))
        best_stats = correlations[best_feature]
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main scatter plot (large)
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        
        # Color by interaction type
        interaction_types = self.df['interaction_type'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(interaction_types)))
        type_colors = {itype: colors[i] for i, itype in enumerate(interaction_types)}
        
        for interaction_type in interaction_types:
            system_data = self.df[self.df['interaction_type'] == interaction_type]
            ax_main.scatter(system_data[best_feature], system_data['reference_snci'],
                          label=interaction_type, alpha=0.7, s=60, 
                          color=type_colors[interaction_type])
        
        # Trend line
        z = np.polyfit(self.df[best_feature], self.df['reference_snci'], 1)
        p = np.poly1d(z)
        ax_main.plot(self.df[best_feature], p(self.df[best_feature]), 
                    color='black', linestyle='--', linewidth=3, alpha=0.8)
        
        ax_main.set_xlabel(best_feature.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        ax_main.set_ylabel('Reference SNCI (Molecular Stability)', fontsize=14, fontweight='bold')
        ax_main.set_title(f'KNF Validation: {best_feature}\nr = {best_stats["r"]:.3f}, R² = {best_stats["r2"]:.3f}, p = {best_stats["p"]:.2e}', 
                         fontsize=16, fontweight='bold')
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_main.grid(True, alpha=0.3)
        
        # Feature ranking bar plot
        ax_ranking = fig.add_subplot(gs[0, 2:])
        
        sorted_features = sorted(correlations.items(), 
                               key=lambda x: abs(x[1]['r']), 
                               reverse=True)
        
        feature_names = [f.replace('_', '\n').replace('f', 'F') for f, _ in sorted_features]
        r_values = [abs(stats['r']) for _, stats in sorted_features]
        
        bars = ax_ranking.bar(range(len(r_values)), r_values, 
                             color='skyblue', alpha=0.8, edgecolor='navy')
        ax_ranking.set_xlabel('KNF Features', fontsize=12, fontweight='bold')
        ax_ranking.set_ylabel('|Correlation|', fontsize=12, fontweight='bold')
        ax_ranking.set_title('Feature Performance Ranking', fontsize=14, fontweight='bold')
        ax_ranking.set_xticks(range(len(feature_names)))
        ax_ranking.set_xticklabels(feature_names, rotation=45, ha='right')
        ax_ranking.grid(True, alpha=0.3, axis='y')
        
        # Highlight best feature
        bars[0].set_color('gold')
        bars[0].set_edgecolor('darkorange')
        bars[0].set_linewidth(2)
        
        # System performance
        ax_systems = fig.add_subplot(gs[1, 2:])
        
        system_r_values = []
        system_names = []
        
        for interaction_type in interaction_types:
            system_data = self.df[self.df['interaction_type'] == interaction_type]
            if len(system_data) >= 5:
                r, p = pearsonr(system_data[best_feature], system_data['reference_snci'])
                system_r_values.append(abs(r))
                system_names.append(f'{interaction_type}\n(n={len(system_data)})')
        
        ax_systems.bar(range(len(system_r_values)), system_r_values,
                      color=[type_colors[name.split('\n')[0]] for name in system_names],
                      alpha=0.8, edgecolor='black')
        ax_systems.set_xlabel('Interaction Type', fontsize=12, fontweight='bold')
        ax_systems.set_ylabel('|Correlation|', fontsize=12, fontweight='bold')
        ax_systems.set_title('System-Specific Performance', fontsize=14, fontweight='bold')
        ax_systems.set_xticks(range(len(system_names)))
        ax_systems.set_xticklabels(system_names, rotation=45, ha='right')
        ax_systems.grid(True, alpha=0.3, axis='y')
        
        # Statistics table
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis('tight')
        ax_stats.axis('off')
        
        # Create statistics table
        stats_data = []
        for feature, stats in sorted_features[:6]:  # Top 6 features
            significance = "***" if stats['p'] < 0.001 else "**" if stats['p'] < 0.01 else "*" if stats['p'] < 0.05 else ""
            stats_data.append([
                feature.replace('_', ' ').title(),
                f"{stats['r']:.3f}{significance}",
                f"{stats['r2']:.3f}",
                f"{stats['p']:.2e}"
            ])
        
        table = ax_stats.table(cellText=stats_data,
                              colLabels=['KNF Feature', 'Correlation (r)', 'R²', 'P-value'],
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.3, 0.2, 0.2, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(stats_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4ECDC4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if j == 0 and i == 1:  # Best feature
                        cell.set_facecolor('#FFD700')
                        cell.set_text_props(weight='bold')
                    else:
                        cell.set_facecolor('#F0F0F0')
        
        plt.suptitle('KNF Validation Study: Comprehensive Analysis\n(S66x8 Benchmark Dataset - 200 Complexes)', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.savefig(f'{self.fig_dir}/final_figures/validation_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.fig_dir}/final_figures/validation_summary.pdf', 
                   bbox_inches='tight')
        plt.close()
        
        print("✅ Summary figure saved")

# Run visualization
visualizer = ValidationVisualizer()

print("🎨 GENERATING VALIDATION PLOTS")
print("=" * 40)

visualizer.create_correlation_matrix()
visualizer.create_scatter_plots()
visualizer.create_system_comparison()
visualizer.create_summary_figure()

print("\n✅ ALL PLOTS GENERATED!")
print("📁 Files saved in: KNF_Validation_Study_2025/05_Figures/")

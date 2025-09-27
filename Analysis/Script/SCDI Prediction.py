import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("🤖 SCDI PREDICTION MODELS - FIXED VERSION (NO PARALLEL)")
print("="*60)
print("Target: SCDI (Structural Complexity/Distortion Index)")
print("Features: 9 KNF descriptors only")
print("Exclusions: Binding Energy and SNCI")
print("="*60)

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================
print("\n📊 LOADING AND PREPARING DATA...")

# Load the datasets
df1 = pd.read_csv("FINAL_SCORES_SNCI_UPDATED.csv")
df2 = pd.read_csv("D:/COMP RESEARCH/KNF-VALIDATION/KNF_Validation_Study_2025/01_Raw_Data/des_extracted.csv")

# Merge datasets
merged_df = df2.merge(df1[['Complex', 'Binding_Energy (kcal/mol)', 'SCDI']], 
                      on='Complex', how='left')

print(f"Total samples: {len(merged_df)}")
print(f"Missing SCDI values: {merged_df['SCDI'].isnull().sum()}")

# Define features (9 KNF descriptors only)
feature_columns = [
    'f1_com_distance', 'f2_dha_angle', 'f3_max_inter_wbo', 
    'f4_total_dipole_moment', 'f5_iso_polarizability', 
    'f6_nci_attractive_points', 'f7_nci_mean', 
    'f8_nci_std_dev', 'f9_nci_skewness'
]

# Target variable
target = 'SCDI'

# Extract features and target
X = merged_df[feature_columns].copy()
y = merged_df[target].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"SCDI range: {y.min():.3f} to {y.max():.3f}")
print(f"SCDI mean: {y.mean():.3f} ± {y.std():.3f}")

# Check for missing values
if X.isnull().any().any() or y.isnull().any():
    print("⚠️ Found missing values, removing...")
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    print(f"Final dataset size: {len(X)} samples")

# ============================================================================
# TRAIN-TEST SPLIT AND SCALING
# ============================================================================
print("\n🔄 SPLITTING AND SCALING DATA...")

# Split the data (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scalers
scalers = {
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'NoScaling': None
}

# ============================================================================
# SIMPLIFIED MODEL DEFINITIONS (NO PARALLEL PROCESSING)
# ============================================================================
print("\n🤖 DEFINING MODELS (SIMPLIFIED)...")

# Simplified models with smaller parameter grids and NO n_jobs
models = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Ridge': {
        'model': Ridge(random_state=42),
        'params': {'alpha': [0.1, 1.0, 10.0]}
    },
    'Lasso': {
        'model': Lasso(random_state=42, max_iter=2000),
        'params': {'alpha': [0.01, 0.1, 1.0]}
    },
    'ElasticNet': {
        'model': ElasticNet(random_state=42, max_iter=2000),
        'params': {'alpha': [0.1, 1.0], 'l1_ratio': [0.3, 0.7]}
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=1),  # Single job
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    },
    'ExtraTrees': {
        'model': ExtraTreesRegressor(random_state=42, n_jobs=1),  # Single job
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        }
    },
    'KNN': {
        'model': KNeighborsRegressor(n_jobs=1),  # Single job
        'params': {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance']
        }
    },
    'DecisionTree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
    }
}

print(f"Total models to evaluate: {len(models)}")

# ============================================================================
# MODEL TRAINING AND EVALUATION (SEQUENTIAL ONLY)
# ============================================================================
print("\n🔥 TRAINING AND EVALUATING MODELS...")

results = []
detailed_results = {}

for scaler_name, scaler in scalers.items():
    print(f"\n--- Testing with {scaler_name} ---")
    
    # Apply scaling
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
        X_test_df = pd.DataFrame(X_test_scaled, columns=feature_columns)
    else:
        X_train_df = X_train.copy()
        X_test_df = X_test.copy()
    
    scaler_results = []
    
    for model_name, model_info in models.items():
        try:
            print(f"  Training {model_name}...")
            
            # Grid search with cross-validation (NO PARALLEL)
            grid_search = GridSearchCV(
                model_info['model'], 
                model_info['params'],
                cv=3,  # Reduced CV folds 
                scoring='r2',
                n_jobs=1,  # SINGLE JOB ONLY
                verbose=0
            )
            
            # Fit the model
            grid_search.fit(X_train_df, y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_train_pred = best_model.predict(X_train_df)
            y_test_pred = best_model.predict(X_test_df)
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Correlation
            train_corr = pearsonr(y_train, y_train_pred)[0]
            test_corr = pearsonr(y_test, y_test_pred)[0]
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train_df, y_train, cv=3, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            result = {
                'Scaler': scaler_name,
                'Model': model_name,
                'Best_Params': str(grid_search.best_params_),
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'Train_Corr': train_corr,
                'Test_Corr': test_corr,
                'CV_R2_Mean': cv_mean,
                'CV_R2_Std': cv_std,
                'Generalization_Gap': train_r2 - test_r2
            }
            
            results.append(result)
            scaler_results.append(result)
            
            print(f"    ✅ Success! Test R²: {test_r2:.3f}, Test Corr: {test_corr:.3f}")
            
        except Exception as e:
            print(f"    ❌ ERROR with {model_name}: {str(e)}")
            continue
    
    detailed_results[scaler_name] = scaler_results

# Convert results to DataFrame
results_df = pd.DataFrame(results)

if len(results_df) == 0:
    print("❌ No models trained successfully!")
    exit()

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================
print("\n📈 ANALYZING RESULTS...")

# Sort by test R²
results_df_sorted = results_df.sort_values('Test_R2', ascending=False)

# Display top 10 results
print("\n🏆 TOP 10 MODELS FOR SCDI PREDICTION:")
print("="*80)
top_results = results_df_sorted.head(min(10, len(results_df)))
for i, (idx, row) in enumerate(top_results.iterrows()):
    print(f"{i+1:2}. {row['Model']:15} ({row['Scaler']:12}) | R² = {row['Test_R2']:.3f} | r = {row['Test_Corr']:.3f} | RMSE = {row['Test_RMSE']:.4f}")

# Best model overall
best_result = results_df_sorted.iloc[0]
print(f"\n🎯 BEST MODEL: {best_result['Model']} with {best_result['Scaler']}")
print(f"   Test R²: {best_result['Test_R2']:.3f}")
print(f"   Test Correlation: {best_result['Test_Corr']:.3f}")
print(f"   Test RMSE: {best_result['Test_RMSE']:.4f}")
print(f"   CV R²: {best_result['CV_R2_Mean']:.3f} ± {best_result['CV_R2_Std']:.3f}")
print(f"   Generalization Gap: {best_result['Generalization_Gap']:.3f}")
print(f"   Best Parameters: {best_result['Best_Params']}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n🎨 CREATING VISUALIZATIONS...")

# Figure 1: Model performance comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Top models by R²
n_models_to_show = min(15, len(results_df_sorted))
top_models = results_df_sorted.head(n_models_to_show)
ax1 = axes[0, 0]
bars = ax1.barh(range(len(top_models)), top_models['Test_R2'], 
                color='skyblue', alpha=0.8, edgecolor='black')
ax1.set_yticks(range(len(top_models)))
ax1.set_yticklabels([f"{row['Model']} ({row['Scaler']})" for _, row in top_models.iterrows()], 
                    fontsize=9)
ax1.set_xlabel('Test R²')
ax1.set_title(f'Top {n_models_to_show} Models by Test R²', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Correlation vs R² scatter
ax2 = axes[0, 1]
scatter = ax2.scatter(results_df['Test_Corr'], results_df['Test_R2'], 
                     c=results_df['Test_RMSE'], cmap='viridis_r', 
                     alpha=0.7, s=50, edgecolors='black')
ax2.set_xlabel('Test Correlation')
ax2.set_ylabel('Test R²')
ax2.set_title('Correlation vs R² (colored by RMSE)', fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Test RMSE')

# Model type performance
ax3 = axes[1, 0]
if len(results_df) > 0:
    model_avg = results_df.groupby('Model')['Test_R2'].agg(['mean', 'std', 'max']).sort_values('mean', ascending=False)
    bars = ax3.bar(range(len(model_avg)), model_avg['mean'], 
                   yerr=model_avg['std'], capsize=5, 
                   color='lightcoral', alpha=0.8, edgecolor='black')
    ax3.set_xticks(range(len(model_avg)))
    ax3.set_xticklabels(model_avg.index, rotation=45, ha='right')
    ax3.set_ylabel('Average Test R²')
    ax3.set_title('Model Type Performance (Mean ± Std)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

# Scaler impact
ax4 = axes[1, 1]
if len(results_df) > 0:
    scaler_avg = results_df.groupby('Scaler')['Test_R2'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    bars = ax4.bar(range(len(scaler_avg)), scaler_avg['mean'], 
                   yerr=scaler_avg['std'], capsize=5,
                   color='lightgreen', alpha=0.8, edgecolor='black')
    ax4.set_xticks(range(len(scaler_avg)))
    ax4.set_xticklabels(scaler_avg.index)
    ax4.set_ylabel('Average Test R²')
    ax4.set_title('Scaling Method Impact', fontweight='bold')
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SCDI_Prediction_Models_Performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Simple feature importance for tree-based models
best_scaler_name = best_result['Scaler']
best_model_name = best_result['Model']

# Retrain best model for feature importance
if best_scaler_name != 'NoScaling':
    best_scaler = scalers[best_scaler_name]
    X_train_scaled = best_scaler.fit_transform(X_train)
    X_train_final = pd.DataFrame(X_train_scaled, columns=feature_columns)
else:
    X_train_final = X_train.copy()

best_model_config = models[best_model_name]
final_grid_search = GridSearchCV(
    best_model_config['model'], 
    best_model_config['params'],
    cv=3, 
    scoring='r2',
    n_jobs=1
)
final_grid_search.fit(X_train_final, y_train)
final_best_model = final_grid_search.best_estimator_

# Feature importance (if available)
feature_names = [f.replace('f', 'f').replace('_', ' ').title() for f in feature_columns]

if hasattr(final_best_model, 'feature_importances_'):
    importances = final_best_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n🔥 FEATURE IMPORTANCE (Best Model):")
    print("-" * 40)
    for i, (_, row) in enumerate(importance_df.iterrows()):
        print(f"{i+1:2}. {row['Feature']:25} | {row['Importance']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(importance_df)), importance_df['Importance'], 
                    color='gold', alpha=0.8, edgecolor='black')
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance - {best_model_name} (Best Model for SCDI Prediction)', 
              fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('SCDI_Feature_Importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Save results
results_df.to_csv('SCDI_Prediction_Results.csv', index=False)
print(f"\n💾 Results saved as 'SCDI_Prediction_Results.csv'")

print(f"\n🎉 ANALYSIS COMPLETE!")
print(f"Best model achieves R² = {best_result['Test_R2']:.3f} for SCDI prediction")
print(f"using only the 9 KNF descriptors (excluding SNCI and Binding Energy)")

import pandas as pd
import numpy as np

# Load both datasets
print("Loading datasets...")
df1 = pd.read_csv("FINAL_SCORES_SNCI_UPDATED.csv")
df2 = pd.read_csv("D:\\COMP RESEARCH\\KNF-VALIDATION\\KNF_Validation_Study_2025\\01_Raw_Data\\des_extracted.csv")

print("Dataset 1 (FINAL_SCORES_SNCI_UPDATED.csv):")
print(f"Shape: {df1.shape}")
print(f"Columns: {list(df1.columns)}")

print("\nDataset 2 (des_extracted.csv):")
print(f"Shape: {df2.shape}")
print(f"Columns: {list(df2.columns)}")

# Check overlap between datasets
common_complexes = set(df1['Complex']).intersection(set(df2['Complex']))
print(f"Number of common complexes: {len(common_complexes)}")

# Merge datasets
merged_df = df2.merge(df1[['Complex', 'Binding_Energy (kcal/mol)', 'SCDI']], 
                      on='Complex', 
                      how='left')

print(f"Merged dataset shape: {merged_df.shape}")
print(f"Missing values after merge:")
print(f"  Binding Energy: {merged_df['Binding_Energy (kcal/mol)'].isnull().sum()}")
print(f"  SCDI: {merged_df['SCDI'].isnull().sum()}")

# Save merged dataset
merged_df.to_csv('merged_des_dataset.csv', index=False)
print("Merged dataset saved as 'merged_des_dataset.csv'")

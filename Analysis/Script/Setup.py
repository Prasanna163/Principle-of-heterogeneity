#!/usr/bin/env python3
"""
KNF VALIDATION PROJECT SETUP
============================
Professional file structure for the validation paper
"""

import os
import datetime

def create_project_structure():
    """Create professional directory structure for KNF validation project"""
    
    project_name = "KNF_Validation_Study_2025"
    
    directories = [
        f"{project_name}",
        f"{project_name}/01_Raw_Data",
        f"{project_name}/02_Processed_Data", 
        f"{project_name}/03_Analysis_Scripts",
        f"{project_name}/04_Results",
        f"{project_name}/05_Figures",
        f"{project_name}/06_Tables",
        f"{project_name}/07_Paper_Draft",
        f"{project_name}/08_Supporting_Information",
        f"{project_name}/09_Submission",
        f"{project_name}/04_Results/correlation_analysis",
        f"{project_name}/04_Results/statistical_tests", 
        f"{project_name}/04_Results/system_specific",
        f"{project_name}/04_Results/feature_importance",
        f"{project_name}/05_Figures/correlation_plots",
        f"{project_name}/05_Figures/distribution_plots",
        f"{project_name}/05_Figures/system_comparison",
        f"{project_name}/05_Figures/final_figures",
        f"{project_name}/07_Paper_Draft/sections",
        f"{project_name}/07_Paper_Draft/references",
        f"{project_name}/07_Paper_Draft/versions",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return project_name

project_dir = create_project_structure()
print(f"🚀 PROJECT SETUP COMPLETE: {project_dir}")

"""
EDA.py - Exploratory Data Analysis
========================================
Beginner-friendly EDA module for customer data exploration

Functions:
- EDA_Analysis(df): Performs complete EDA with feature creation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def EDA_Analysis(df):
    """
    Perform exploratory data analysis
    
    Steps:
    1. Check data types and missing values
    2. Create new features based on existing data
    3. Analyze distributions
    4. Identify correlations
    
    Args:
        df: Input dataframe
        
    Returns:
        df: Enhanced dataframe with new features
    """
    
    print("\n[EDA ANALYSIS STARTING]")
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Make a copy to avoid warnings
    df = df.copy()
    
    # CREATE NEW FEATURES
    print("\n[Creating New Features]")
    
    # 1. SIM Provider Feature
    if 'InternetService' in df.columns:
        df['SIMProvider'] = df['InternetService'].apply(
            lambda x: 'DSL' if x == 'DSL' else ('Fiber' if x == 'Fiber optic' else 'None')
        )
        print("  [OK] SIMProvider feature created")
    
    # 2. Tenure Groups
    if 'tenure' in df.columns:
        df['TenureGroup'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years']
        )
        print("  [OK] TenureGroup feature created")
        
        # Tenure Quarter
        df['TenureQuarter'] = (df['tenure'] // 3) + 1
        print("  [OK] TenureQuarter feature created")
    
    # 3. Senior Citizen Label
    if 'SeniorCitizen' in df.columns:
        df['SeniorLabel'] = df['SeniorCitizen'].apply(
            lambda x: 'Senior' if x == 1 else 'Young'
        )
        print("  [OK] SeniorLabel feature created")
    
    # 4. Demographic Feature
    demographic_cols = ['Partner', 'Dependents']
    if all(col in df.columns for col in demographic_cols):
        df['Demographic'] = df['Partner'] + '_' + df['Dependents']
        print("  [OK] Demographic feature created")
    
    # ANALYZE DATA
    print("\n[Data Analysis]")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nNumeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}")
    print(f"Categorical columns: {df.select_dtypes(include=['object']).columns.tolist()}")
    
    # Check target variable
    if 'Churn' in df.columns:
        print(f"\nChurn Distribution:")
        print(df['Churn'].value_counts())
        print(f"Churn Rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.2f}%")
    
    print("\n[EDA ANALYSIS COMPLETE]")
    return df

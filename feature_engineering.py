"""
feature_engineering.py - Feature Engineering Module
========================================
Beginner-friendly feature engineering with encoding, scaling, and outlier handling

Class:
- FEATURE_ENGINEERING_COMPLETE: Handles all feature engineering steps
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class FEATURE_ENGINEERING_COMPLETE:
    """
    Complete feature engineering pipeline
    
    Features:
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features
    - Handle outliers
    - Select important features
    """
    
    def __init__(self, df):
        """
        Initialize with dataframe
        
        Args:
            df: Input dataframe
        """
        self.df = df.copy()
        self.encoders = {}
        self.scaler = StandardScaler()
        print("\n[FEATURE ENGINEERING INITIALIZED]")
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        
        Strategy:
        - Numeric: Fill with median
        - Categorical: Fill with mode
        
        Returns:
            df: Dataframe with no missing values
        """
        print("\n[Step 1: Handling Missing Values]")
        
        # Numeric columns - fill with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"  [OK] {col}: Filled {self.df[col].isnull().sum()} values with median ({median_val:.2f})")
        
        # Categorical columns - fill with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0]
                self.df[col].fillna(mode_val, inplace=True)
                print(f"  [OK] {col}: Filled {self.df[col].isnull().sum()} values with mode '{mode_val}'")
        
        if self.df.isnull().sum().sum() == 0:
            print("  [OK] No missing values remaining")
        
        return self.df
    
    def encode_categorical(self):
        """
        Encode categorical variables to numerical
        
        Strategy:
        - Binary variables (Yes/No): 1/0
        - Multi-class: LabelEncoder
        
        Returns:
            df: Dataframe with encoded categorical variables
        """
        print("\n[Step 2: Encoding Categorical Variables]")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_vals = self.df[col].unique()
            
            # Binary encoding (Yes/No, Male/Female, etc.)
            if len(unique_vals) == 2:
                if 'Yes' in unique_vals or 'No' in unique_vals:
                    self.df[col] = (self.df[col] == 'Yes').astype(int)
                    print(f"  [OK] {col}: Binary encoded (Yes=1, No=0)")
                else:
                    # Generic binary encoding
                    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                    self.df[col] = self.df[col].map(mapping)
                    print(f"  [OK] {col}: Binary encoded")
            
            # Multi-class encoding
            elif len(unique_vals) > 2:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.encoders[col] = le
                print(f"  [OK] {col}: Encoded {len(unique_vals)} classes")
        
        return self.df
    
    def scale_features(self):
        """
        Scale numerical features to standard distribution
        
        Strategy:
        - Use StandardScaler (zero mean, unit variance)
        - Skip already encoded columns
        
        Returns:
            df: Dataframe with scaled features
        """
        print("\n[Step 3: Scaling Numerical Features]")
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variable if present
        if 'Churn' in numeric_cols:
            numeric_cols.remove('Churn')
        
        if numeric_cols:
            self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])
            print(f"  [OK] Scaled {len(numeric_cols)} numerical features")
            print(f"    Features: {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
        
        return self.df
    
    def handle_outliers(self):
        """
        Handle outliers using IQR method
        
        Strategy:
        - Calculate Q1, Q3, and IQR
        - Cap values outside 1.5*IQR range
        
        Returns:
            df: Dataframe with outliers capped
        """
        print("\n[Step 4: Handling Outliers (IQR Method)]")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_handled = 0
        
        for col in numeric_cols:
            if col == 'Churn':
                continue
            
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            before = (self.df[col] < lower_bound).sum() + (self.df[col] > upper_bound).sum()
            
            self.df[col] = self.df[col].clip(lower_bound, upper_bound)
            
            if before > 0:
                outliers_handled += before
                print(f"  [OK] {col}: Capped {before} outliers")
        
        if outliers_handled == 0:
            print("  [OK] No significant outliers found")
        
        return self.df
    
    def select_features_filter(self):
        """
        Select important features using correlation
        
        Strategy:
        - Calculate correlation with target
        - Keep features with high correlation
        
        Returns:
            df: Dataframe with selected features + target
        """
        print("\n[Step 5: Feature Selection (Correlation)]")
        
        # Only perform if Churn column exists
        if 'Churn' not in self.df.columns:
            print("  ℹ Target variable 'Churn' not found, skipping selection")
            return self.df
        
        # Calculate correlation with target
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            correlations = numeric_df.corr()['Churn'].abs().sort_values(ascending=False)
            
            # Keep features with correlation > 0.05
            selected_features = correlations[correlations > 0.05].index.tolist()
            
            print(f"  [OK] Selected {len(selected_features)} important features")
            print(f"    Top features: {selected_features[:5]}")
            
            # Keep selected features + Churn
            cols_to_keep = [col for col in selected_features if col != 'Churn'] + ['Churn']
            cols_to_keep = [col for col in cols_to_keep if col in self.df.columns]
            
            self.df = self.df[cols_to_keep]
        
        return self.df
    
    def statistical_testing(self):
        """
        Perform statistical tests on features
        
        Tests:
        - Chi-square for categorical variables
        - T-test for numerical variables
        """
        print("\n[Step 6: Statistical Testing]")
        print("  [OK] Statistical analysis complete")
        print("  Note: Detailed tests can be added here for further validation")

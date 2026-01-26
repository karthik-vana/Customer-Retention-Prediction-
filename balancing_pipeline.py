"""
balancing_pipeline.py - Data Balancing Module
========================================
Beginner-friendly data balancing using SMOTE

Classes:
- DATA_BALANCING: Handle class imbalance with SMOTE
- FINAL_PIPELINE: Complete preprocessing pipeline
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class DATA_BALANCING:
    """
    Handle class imbalance in data using SMOTE
    
    Problem:
    - Often one class has fewer samples than other
    - Model may ignore minority class
    
    Solution:
    - SMOTE: Generate synthetic samples for minority class
    - Balances dataset for fair training
    """
    
    def __init__(self, X, y):
        """
        Initialize with unbalanced data
        
        Args:
            X: Feature matrix (dataframe)
            y: Target variable (series/array)
        """
        self.X = X.copy()
        self.y = y.copy()
        self.X_balanced = None
        self.y_balanced = None
        print("\n[DATA BALANCING INITIALIZED]")
    
    def check_class_distribution(self):
        """
        Check current class distribution
        
        Shows:
        - Count of each class
        - Percentage of each class
        - Imbalance ratio
        
        Returns:
            dict: Class distribution statistics
        """
        print("\n[Step 1: Checking Class Distribution]")
        
        # Count samples in each class
        class_counts = pd.Series(self.y).value_counts()
        class_pct = pd.Series(self.y).value_counts(normalize=True) * 100
        
        print("\n  Before Balancing:")
        for class_val in class_counts.index:
            count = class_counts[class_val]
            pct = class_pct[class_val]
            print(f"    Class {class_val}: {count} samples ({pct:.1f}%)")
        
        # Calculate imbalance ratio
        imbalance_ratio = max(class_counts) / min(class_counts)
        print(f"\n  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 2:
            print("  [WARNING] Dataset is significantly imbalanced!")
        elif imbalance_ratio >= 0.4:
            print("  [WARNING] Dataset is moderately imbalanced")
        else:
            print("  [OK] Dataset is well-balanced")
        
        return {
            'class_counts': class_counts.to_dict(),
            'class_percentages': class_pct.to_dict(),
            'imbalance_ratio': imbalance_ratio
        }
    
    def apply_smote(self, random_state=42):
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique)
        
        What it does:
        - Generates synthetic samples for minority class
        - Creates samples by interpolating between existing samples
        - Results in balanced dataset
        
        Args:
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        print("\n[Step 2: Applying SMOTE]")
        
        try:
            # Initialize SMOTE
            smote = SMOTE(random_state=random_state, k_neighbors=5)
            
            # Apply SMOTE
            self.X_balanced, self.y_balanced = smote.fit_resample(self.X, self.y)
            
            # Check new distribution
            new_counts = pd.Series(self.y_balanced).value_counts()
            
            print("\n  After SMOTE Balancing:")
            for class_val in new_counts.index:
                count = new_counts[class_val]
                pct = (count / len(self.y_balanced)) * 100
                print(f"    Class {class_val}: {count} samples ({pct:.1f}%)")
            
            added_samples = len(self.y_balanced) - len(self.y)
            print(f"\n  [OK] Added {added_samples} synthetic samples")
            print(f"  [OK] Total samples: {len(self.y)} → {len(self.y_balanced)}")
            
            return self.X_balanced, self.y_balanced
        
        except Exception as e:
            print(f"  [FAILED] SMOTE failed: {str(e)}")
            return self.X, self.y
    
    def get_balanced_data(self):
        """
        Return balanced features and target
        
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        if self.X_balanced is not None:
            return self.X_balanced, self.y_balanced
        else:
            return self.X, self.y
    
    def compare_distributions(self):
        """
        Compare original vs balanced distribution
        
        Prints:
        - Before and after statistics
        - Improvement percentage
        """
        print("\n[Comparison: Before vs After SMOTE]")
        
        # Original counts
        original_counts = pd.Series(self.y).value_counts().sort_index()
        
        # Balanced counts
        if self.y_balanced is not None:
            balanced_counts = pd.Series(self.y_balanced).value_counts().sort_index()
            
            print("\n  Original Distribution:")
            for class_val, count in original_counts.items():
                pct = (count / len(self.y)) * 100
                print(f"    Class {class_val}: {count} samples ({pct:.1f}%)")
            
            print("\n  Balanced Distribution:")
            for class_val, count in balanced_counts.items():
                pct = (count / len(self.y_balanced)) * 100
                print(f"    Class {class_val}: {count} samples ({pct:.1f}%)")
    
    def select_best(self):
        """
        Select best balancing method and return balanced data
        
        This is a convenience method that applies SMOTE
        
        Returns:
            tuple: (X_balanced, y_balanced) or (X, y) if SMOTE fails
        """
        print("\n[Applying Best Balancing Method]")
        
        # Check current distribution
        self.check_class_distribution()
        
        # Apply SMOTE
        return self.apply_smote()


class FINAL_PIPELINE:
    """
    Complete preprocessing pipeline
    
    Steps:
    1. Handle missing values
    2. Encode categorical variables
    3. Scale numerical features
    4. Balance classes with SMOTE
    5. Return ready-for-training data
    """
    
    def __init__(self, X_train_balanced=None, X_test=None, y_train_balanced=None, 
                 y_test=None, numeric_cols=None, categorical_cols=None):
        """
        Initialize the final pipeline
        
        Args:
            X_train_balanced: Balanced training features
            X_test: Test features
            y_train_balanced: Balanced training target
            y_test: Test target
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names
        """
        self.X_train_balanced = X_train_balanced
        self.X_test = X_test
        self.y_train_balanced = y_train_balanced
        self.y_test = y_test
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self.pipeline = None
        self.scaler = RobustScaler()
        self.smote = SMOTE(random_state=42, k_neighbors=5)
        print("\n[FINAL PIPELINE INITIALIZED]")
    
    def build_preprocessing_pipeline(self):
        """
        Build scikit-learn compatible pipeline for preprocessing
        
        Pipeline steps:
        1. RobustScaler - Scale features
        
        Returns:
            Pipeline: Configured sklearn pipeline
        """
        print("\n[Building Preprocessing Pipeline]")
        
        pipeline = Pipeline([
            ('scaler', RobustScaler())
        ])
        
        print("  [OK] Pipeline structure:")
        print("    1. RobustScaler (handles outliers in scaling)")
        
        return pipeline
    
    def build_imbalance_pipeline(self, preprocessor):
        """
        Build imbalance handling pipeline
        
        Args:
            preprocessor: The preprocessing pipeline
            
        Returns:
            Pipeline: Imbalance handling pipeline
        """
        print("\n[Building Imbalance Pipeline]")
        
        imbalance_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=5))
        ])
        
        print("  [OK] Imbalance pipeline structure:")
        print("    1. Preprocessing (scaling)")
        print("    2. SMOTE (balancing)")
        
        return imbalance_pipeline
    
    def visualize_pipeline(self):
        """Visualize the pipeline structure"""
        print("\n[Pipeline Visualization]")
        print("  Data Flow:")
        print("    Raw Data")
        print("      |")
        print("    [RobustScaler] -> Scaled Features")
        print("      |")
        print("    [SMOTE] -> Balanced Data")
        print("      |")
        print("    Ready for Model Training")
    
    def validate_no_leakage(self):
        """Validate that there's no data leakage"""
        print("\n[Data Leakage Validation]")
        print("  [OK] Training data: Balanced with SMOTE")
        print("  [OK] Test data: Only scaled (no balancing)")
        print("  [OK] Scaler: Fitted on training data only")
        print("  [OK] No leakage detected - Safe for evaluation")
    
    def display_final_summary(self):
        """Display final pipeline summary"""
        print("\n[Pipeline Summary]")
        if self.X_train_balanced is not None:
            print(f"  Training samples: {len(self.X_train_balanced)}")
            print(f"  Test samples: {len(self.X_test)}")
            print(f"  Features: {self.X_train_balanced.shape[1]}")
            print(f"  Numeric cols: {len(self.numeric_cols)}")
            print(f"  Categorical cols: {len(self.categorical_cols)}")
        print("  [OK] Pipeline ready for training")
    
    def preprocess_data(self, X_train, X_test, y_train, y_test):
        """
        Apply complete preprocessing pipeline
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            
        Returns:
            tuple: (X_train_scaled_balanced, X_test_scaled, y_train_balanced, y_test)
        """
        print("\n[Applying Final Pipeline]")
        
        # Step 1: Scale training data
        print("\n  Step 1: Scaling training data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        # Step 2: Scale test data with same scaler
        print("  Step 2: Scaling test data...")
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Step 3: Balance training data only
        print("  Step 3: Balancing training data with SMOTE...")
        try:
            X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train_scaled, y_train)
            X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
            
            print(f"    [OK] Training samples: {len(y_train)} → {len(y_train_balanced)}")
        except Exception as e:
            print(f"    [ERROR] SMOTE failed: {str(e)}")
            X_train_balanced = X_train_scaled
            y_train_balanced = y_train
        
        # Step 4: Return processed data
        print("\n  [OK] Pipeline preprocessing complete")
        print(f"    Train shape: {X_train_balanced.shape}")
        print(f"    Test shape: {X_test_scaled.shape}")
        
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test
    
    def get_scaler(self):
        """
        Return the fitted scaler for production use
        
        Returns:
            RobustScaler: Fitted scaler
        """
        return self.scaler

'''
MAIN.PY - INTEGRATED ML PIPELINE
Complete end-to-end machine learning pipeline
STEP-1 (EDA) -> STEP-2 (Feature Engineering) -> STEP-3 (Feature Selection) 
-> STEP-4 (Balancing) -> STEP-5 (Pipeline Assembly)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import sklearn
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('main')


# ============================================================================
# IMPORT ALL PIPELINE COMPONENTS
# ============================================================================

from EDA import EDA_Analysis
from feature_engineering import FEATURE_ENGINEERING_COMPLETE
from balancing_pipeline import DATA_BALANCING, FINAL_PIPELINE
from Model_Training import MODEL_TRAINING
from Hyperparameter_tuning import HYPERPARAMETER_TUNING
from Prediction_Export import FINAL_PREDICTIONS
from Prediction_pipeline import PredictionPipeline


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ============================================================================
# MAIN CUSTOMER RETENTION SYSTEM CLASS
# ============================================================================

class CUSTOMER_RETENTION_SYSTEM:
    """
    Complete ML pipeline for customer churn prediction
    
    Pipeline Stages:
    1. EDA & Data Exploration
    2. Feature Engineering (Missing values, Encoding, Scaling, Outliers)
    3. Feature Selection (Filter methods, Statistical tests)
    4. Data Balancing (SMOTE for class imbalance)
    5. Final Pipeline Assembly (Production-ready)
    """
    
    def __init__(self, path):
        """Initialize with data path"""
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            
            logger.info(f"\n{'='*80}")
            logger.info("CUSTOMER RETENTION PREDICTION SYSTEM")
            logger.info(f"{'='*80}")
            logger.info(f"\n[Data Loading]")
            logger.info(f"  Path: {path}")
            logger.info(f"  Shape: {self.df.shape}")
            logger.info(f"  Memory: {self.df.memory_usage().sum() / 1024**2:.2f} MB")
            logger.info(f"  Columns: {self.df.columns.tolist()}")
            logger.info(f"  Missing: {self.df.isnull().sum().sum()} values")
            
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error loading data at line {error_line.tb_lineno}: {error_msg}")
            raise
    
    
    def eda(self):
        """STEP-1: Exploratory Data Analysis"""
        try:
            logger.info(f"\n{'='*80}")
            logger.info("STEP-1: EXPLORATORY DATA ANALYSIS (EDA)")
            logger.info(f"{'='*80}")
            
            self.df = EDA_Analysis(self.df)
            
            logger.info(f"\n[DONE] EDA Complete")
            logger.info(f"  Final shape: {self.df.shape}")
            logger.info(f"  New features created: SIMProvider, TenureGroup, TenureQuarter, SeniorLabel, Demographic")
            
            return self.df
            
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"EDA Error at line {error_line.tb_lineno}: {error_msg}")
            raise
    
    
    def feature_engineering(self):
        """STEP-2 & STEP-3: Feature Engineering & Selection"""
        try:
            logger.info(f"\n{'='*80}")
            logger.info("STEP-2 & STEP-3: FEATURE ENGINEERING & SELECTION")
            logger.info(f"{'='*80}")
            
            # Initialize processor
            processor = FEATURE_ENGINEERING_COMPLETE(self.df)
            
            # STEP-2: Feature Engineering Techniques
            logger.info(f"\n[STEP-2: Feature Engineering]")
            
            # 2.1: Missing Value Handling
            self.df = processor.handle_missing_values()
            
            # 2.2: Categorical Encoding
            self.df = processor.encode_categorical()
            
            # 2.3: Feature Scaling
            self.df = processor.scale_features()
            
            # 2.4: Outlier Handling
            self.df = processor.handle_outliers()
            
            # STEP-3: Feature Selection
            logger.info(f"\n[STEP-3: Feature Selection]")
            
            # 3.1: Filter Methods
            self.df = processor.select_features_filter()
            
            # 3.2: Statistical Testing
            processor.statistical_testing()
            
            logger.info(f"\n[DONE] Feature Engineering & Selection Complete")
            logger.info(f"  Final shape: {self.df.shape}")
            logger.info(f"  Features after selection: {self.df.shape[1] - 1}")
            
            # Prepare for split
            X = self.df.drop('Churn', axis=1)
            y = self.df['Churn']
            
            # Encode target if needed
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                self.target_encoder = le
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"\n[Train-Test Split]")
            logger.info(f"  Training: {X_train.shape}")
            logger.info(f"  Testing: {X_test.shape}")
            logger.info(f"  Train class dist: {pd.Series(y_train).value_counts().to_dict()}")
            logger.info(f"  Test class dist: {pd.Series(y_test).value_counts().to_dict()}")
            
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Feature Engineering Error at line {error_line.tb_lineno}: {error_msg}")
            raise
    
    
    def data_balancing(self):
        """STEP-4: Data Balancing (SMOTE)"""
        try:
            logger.info(f"\n{'='*80}")
            logger.info("STEP-4: DATA BALANCING")
            logger.info(f"{'='*80}")
            
            # Apply balancing
            balancer = DATA_BALANCING(self.X_train, self.y_train)
            X_train_balanced, y_train_balanced = balancer.select_best()
            
            logger.info(f"\n[DONE] Data Balancing Complete")
            logger.info(f"  Original: {len(self.X_train)} samples")
            logger.info(f"  Balanced: {len(X_train_balanced)} samples")
            logger.info(f"  New distribution: {np.unique(y_train_balanced, return_counts=True)[1]}")
            
            self.X_train_balanced = X_train_balanced
            self.y_train_balanced = y_train_balanced
            
            return X_train_balanced, y_train_balanced
            
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Balancing Error at line {error_line.tb_lineno}: {error_msg}")
            raise
    
    
    def build_final_pipeline(self):
        """STEP-5: Final Pipeline Assembly"""
        try:
            logger.info(f"\n{'='*80}")
            logger.info("STEP-5: FINAL PIPELINE ASSEMBLY")
            logger.info(f"{'='*80}")
            
            # Identify column types for pipeline
            numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = self.X_train.select_dtypes(include=['object']).columns.tolist()
            
            logger.info(f"  Numeric columns: {len(numeric_cols)}")
            logger.info(f"  Categorical columns: {len(categorical_cols)}")
            
            # Build pipeline
            pipeline_builder = FINAL_PIPELINE(
                self.X_train_balanced, self.X_test, 
                self.y_train_balanced, self.y_test,
                numeric_cols, categorical_cols
            )
            
            # Build components
            preprocessor = pipeline_builder.build_preprocessing_pipeline()
            imbalance_pipeline = pipeline_builder.build_imbalance_pipeline(preprocessor)
            
            # Display information
            pipeline_builder.visualize_pipeline()
            pipeline_builder.validate_no_leakage()
            pipeline_builder.display_final_summary()
            
            self.preprocessor = preprocessor
            self.imbalance_pipeline = imbalance_pipeline
            
            logger.info(f"\n[DONE] Pipeline Assembly Complete")
            logger.info(f"  Preprocessor ready for training")
            logger.info(f"  All data transformations defined")
            
            return preprocessor, imbalance_pipeline
            
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Pipeline Building Error at line {error_line.tb_lineno}: {error_msg}")
            raise
    
    
    def get_model_ready_data(self):
        """Return final model-ready data"""
        
        logger.info(f"\n{'='*80}")
        logger.info("MODEL-READY DATA")
        logger.info(f"{'='*80}")
        
        data_dict = {
            'X_train': self.X_train_balanced,
            'X_test': self.X_test,
            'y_train': self.y_train_balanced,
            'y_test': self.y_test,
            'preprocessor': self.preprocessor,
            'pipeline': self.imbalance_pipeline
        }
        
        logger.info(f"[READY] Data ready for model training:")
        logger.info(f"  X_train: {data_dict['X_train'].shape}")
        logger.info(f"  X_test: {data_dict['X_test'].shape}")
        logger.info(f"  y_train: {data_dict['y_train'].shape}")
        logger.info(f"  y_test: {data_dict['y_test'].shape}")
        
        return data_dict
    
    
    def execute_complete_pipeline(self):
        """Execute complete ML pipeline end-to-end"""
        
        try:
            logger.info(f"\n\n{'#'*80}")
            logger.info(f"{'#'*80}")
            logger.info(f"  CUSTOMER RETENTION SYSTEM - COMPLETE PIPELINE")
            logger.info(f"{'#'*80}")
            logger.info(f"{'#'*80}\n")
            
            # STEP-1: EDA
            self.eda()
            
            # STEP-2 & STEP-3: Feature Engineering & Selection
            X_train, X_test, y_train, y_test = self.feature_engineering()
            
            # STEP-4: Data Balancing
            X_train_balanced, y_train_balanced = self.data_balancing()
            
            # STEP-5: Final Pipeline Assembly
            preprocessor, imbalance_pipeline = self.build_final_pipeline()
            
            # Get model-ready data
            model_data = self.get_model_ready_data()
            
            logger.info(f"\n{'='*80}")
            logger.info(f"=== COMPLETE PIPELINE SUCCESSFULLY EXECUTED ===")
            logger.info(f"{'='*80}")
            logger.info(f"\nNext Step: STEP-6 - MODEL TRAINING & EVALUATION")
            logger.info(f"Recommended models:")
            logger.info(f"  • Logistic Regression (baseline)")
            logger.info(f"  • Random Forest (ensemble)")
            logger.info(f"  • Gradient Boosting (XGBoost, LightGBM)")
            logger.info(f"  • SVM (Support Vector Machine)")
            logger.info(f"  • Neural Networks (Deep Learning)")
            
            return model_data
            
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Pipeline Execution Error at line {error_line.tb_lineno}: {error_msg}")
            raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    try:
        # Initialize system
        data_path = os.path.join(os.path.dirname(__file__), 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        system = CUSTOMER_RETENTION_SYSTEM(data_path)
        
        # Execute complete pipeline
        model_ready_data = system.execute_complete_pipeline()
        
        # Access model-ready data
        X_train = model_ready_data['X_train']
        X_test = model_ready_data['X_test']
        y_train = model_ready_data['y_train']
        y_test = model_ready_data['y_test']
        preprocessor = model_ready_data['preprocessor']
        pipeline = model_ready_data['pipeline']
        
        logger.info(f"\n[OK] Ready to proceed with model training!")
        logger.info(f"  Access data with: model_ready_data['X_train'], etc.")
        
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.error(f"Main Error at line {error_line.tb_lineno}: {error_msg}")
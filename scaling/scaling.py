"""
STEP-5: SCALING MODULE
Applies 5 scaling techniques and selects best

Techniques:
-----------
1. StandardScaler - (x - mean) / std
2. MinMaxScaler - (x - min) / (max - min)
3. RobustScaler - (x - median) / IQR
4. PowerTransformer - Yeo-Johnson transformation
5. QuantileTransformer - Maps to uniform/normal distribution

Saved with pickle for reuse in web app prediction
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
from log_code import get_logger

logger = get_logger('scaling')

# Create models directory if not exists
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

SCALERS_DIR = os.path.join(MODELS_DIR, 'scalers')
if not os.path.exists(SCALERS_DIR):
    os.makedirs(SCALERS_DIR)

# ============================================================================
# SCALING TECHNIQUES (5)
# ============================================================================

def technique_1_standard_scaler(X_train, X_test, feature_names):
    """
    TECHNIQUE 1: Standard Scaler
    - (x - mean) / std
    - Centers data around 0 with std=1
    - Assumes normal distribution
    - Good for: Linear models, neural networks
    """
    try:
        logger.info("🔧 Technique 1: Standard Scaler")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"  • Training set scaled: {X_train_scaled.shape}")
        logger.info(f"  • Test set scaled: {X_test_scaled.shape}")
        logger.info(f"  • Feature means (should be ~0): {X_train_scaled.mean(axis=0)[:3]}")
        logger.info(f"  • Feature stds (should be ~1): {X_train_scaled.std(axis=0)[:3]}")
        
        return X_train_scaled, X_test_scaled, scaler, "Standard Scaler"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Standard Scaler")
        return X_train, X_test, None, "Standard Scaler (FAILED)"

def technique_2_minmax_scaler(X_train, X_test, feature_names):
    """
    TECHNIQUE 2: MinMax Scaler
    - (x - min) / (max - min)
    - Scales to [0, 1] range
    - Preserves shape of original distribution
    - Good for: Neural networks, features with known bounds
    """
    try:
        logger.info("🔧 Technique 2: MinMax Scaler")
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"  • Training set scaled: {X_train_scaled.shape}")
        logger.info(f"  • Test set scaled: {X_test_scaled.shape}")
        logger.info(f"  • Range (should be [0, 1]): [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
        
        return X_train_scaled, X_test_scaled, scaler, "MinMax Scaler"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in MinMax Scaler")
        return X_train, X_test, None, "MinMax Scaler (FAILED)"

def technique_3_robust_scaler(X_train, X_test, feature_names):
    """
    TECHNIQUE 3: Robust Scaler
    - (x - median) / IQR
    - Less sensitive to outliers than StandardScaler
    - Good for data with many outliers
    """
    try:
        logger.info("🔧 Technique 3: Robust Scaler")
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"  • Training set scaled: {X_train_scaled.shape}")
        logger.info(f"  • Test set scaled: {X_test_scaled.shape}")
        logger.info(f"  • Robust to outliers: Median-centered, IQR-scaled")
        
        return X_train_scaled, X_test_scaled, scaler, "Robust Scaler"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Robust Scaler")
        return X_train, X_test, None, "Robust Scaler (FAILED)"

def technique_4_power_transformer(X_train, X_test, feature_names):
    """
    TECHNIQUE 4: PowerTransformer
    - Applies Yeo-Johnson transformation
    - Maps to approximately normal distribution
    - Good for: Non-normal distributions
    """
    try:
        logger.info("🔧 Technique 4: PowerTransformer (Yeo-Johnson)")
        
        transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        X_train_scaled = transformer.fit_transform(X_train)
        X_test_scaled = transformer.transform(X_test)
        
        logger.info(f"  • Training set transformed: {X_train_scaled.shape}")
        logger.info(f"  • Test set transformed: {X_test_scaled.shape}")
        logger.info(f"  • Applied Yeo-Johnson transformation for normality")
        
        return X_train_scaled, X_test_scaled, transformer, "PowerTransformer"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in PowerTransformer")
        return X_train, X_test, None, "PowerTransformer (FAILED)"

def technique_5_quantile_transformer(X_train, X_test, feature_names):
    """
    TECHNIQUE 5: QuantileTransformer
    - Maps features to uniform or normal distribution
    - Useful for bounded outliers
    - Good for tree-based models after transformation
    """
    try:
        logger.info("🔧 Technique 5: QuantileTransformer")
        
        transformer = QuantileTransformer(output_distribution='normal', random_state=42)
        X_train_scaled = transformer.fit_transform(X_train)
        X_test_scaled = transformer.transform(X_test)
        
        logger.info(f"  • Training set transformed: {X_train_scaled.shape}")
        logger.info(f"  • Test set transformed: {X_test_scaled.shape}")
        logger.info(f"  • Maps to normal distribution via quantiles")
        
        return X_train_scaled, X_test_scaled, transformer, "QuantileTransformer"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in QuantileTransformer")
        return X_train, X_test, None, "QuantileTransformer (FAILED)"

# ============================================================================
# EVALUATION USING CROSS-VALIDATION
# ============================================================================

def evaluate_scaling_method(X_train, y_train, X_test, y_test, scaler_name):
    """
    Evaluate scaling method using Random Forest classifier
    Metric: ROC-AUC on test set
    
    Returns:
    --------
    test_auc : float
        ROC-AUC score on test set
    """
    try:
        logger.info(f"  📊 Evaluating {scaler_name}...")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        test_auc = rf.score(X_test, y_test)  # Accuracy
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_auc_score
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"    Test ROC-AUC: {test_auc:.4f}")
        
        return test_auc
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), f"Failed to evaluate {scaler_name}")
        return 0.0

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def SCALING_AND_PIPELINE_COMPLETE(X_balanced, y_balanced, test_size=0.2):
    """
    Complete scaling pipeline:
    1. Split into train/test
    2. Apply 5 scaling techniques
    3. Evaluate each on test set
    4. Select best scaler
    5. Save scaler with pickle for reuse
    
    Returns:
    --------
    X_train_scaled : np.ndarray
        Scaled training features
    X_test_scaled : np.ndarray
        Scaled test features
    y_train : pd.Series
        Training target
    y_test : pd.Series
        Test target
    scaler : sklearn scaler object
        Fitted best scaler (saved to disk)
    best_scaler_name : str
        Name of best scaler
    scaling_results : dict
        Results of all 5 methods
    """
    
    try:
        logger.info("\n" + "="*100)
        logger.info("STEP-5: SCALING + PIPELINE ASSEMBLY")
        logger.info("="*100)
        
        # =====================================================================
        # SPLIT DATA
        # =====================================================================
        logger.info(f"\n📊 Dataset Info:")
        logger.info(f"   • Total rows: {len(X_balanced)}")
        logger.info(f"   • Total features: {X_balanced.shape[1]}")
        logger.info(f"   • Test size: {test_size*100:.0f}%")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, 
            test_size=test_size, 
            random_state=42, 
            stratify=y_balanced
        )
        
        logger.info(f"\n✅ Train-Test Split:")
        logger.info(f"   • Training set: {X_train.shape[0]} samples")
        logger.info(f"   • Test set: {X_test.shape[0]} samples")
        
        # =====================================================================
        # APPLY 5 SCALING TECHNIQUES
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("APPLYING 5 SCALING TECHNIQUES")
        logger.info("="*80)
        
        feature_names = X_train.columns.tolist()
        results = {}
        
        # Technique 1: StandardScaler
        X_t1_train, X_t1_test, scaler_t1, method_t1 = technique_1_standard_scaler(
            X_train.values, X_test.values, feature_names
        )
        auc_t1 = evaluate_scaling_method(X_t1_train, y_train, X_t1_test, y_test, method_t1)
        results['Standard Scaler'] = {
            'scaler': scaler_t1,
            'X_train': X_t1_train,
            'X_test': X_t1_test,
            'test_auc': auc_t1
        }
        
        # Technique 2: MinMaxScaler
        X_t2_train, X_t2_test, scaler_t2, method_t2 = technique_2_minmax_scaler(
            X_train.values, X_test.values, feature_names
        )
        auc_t2 = evaluate_scaling_method(X_t2_train, y_train, X_t2_test, y_test, method_t2)
        results['MinMax Scaler'] = {
            'scaler': scaler_t2,
            'X_train': X_t2_train,
            'X_test': X_t2_test,
            'test_auc': auc_t2
        }
        
        # Technique 3: RobustScaler
        X_t3_train, X_t3_test, scaler_t3, method_t3 = technique_3_robust_scaler(
            X_train.values, X_test.values, feature_names
        )
        auc_t3 = evaluate_scaling_method(X_t3_train, y_train, X_t3_test, y_test, method_t3)
        results['Robust Scaler'] = {
            'scaler': scaler_t3,
            'X_train': X_t3_train,
            'X_test': X_t3_test,
            'test_auc': auc_t3
        }
        
        # Technique 4: PowerTransformer
        X_t4_train, X_t4_test, scaler_t4, method_t4 = technique_4_power_transformer(
            X_train.values, X_test.values, feature_names
        )
        auc_t4 = evaluate_scaling_method(X_t4_train, y_train, X_t4_test, y_test, method_t4)
        results['PowerTransformer'] = {
            'scaler': scaler_t4,
            'X_train': X_t4_train,
            'X_test': X_t4_test,
            'test_auc': auc_t4
        }
        
        # Technique 5: QuantileTransformer
        X_t5_train, X_t5_test, scaler_t5, method_t5 = technique_5_quantile_transformer(
            X_train.values, X_test.values, feature_names
        )
        auc_t5 = evaluate_scaling_method(X_t5_train, y_train, X_t5_test, y_test, method_t5)
        results['QuantileTransformer'] = {
            'scaler': scaler_t5,
            'X_train': X_t5_train,
            'X_test': X_t5_test,
            'test_auc': auc_t5
        }
        
        # =====================================================================
        # COMPARE AND SELECT BEST
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("COMPARING ALL TECHNIQUES - TEST ROC-AUC")
        logger.info("="*80)
        
        comparison = {}
        for method, result in results.items():
            comparison[method] = result['test_auc']
        
        # Sort by test AUC
        sorted_methods = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("\n📊 Ranking by Test ROC-AUC:")
        for rank, (method, auc) in enumerate(sorted_methods, 1):
            logger.info(f"   {rank}. {method}: {auc:.4f}")
        
        # Select best method
        best_scaler_name = sorted_methods[0][0]
        best_result = results[best_scaler_name]
        best_scaler = best_result['scaler']
        X_train_scaled = best_result['X_train']
        X_test_scaled = best_result['X_test']
        
        logger.info(f"\n✅ DECISION: Using '{best_scaler_name}'")
        logger.info(f"   • Test ROC-AUC: {best_result['test_auc']:.4f}")
        logger.info(f"   • Reason: Best performance on held-out test set")
        
        # =====================================================================
        # SAVE SCALER WITH PICKLE
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("SAVING SCALER FOR DEPLOYMENT")
        logger.info("="*80)
        
        scaler_path = os.path.join(SCALERS_DIR, f'{best_scaler_name.replace(" ", "_")}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(best_scaler, f)
        
        logger.info(f"  ✅ Scaler saved to: {scaler_path}")
        logger.info(f"     This scaler will be used for:")
        logger.info(f"     1. Training phase")
        logger.info(f"     2. Web app prediction (user inputs)")
        logger.info(f"     3. Inference pipeline")
        
        logger.info("\n" + "="*100)
        logger.info("✅ SCALING + PIPELINE COMPLETE")
        logger.info("="*100)
        logger.info(f"\nScaled data shapes:")
        logger.info(f"   • X_train_scaled: {X_train_scaled.shape}")
        logger.info(f"   • X_test_scaled: {X_test_scaled.shape}")
        logger.info(f"   • y_train: {y_train.shape}")
        logger.info(f"   • y_test: {y_test.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, best_scaler, best_scaler_name, results
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in SCALING_AND_PIPELINE_COMPLETE")
        return X_balanced.values, X_balanced.values, y_balanced, y_balanced, None, "No Scaling (FAILED)", {}

# ============================================================================
# SCALING ONLY (WITHOUT TRAIN-TEST SPLIT)
# ============================================================================

def SCALING_ONLY_COMPLETE(X_train_balanced, X_test, y_train_balanced, y_test):
    """
    Complete scaling pipeline for already-split data:
    1. Apply 5 scaling techniques to pre-split train/test data
    2. Evaluate each on test set
    3. Select best scaler
    4. Save scaler with pickle for reuse
    
    Parameters:
    -----------
    X_train_balanced : pd.DataFrame or np.ndarray
        Balanced training features
    X_test : pd.DataFrame or np.ndarray
        Test features (unseen, not balanced)
    y_train_balanced : pd.Series or np.ndarray
        Balanced training target
    y_test : pd.Series or np.ndarray
        Test target
    
    Returns:
    --------
    X_train_scaled : np.ndarray
        Scaled training features
    X_test_scaled : np.ndarray
        Scaled test features
    scaler : sklearn scaler object
        Fitted best scaler (saved to disk)
    best_scaler_name : str
        Name of best scaler
    scaling_results : dict
        Results of all 5 methods
    """
    
    try:
        logger.info("\n" + "="*100)
        logger.info("STEP-5: SCALING (DATA ALREADY SPLIT)")
        logger.info("="*100)
        
        # =====================================================================
        # APPLY 5 SCALING TECHNIQUES
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("APPLYING 5 SCALING TECHNIQUES")
        logger.info("="*80)
        
        feature_names = X_train_balanced.columns.tolist() if hasattr(X_train_balanced, 'columns') else []
        results = {}
        
        # Technique 1: StandardScaler
        X_t1_train, X_t1_test, scaler_t1, method_t1 = technique_1_standard_scaler(
            X_train_balanced.values if hasattr(X_train_balanced, 'values') else X_train_balanced,
            X_test.values if hasattr(X_test, 'values') else X_test,
            feature_names
        )
        auc_t1 = evaluate_scaling_method(X_t1_train, y_train_balanced, X_t1_test, y_test, method_t1)
        results['Standard Scaler'] = {
            'scaler': scaler_t1,
            'X_train': X_t1_train,
            'X_test': X_t1_test,
            'test_auc': auc_t1
        }
        
        # Technique 2: MinMaxScaler
        X_t2_train, X_t2_test, scaler_t2, method_t2 = technique_2_minmax_scaler(
            X_train_balanced.values if hasattr(X_train_balanced, 'values') else X_train_balanced,
            X_test.values if hasattr(X_test, 'values') else X_test,
            feature_names
        )
        auc_t2 = evaluate_scaling_method(X_t2_train, y_train_balanced, X_t2_test, y_test, method_t2)
        results['MinMax Scaler'] = {
            'scaler': scaler_t2,
            'X_train': X_t2_train,
            'X_test': X_t2_test,
            'test_auc': auc_t2
        }
        
        # Technique 3: RobustScaler
        X_t3_train, X_t3_test, scaler_t3, method_t3 = technique_3_robust_scaler(
            X_train_balanced.values if hasattr(X_train_balanced, 'values') else X_train_balanced,
            X_test.values if hasattr(X_test, 'values') else X_test,
            feature_names
        )
        auc_t3 = evaluate_scaling_method(X_t3_train, y_train_balanced, X_t3_test, y_test, method_t3)
        results['Robust Scaler'] = {
            'scaler': scaler_t3,
            'X_train': X_t3_train,
            'X_test': X_t3_test,
            'test_auc': auc_t3
        }
        
        # Technique 4: PowerTransformer
        X_t4_train, X_t4_test, scaler_t4, method_t4 = technique_4_power_transformer(
            X_train_balanced.values if hasattr(X_train_balanced, 'values') else X_train_balanced,
            X_test.values if hasattr(X_test, 'values') else X_test,
            feature_names
        )
        auc_t4 = evaluate_scaling_method(X_t4_train, y_train_balanced, X_t4_test, y_test, method_t4)
        results['Power Transformer'] = {
            'scaler': scaler_t4,
            'X_train': X_t4_train,
            'X_test': X_t4_test,
            'test_auc': auc_t4
        }
        
        # Technique 5: QuantileTransformer
        X_t5_train, X_t5_test, scaler_t5, method_t5 = technique_5_quantile_transformer(
            X_train_balanced.values if hasattr(X_train_balanced, 'values') else X_train_balanced,
            X_test.values if hasattr(X_test, 'values') else X_test,
            feature_names
        )
        auc_t5 = evaluate_scaling_method(X_t5_train, y_train_balanced, X_t5_test, y_test, method_t5)
        results['Quantile Transformer'] = {
            'scaler': scaler_t5,
            'X_train': X_t5_train,
            'X_test': X_t5_test,
            'test_auc': auc_t5
        }
        
        # =====================================================================
        # SELECT BEST SCALER (BY TEST ROC-AUC)
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("COMPARING SCALING TECHNIQUES")
        logger.info("="*80)
        
        best_scaler_name = max(results.keys(), key=lambda x: results[x]['test_auc'])
        best_result = results[best_scaler_name]
        best_scaler = best_result['scaler']
        X_train_scaled = best_result['X_train']
        X_test_scaled = best_result['X_test']
        
        logger.info(f"\n🏆 BEST SCALER: {best_scaler_name}")
        for scaler_name, result in results.items():
            logger.info(f"   • {scaler_name}: {result['test_auc']:.4f}")
        
        # =====================================================================
        # SAVE SCALER WITH PICKLE
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("SAVING SCALER FOR DEPLOYMENT")
        logger.info("="*80)
        
        scaler_path = os.path.join(SCALERS_DIR, f'{best_scaler_name.replace(" ", "_")}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(best_scaler, f)
        
        logger.info(f"  ✅ Scaler saved to: {scaler_path}")
        logger.info(f"     This scaler will be used for:")
        logger.info(f"     1. Training phase")
        logger.info(f"     2. Web app prediction (user inputs)")
        logger.info(f"     3. Inference pipeline")
        
        logger.info("\n" + "="*100)
        logger.info("✅ SCALING COMPLETE")
        logger.info("="*100)
        logger.info(f"\nScaled data shapes:")
        logger.info(f"   • X_train_scaled: {X_train_scaled.shape}")
        logger.info(f"   • X_test_scaled: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, best_scaler, best_scaler_name, results
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in SCALING_ONLY_COMPLETE")
        X_train_vals = X_train_balanced.values if hasattr(X_train_balanced, 'values') else X_train_balanced
        X_test_vals = X_test.values if hasattr(X_test, 'values') else X_test
        return X_train_vals, X_test_vals, None, "No Scaling (FAILED)", {}


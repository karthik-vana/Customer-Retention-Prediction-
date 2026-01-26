"""
STEP-4: DATA BALANCING MODULE
Handles class imbalance using 5 techniques

Techniques:
-----------
1. Random Oversampling
2. Random Undersampling
3. SMOTE
4. ADASYN
5. SMOTEENN

Evaluation: Cross-validation ROC-AUC score
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')
from log_code import get_logger

logger = get_logger('balancing')

# ============================================================================
# BALANCING TECHNIQUES (5)
# ============================================================================

def technique_1_random_oversampling(X, y):
    """
    TECHNIQUE 1: Random Oversampling
    - Randomly duplicates minority class samples
    - Simple but can lead to overfitting
    - Good for small datasets
    """
    try:
        logger.info("⚖️ Technique 1: Random Oversampling")
        
        oversampler = RandomOverSampler(random_state=42)
        X_balanced, y_balanced = oversampler.fit_resample(X, y)
        
        logger.info(f"  • Original distribution: {dict(pd.Series(y).value_counts())}")
        logger.info(f"  • Balanced distribution: {dict(pd.Series(y_balanced).value_counts())}")
        logger.info(f"  • New size: {len(X_balanced)} (from {len(X)})")
        
        return X_balanced, y_balanced, "Random Oversampling"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Random Oversampling")
        return X, y, "Random Oversampling (FAILED)"

def technique_2_random_undersampling(X, y):
    """
    TECHNIQUE 2: Random Undersampling
    - Randomly removes majority class samples
    - Simple but loses data
    - Good for very large datasets with imbalance
    """
    try:
        logger.info("⚖️ Technique 2: Random Undersampling")
        
        undersampler = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = undersampler.fit_resample(X, y)
        
        logger.info(f"  • Original distribution: {dict(pd.Series(y).value_counts())}")
        logger.info(f"  • Balanced distribution: {dict(pd.Series(y_balanced).value_counts())}")
        logger.info(f"  • New size: {len(X_balanced)} (from {len(X)})")
        
        return X_balanced, y_balanced, "Random Undersampling"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Random Undersampling")
        return X, y, "Random Undersampling (FAILED)"

def technique_3_smote(X, y):
    """
    TECHNIQUE 3: SMOTE (Synthetic Minority Oversampling Technique)
    - Creates synthetic samples in feature space
    - Better than random oversampling
    - Creates more realistic synthetic data
    """
    try:
        logger.info("⚖️ Technique 3: SMOTE")
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        logger.info(f"  • Original distribution: {dict(pd.Series(y).value_counts())}")
        logger.info(f"  • Balanced distribution: {dict(pd.Series(y_balanced).value_counts())}")
        logger.info(f"  • New size: {len(X_balanced)} (from {len(X)})")
        logger.info(f"  • Created {len(X_balanced) - len(X)} synthetic samples")
        
        return X_balanced, y_balanced, "SMOTE"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in SMOTE")
        return X, y, "SMOTE (FAILED)"

def technique_4_adasyn(X, y):
    """
    TECHNIQUE 4: ADASYN (Adaptive Synthetic Sampling)
    - Similar to SMOTE but adaptive
    - Focuses on hard-to-learn samples
    - Better for complex decision boundaries
    """
    try:
        logger.info("⚖️ Technique 4: ADASYN")
        
        adasyn = ADASYN(random_state=42, n_neighbors=5)
        X_balanced, y_balanced = adasyn.fit_resample(X, y)
        
        logger.info(f"  • Original distribution: {dict(pd.Series(y).value_counts())}")
        logger.info(f"  • Balanced distribution: {dict(pd.Series(y_balanced).value_counts())}")
        logger.info(f"  • New size: {len(X_balanced)} (from {len(X)})")
        logger.info(f"  • Created {len(X_balanced) - len(X)} synthetic samples (adaptive)")
        
        return X_balanced, y_balanced, "ADASYN"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in ADASYN")
        return X, y, "ADASYN (FAILED)"

def technique_5_smoteenn(X, y):
    """
    TECHNIQUE 5: SMOTEENN (SMOTE + Edited Nearest Neighbors)
    - Combines oversampling (SMOTE) + undersampling (ENN)
    - Over-sample minority, then remove noisy samples
    - Best balance between oversampling and undersampling
    """
    try:
        logger.info("⚖️ Technique 5: SMOTEENN")
        
        smoteenn = SMOTEENN(random_state=42)
        X_balanced, y_balanced = smoteenn.fit_resample(X, y)
        
        logger.info(f"  • Original distribution: {dict(pd.Series(y).value_counts())}")
        logger.info(f"  • Balanced distribution: {dict(pd.Series(y_balanced).value_counts())}")
        logger.info(f"  • New size: {len(X_balanced)} (from {len(X)})")
        
        return X_balanced, y_balanced, "SMOTEENN"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in SMOTEENN")
        return X, y, "SMOTEENN (FAILED)"

# ============================================================================
# EVALUATION USING CROSS-VALIDATION
# ============================================================================

def evaluate_balancing_method(X_balanced, y_balanced, method_name):
    """
    Evaluate balancing method using 5-fold stratified cross-validation ROC-AUC
    
    Returns:
    --------
    cv_scores : array
        Cross-validation scores
    mean_auc : float
        Mean ROC-AUC score
    """
    try:
        logger.info(f"\n  📊 Evaluating {method_name} with 5-Fold Cross-Validation...")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(rf, X_balanced, y_balanced, cv=cv, scoring='roc_auc', n_jobs=1)
        mean_auc = cv_scores.mean()
        std_auc = cv_scores.std()
        
        logger.info(f"    CV ROC-AUC Scores: {[f'{score:.4f}' for score in cv_scores]}")
        logger.info(f"    Mean ROC-AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
        
        return cv_scores, mean_auc
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), f"Failed to evaluate {method_name}")
        return np.array([0.0]), 0.0

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def DATA_BALANCING_COMPLETE(df, target_col='Churn'):
    """
    Complete data balancing pipeline:
    1. Separates features and target
    2. Encodes target to numeric
    3. Applies 5 balancing techniques
    4. Evaluates each using cross-validation ROC-AUC
    5. Selects best method
    
    Returns:
    --------
    X_balanced : pd.DataFrame
        Balanced feature matrix
    y_balanced : pd.Series
        Balanced target vector
    balancing_method : str
        Name of selected method
    evaluation_results : dict
        Results of all 5 methods
    """
    
    try:
        logger.info("\n" + "="*100)
        logger.info("STEP-4: DATA BALANCING - COMPLETE PIPELINE")
        logger.info("="*100)
        
        # =====================================================================
        # PREPARE DATA
        # =====================================================================
        logger.info(f"\n📊 Dataset Info:")
        logger.info(f"   • Total rows: {len(df)}")
        logger.info(f"   • Total columns: {len(df.columns)}")
        
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        logger.info(f"\n📊 Class Distribution (BEFORE balancing):")
        class_counts = pd.Series(y).value_counts()
        for class_val, count in class_counts.items():
            pct = (count / len(y)) * 100
            logger.info(f"   • {class_val}: {count} ({pct:.1f}%)")
        
        # Encode target to numeric
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # =====================================================================
        # APPLY 5 BALANCING TECHNIQUES
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("APPLYING 5 BALANCING TECHNIQUES")
        logger.info("="*80)
        
        results = {}
        
        # Technique 1: Random Oversampling
        X_t1, y_t1, method_t1 = technique_1_random_oversampling(X.values, y_encoded)
        cv_scores_t1, mean_auc_t1 = evaluate_balancing_method(X_t1, y_t1, method_t1)
        results['Random Oversampling'] = {
            'X': X_t1, 'y': y_t1, 'mean_auc': mean_auc_t1, 'cv_scores': cv_scores_t1
        }
        
        # Technique 2: Random Undersampling
        X_t2, y_t2, method_t2 = technique_2_random_undersampling(X.values, y_encoded)
        cv_scores_t2, mean_auc_t2 = evaluate_balancing_method(X_t2, y_t2, method_t2)
        results['Random Undersampling'] = {
            'X': X_t2, 'y': y_t2, 'mean_auc': mean_auc_t2, 'cv_scores': cv_scores_t2
        }
        
        # Technique 3: SMOTE
        X_t3, y_t3, method_t3 = technique_3_smote(X.values, y_encoded)
        cv_scores_t3, mean_auc_t3 = evaluate_balancing_method(X_t3, y_t3, method_t3)
        results['SMOTE'] = {
            'X': X_t3, 'y': y_t3, 'mean_auc': mean_auc_t3, 'cv_scores': cv_scores_t3
        }
        
        # Technique 4: ADASYN
        X_t4, y_t4, method_t4 = technique_4_adasyn(X.values, y_encoded)
        cv_scores_t4, mean_auc_t4 = evaluate_balancing_method(X_t4, y_t4, method_t4)
        results['ADASYN'] = {
            'X': X_t4, 'y': y_t4, 'mean_auc': mean_auc_t4, 'cv_scores': cv_scores_t4
        }
        
        # Technique 5: SMOTEENN
        X_t5, y_t5, method_t5 = technique_5_smoteenn(X.values, y_encoded)
        cv_scores_t5, mean_auc_t5 = evaluate_balancing_method(X_t5, y_t5, method_t5)
        results['SMOTEENN'] = {
            'X': X_t5, 'y': y_t5, 'mean_auc': mean_auc_t5, 'cv_scores': cv_scores_t5
        }
        
        # =====================================================================
        # COMPARE AND SELECT BEST
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("COMPARING ALL TECHNIQUES - ROC-AUC SCORES")
        logger.info("="*80)
        
        comparison = {}
        for method, result in results.items():
            comparison[method] = result['mean_auc']
        
        # Sort by mean AUC
        sorted_methods = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("\n📊 Ranking by Mean ROC-AUC:")
        for rank, (method, auc) in enumerate(sorted_methods, 1):
            logger.info(f"   {rank}. {method}: {auc:.4f}")
        
        # Select best method
        best_method = sorted_methods[0][0]
        best_result = results[best_method]
        X_balanced = best_result['X']
        y_balanced = best_result['y']
        
        logger.info(f"\n✅ DECISION: Using '{best_method}'")
        logger.info(f"   • Mean ROC-AUC: {best_result['mean_auc']:.4f}")
        logger.info(f"   • Reason: Highest cross-validation performance")
        
        # Convert back to DataFrame
        X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
        y_balanced = pd.Series(y_balanced, name=target_col)
        
        logger.info("\n" + "="*100)
        logger.info("✅ DATA BALANCING COMPLETE")
        logger.info("="*100)
        logger.info(f"\nBalanced dataset shape: {X_balanced.shape}")
        logger.info(f"Class distribution:")
        for class_val in np.unique(y_balanced):
            count = (y_balanced == class_val).sum()
            pct = (count / len(y_balanced)) * 100
            logger.info(f"   • Class {class_val}: {count} ({pct:.1f}%)")
        
        return X_balanced, y_balanced, best_method, results
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in DATA_BALANCING_COMPLETE")
        # Return original data as fallback
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X, pd.Series(y_encoded, name=target_col), "No Balancing (FAILED)", {}

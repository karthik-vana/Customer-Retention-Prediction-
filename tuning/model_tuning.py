"""
STEP-6B: HYPERPARAMETER TUNING MODULE
Tunes best model using 3 techniques

Techniques:
-----------
1. GridSearchCV - Exhaustive parameter search
2. RandomizedSearchCV - Random parameter sampling
3. Bayesian Optimization - Smart sequential search

Uses cross-validation to find best parameters
Saves final tuned model with pickle
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

try:
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real, Categorical
    BAYES_AVAILABLE = True
except:
    BAYES_AVAILABLE = False

from log_code import get_logger

logger = get_logger('model_tuning')

# Create models directory if not exists
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# ============================================================================
# HYPERPARAMETER TUNING TECHNIQUES (3)
# ============================================================================

def technique_1_gridsearchcv(model, param_grid, X_train, y_train, cv=5):
    """
    TECHNIQUE 1: GridSearchCV
    - Exhaustive search over all parameter combinations
    - Guaranteed to find best in the grid
    - Computationally expensive for large grids
    - Good for final refinement
    """
    try:
        logger.info("🔍 Technique 1: GridSearchCV")
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=1,
            verbose=1
        )
        
        logger.info(f"  • Searching {len(param_grid)} parameter combinations...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"  • Best parameters: {grid_search.best_params_}")
        logger.info(f"  • Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in GridSearchCV")
        return model, {}, 0.0

def technique_2_randomizedsearchcv(model, param_distributions, X_train, y_train, cv=5, n_iter=20):
    """
    TECHNIQUE 2: RandomizedSearchCV
    - Random sampling from parameter distributions
    - Faster than GridSearch for large spaces
    - Good for exploring hyperparameter space
    - Can find good solutions faster
    """
    try:
        logger.info("🔍 Technique 2: RandomizedSearchCV")
        
        random_search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='roc_auc',
            n_jobs=1,
            random_state=42,
            verbose=1
        )
        
        logger.info(f"  • Sampling {n_iter} random combinations...")
        random_search.fit(X_train, y_train)
        
        logger.info(f"  • Best parameters: {random_search.best_params_}")
        logger.info(f"  • Best CV ROC-AUC: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in RandomizedSearchCV")
        return model, {}, 0.0

def technique_3_bayesian_optimization(model, param_space, X_train, y_train, cv=5, n_calls=30):
    """
    TECHNIQUE 3: Bayesian Optimization
    - Smart sequential search using Gaussian Process
    - Learns from previous iterations
    - Most sample-efficient
    - Best for expensive function evaluations
    
    Note: Requires scikit-optimize package
    """
    try:
        if not BAYES_AVAILABLE:
            logger.warning("  ⚠ scikit-optimize not installed. Skipping Bayesian Optimization.")
            return model, {}, 0.0
        
        logger.info("🔍 Technique 3: Bayesian Optimization")
        
        bayes_search = BayesSearchCV(
            model,
            param_space,
            n_calls=n_calls,
            cv=cv,
            scoring='roc_auc',
            n_jobs=1,
            random_state=42,
            verbose=1
        )
        
        logger.info(f"  • Bayesian optimization with {n_calls} calls...")
        bayes_search.fit(X_train, y_train)
        
        logger.info(f"  • Best parameters: {bayes_search.best_params_}")
        logger.info(f"  • Best CV ROC-AUC: {bayes_search.best_score_:.4f}")
        
        return bayes_search.best_estimator_, bayes_search.best_params_, bayes_search.best_score_
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Bayesian Optimization")
        return model, {}, 0.0

# ============================================================================
# PARAMETER GRIDS FOR DIFFERENT MODELS
# ============================================================================

def get_param_grid_random_forest():
    """Parameter grid for Random Forest"""
    return {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [10, 12, 15, 18, 20],
        'min_samples_split': [5, 8, 10, 12],
        'min_samples_leaf': [2, 4, 5, 6],
        'max_features': ['sqrt', 'log2']
    }

def get_param_distributions_random_forest():
    """Parameter distributions for Random Forest (RandomizedSearch)"""
    return {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [8, 10, 12, 15, 18, 20, 25],
        'min_samples_split': [2, 5, 8, 10, 15],
        'min_samples_leaf': [1, 2, 4, 5, 6, 8],
        'max_features': ['sqrt', 'log2', None]
    }

def get_param_space_random_forest():
    """Parameter space for Bayesian Optimization (Random Forest)"""
    return {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(8, 25),
        'min_samples_split': Integer(2, 15),
        'min_samples_leaf': Integer(1, 8),
        'max_features': Categorical(['sqrt', 'log2', None])
    }

def get_param_grid_xgboost():
    """Parameter grid for XGBoost"""
    return {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 6, 7, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2]
    }

def get_param_distributions_xgboost():
    """Parameter distributions for XGBoost (RandomizedSearch)"""
    return {
        'n_estimators': [50, 100, 150, 200, 250],
        'max_depth': [4, 5, 6, 7, 8, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5]
    }

def get_param_space_xgboost():
    """Parameter space for Bayesian Optimization (XGBoost)"""
    return {
        'n_estimators': Integer(50, 250),
        'max_depth': Integer(4, 9),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'gamma': Real(0, 0.5)
    }

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def HYPERPARAMETER_TUNING_COMPLETE(best_model, best_model_name, X_train_scaled, y_train):
    """
    Complete hyperparameter tuning pipeline:
    1. Determine model type and appropriate parameter space
    2. Apply 3 tuning techniques
    3. Compare results
    4. Select best tuned model
    5. Save tuned model with pickle
    
    Returns:
    --------
    tuned_model : sklearn model object
        Best tuned model
    tuning_results : dict
        Results from all 3 techniques
    """
    
    try:
        logger.info("\n" + "="*100)
        logger.info("STEP-6B: HYPERPARAMETER TUNING - 3 TECHNIQUES")
        logger.info("="*100)
        
        logger.info(f"\n🎯 Tuning Model: {best_model_name}")
        logger.info(f"   • Training set: {X_train_scaled.shape[0]} samples")
        logger.info(f"   • Cross-validation: 5-fold stratified")
        logger.info(f"   • Metric: ROC-AUC")
        
        tuning_results = {}
        
        # =====================================================================
        # DETERMINE PARAMETER SPACE BASED ON MODEL TYPE
        # =====================================================================
        
        if 'Random Forest' in best_model_name:
            param_grid = get_param_grid_random_forest()
            param_distributions = get_param_distributions_random_forest()
            param_space = get_param_space_random_forest()
        
        elif 'XGBoost' in best_model_name:
            param_grid = get_param_grid_xgboost()
            param_distributions = get_param_distributions_xgboost()
            param_space = get_param_space_xgboost()
        
        else:
            logger.warning(f"  ⚠ Model {best_model_name} has no predefined parameter space")
            logger.info("  → Returning original model without tuning")
            return best_model, {}
        
        # =====================================================================
        # APPLY 3 TUNING TECHNIQUES
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("APPLYING 3 TUNING TECHNIQUES")
        logger.info("="*80)
        
        # Technique 1: GridSearchCV
        logger.info("\n" + "-"*60)
        tuned_1, params_1, score_1 = technique_1_gridsearchcv(
            best_model, param_grid, X_train_scaled, y_train, cv=5
        )
        tuning_results['GridSearchCV'] = {
            'model': tuned_1,
            'params': params_1,
            'cv_auc': score_1
        }
        
        # Technique 2: RandomizedSearchCV
        logger.info("\n" + "-"*60)
        tuned_2, params_2, score_2 = technique_2_randomizedsearchcv(
            best_model, param_distributions, X_train_scaled, y_train, cv=5, n_iter=20
        )
        tuning_results['RandomizedSearchCV'] = {
            'model': tuned_2,
            'params': params_2,
            'cv_auc': score_2
        }
        
        # Technique 3: Bayesian Optimization
        logger.info("\n" + "-"*60)
        tuned_3, params_3, score_3 = technique_3_bayesian_optimization(
            best_model, param_space, X_train_scaled, y_train, cv=5, n_calls=30
        )
        if params_3:  # Only if successful
            tuning_results['Bayesian Optimization'] = {
                'model': tuned_3,
                'params': params_3,
                'cv_auc': score_3
            }
        
        # =====================================================================
        # COMPARE AND SELECT BEST TUNED MODEL
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("COMPARING TUNING TECHNIQUES")
        logger.info("="*80)
        
        logger.info("\n📊 Tuning Results:")
        for technique, result in tuning_results.items():
            logger.info(f"\n   {technique}:")
            logger.info(f"      CV ROC-AUC: {result['cv_auc']:.4f}")
            logger.info(f"      Parameters: {result['params']}")
        
        # Select best
        best_technique = max(tuning_results.items(), key=lambda x: x[1]['cv_auc'])
        best_tuned_model = best_technique[1]['model']
        best_technique_name = best_technique[0]
        best_cv_auc = best_technique[1]['cv_auc']
        
        logger.info(f"\n✅ DECISION: Using '{best_technique_name}'")
        logger.info(f"   • CV ROC-AUC: {best_cv_auc:.4f}")
        logger.info(f"   • Reason: Highest cross-validation performance")
        
        # =====================================================================
        # SAVE TUNED MODEL
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("SAVING TUNED MODEL")
        logger.info("="*80)
        
        model_filename = best_model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        model_path = os.path.join(MODELS_DIR, f'{model_filename}_tuned.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(best_tuned_model, f)
        
        logger.info(f"\n  ✅ Tuned model saved to: {model_path}")
        logger.info(f"     This model will be used for:")
        logger.info(f"     1. Prediction on test set")
        logger.info(f"     2. Web app predictions")
        logger.info(f"     3. Production deployment")
        
        logger.info("\n" + "="*100)
        logger.info("✅ HYPERPARAMETER TUNING COMPLETE")
        logger.info("="*100)
        
        return best_tuned_model, tuning_results
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in HYPERPARAMETER_TUNING_COMPLETE")
        logger.info("  → Returning original model without tuning")
        return best_model, {}

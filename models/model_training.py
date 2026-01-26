"""
STEP-6: MODEL TRAINING MODULE
Trains 5 different models and evaluates each comprehensively

Models:
-------
1. Logistic Regression
2. Random Forest
3. XGBoost / Gradient Boosting
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

Evaluation Metrics:
------------------
- ROC-AUC (PRIMARY METRIC)
- ROC Curve
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, confusion_matrix, 
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')
from log_code import get_logger

logger = get_logger('model_training')

# ============================================================================
# MODEL DEFINITIONS (5)
# ============================================================================

def model_1_logistic_regression(X_train, y_train, X_test, y_test):
    """
    MODEL 1: Logistic Regression
    - Linear classifier
    - Fast training
    - Interpretable
    - Good baseline
    """
    try:
        logger.info("🤖 Model 1: Logistic Regression")
        
        model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        logger.info(f"  • Model trained successfully")
        
        return model, y_pred, y_pred_proba, "Logistic Regression"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Logistic Regression")
        return None, None, None, "Logistic Regression (FAILED)"

def model_2_random_forest(X_train, y_train, X_test, y_test):
    """
    MODEL 2: Random Forest
    - Ensemble of decision trees
    - Handles non-linearity well
    - Robust to outliers
    - Provides feature importance
    """
    try:
        logger.info("🤖 Model 2: Random Forest")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        logger.info(f"  • Model trained with {model.n_estimators} trees")
        logger.info(f"  • Max depth: {model.max_depth}")
        
        return model, y_pred, y_pred_proba, "Random Forest"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in Random Forest")
        return None, None, None, "Random Forest (FAILED)"

def model_3_xgboost(X_train, y_train, X_test, y_test):
    """
    MODEL 3: XGBoost
    - Gradient boosting framework
    - Excellent performance
    - Handles large datasets efficiently
    - Regularization built-in
    """
    try:
        logger.info("🤖 Model 3: XGBoost")
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        model.fit(X_train, y_train, verbose=False)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        logger.info(f"  • Model trained with {model.n_estimators} boosting rounds")
        logger.info(f"  • Learning rate: {model.learning_rate}")
        
        return model, y_pred, y_pred_proba, "XGBoost"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in XGBoost")
        return None, None, None, "XGBoost (FAILED)"

def model_4_support_vector_machine(X_train, y_train, X_test, y_test):
    """
    MODEL 4: Support Vector Machine (SVM)
    - RBF kernel for non-linear boundaries
    - Good for high-dimensional data
    - Robust to outliers
    - Probability estimates enabled
    """
    try:
        logger.info("🤖 Model 4: Support Vector Machine (SVM)")
        
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,  # Enable probability estimates
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        logger.info(f"  • Model trained with RBF kernel")
        logger.info(f"  • Support vectors: {model.n_support_[0] + model.n_support_[1]}")
        
        return model, y_pred, y_pred_proba, "SVM"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in SVM")
        return None, None, None, "SVM (FAILED)"

def model_5_knn(X_train, y_train, X_test, y_test):
    """
    MODEL 5: K-Nearest Neighbors (KNN)
    - Non-parametric instance-based learner
    - Simple and interpretable
    - Can be slow for large datasets
    - k=5 typically works well
    """
    try:
        logger.info("🤖 Model 5: K-Nearest Neighbors (KNN)")
        
        model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        logger.info(f"  • Model trained with k=5 neighbors")
        logger.info(f"  • Distance weighted")
        
        return model, y_pred, y_pred_proba, "KNN"
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in KNN")
        return None, None, None, "KNN (FAILED)"

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(y_test, y_pred, y_pred_proba, model_name):
    """
    Comprehensive model evaluation
    
    Metrics:
    - ROC-AUC (primary)
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - Confusion Matrix
    - ROC Curve
    """
    try:
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        logger.info(f"\n  📊 {model_name} - Evaluation Results:")
        logger.info(f"     ROC-AUC:  {roc_auc:.4f} ⭐ (Primary Metric)")
        logger.info(f"     Accuracy: {accuracy:.4f}")
        logger.info(f"     Precision: {precision:.4f}")
        logger.info(f"     Recall:   {recall:.4f}")
        logger.info(f"     F1-Score: {f1:.4f}")
        
        # Confusion Matrix
        tn, fp, fn, tp = cm.ravel()
        logger.info(f"     Confusion Matrix:")
        logger.info(f"       TN: {tn} | FP: {fp}")
        logger.info(f"       FN: {fn} | TP: {tp}")
        
        return {
            'model_name': model_name,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr
        }
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), f"Failed to evaluate {model_name}")
        return None

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_roc_curves(evaluation_results):
    """
    Plot ROC curves for all models
    """
    try:
        plt.figure(figsize=(10, 8))
        
        for result in evaluation_results:
            if result is not None:
                fpr = result['fpr']
                tpr = result['tpr']
                auc_score = result['roc_auc']
                model_name = result['model_name']
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.4f})', linewidth=2)
        
        # Diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('models/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("\n  ✅ ROC Curves saved to: models/roc_curves_comparison.png")
        plt.close()
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed to plot ROC curves")

def plot_model_comparison(evaluation_results):
    """
    Plot model comparison bar charts
    """
    try:
        metrics = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']
        model_names = [r['model_name'] for r in evaluation_results if r is not None]
        
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        
        for idx, metric in enumerate(metrics):
            values = [r[metric] for r in evaluation_results if r is not None]
            colors = ['red' if v < 0.7 else 'orange' if v < 0.8 else 'green' for v in values]
            
            axes[idx].bar(range(len(values)), values, color=colors, edgecolor='black', linewidth=1.5)
            axes[idx].set_title(metric.replace('_', ' ').upper(), fontweight='bold')
            axes[idx].set_ylim([0, 1])
            axes[idx].set_xticks(range(len(model_names)))
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("  ✅ Model comparison saved to: models/model_comparison.png")
        plt.close()
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed to plot model comparison")

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def MODEL_TRAINING_COMPLETE(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Complete model training pipeline:
    1. Train 5 different models
    2. Evaluate each comprehensively
    3. Generate comparison visualizations
    4. Select best model based on ROC-AUC
    
    Returns:
    --------
    best_model : sklearn model object
        Best performing model
    best_model_name : str
        Name of best model
    evaluation_results : list
        List of evaluation dictionaries for all models
    """
    
    try:
        logger.info("\n" + "="*100)
        logger.info("STEP-6: MODEL TRAINING - 5 MODELS COMPARISON")
        logger.info("="*100)
        
        logger.info(f"\n📊 Dataset Info:")
        logger.info(f"   • Training set: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
        logger.info(f"   • Test set: {X_test_scaled.shape[0]} samples")
        logger.info(f"   • Target distribution - Train: {dict(pd.Series(y_train).value_counts())}")
        logger.info(f"   • Target distribution - Test: {dict(pd.Series(y_test).value_counts())}")
        
        # =====================================================================
        # TRAIN 5 MODELS
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("TRAINING 5 MODELS")
        logger.info("="*80)
        
        models = {}
        evaluation_results = []
        
        # Model 1: Logistic Regression
        model_1, pred_1, proba_1, name_1 = model_1_logistic_regression(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        if model_1 is not None:
            result_1 = evaluate_model(y_test, pred_1, proba_1, name_1)
            if result_1 is not None:
                models[name_1] = model_1
                evaluation_results.append(result_1)
        
        # Model 2: Random Forest
        model_2, pred_2, proba_2, name_2 = model_2_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        if model_2 is not None:
            result_2 = evaluate_model(y_test, pred_2, proba_2, name_2)
            if result_2 is not None:
                models[name_2] = model_2
                evaluation_results.append(result_2)
        
        # Model 3: XGBoost
        model_3, pred_3, proba_3, name_3 = model_3_xgboost(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        if model_3 is not None:
            result_3 = evaluate_model(y_test, pred_3, proba_3, name_3)
            if result_3 is not None:
                models[name_3] = model_3
                evaluation_results.append(result_3)
        
        # Model 4: SVM
        model_4, pred_4, proba_4, name_4 = model_4_support_vector_machine(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        if model_4 is not None:
            result_4 = evaluate_model(y_test, pred_4, proba_4, name_4)
            if result_4 is not None:
                models[name_4] = model_4
                evaluation_results.append(result_4)
        
        # Model 5: KNN
        model_5, pred_5, proba_5, name_5 = model_5_knn(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        if model_5 is not None:
            result_5 = evaluate_model(y_test, pred_5, proba_5, name_5)
            if result_5 is not None:
                models[name_5] = model_5
                evaluation_results.append(result_5)
        
        # =====================================================================
        # COMPARE AND SELECT BEST
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("MODEL RANKING BY ROC-AUC (PRIMARY METRIC)")
        logger.info("="*80)
        
        # Sort by ROC-AUC
        evaluation_results_sorted = sorted(evaluation_results, key=lambda x: x['roc_auc'], reverse=True)
        
        logger.info("\n📊 Final Rankings:")
        for rank, result in enumerate(evaluation_results_sorted, 1):
            logger.info(f"\n   {rank}. {result['model_name']}")
            logger.info(f"      ROC-AUC:  {result['roc_auc']:.4f}")
            logger.info(f"      Accuracy: {result['accuracy']:.4f}")
            logger.info(f"      F1-Score: {result['f1_score']:.4f}")
        
        # Select best
        best_result = evaluation_results_sorted[0]
        best_model_name = best_result['model_name']
        best_model = models[best_model_name]
        
        logger.info(f"\n✅ DECISION: Selected '{best_model_name}'")
        logger.info(f"   • ROC-AUC: {best_result['roc_auc']:.4f} (Highest)")
        logger.info(f"   • Reason: Best ROC-AUC score (primary metric for imbalanced data)")
        
        # =====================================================================
        # VISUALIZATIONS
        # =====================================================================
        logger.info("\n" + "="*80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*80)
        
        plot_roc_curves(evaluation_results)
        plot_model_comparison(evaluation_results)
        
        logger.info("\n" + "="*100)
        logger.info("✅ MODEL TRAINING COMPLETE")
        logger.info("="*100)
        
        return best_model, best_model_name, evaluation_results
    
    except Exception as e:
        log_error(logger, type(e).__name__, str(e), "Failed in MODEL_TRAINING_COMPLETE")
        return None, "No Model (FAILED)", []

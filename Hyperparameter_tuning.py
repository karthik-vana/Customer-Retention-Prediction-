"""
Hyperparameter_tuning.py - Hyperparameter Tuning Module
========================================
Beginner-friendly hyperparameter optimization

Classes:
- HYPERPARAMETER_TUNING: Fine-tune model parameters
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

class HYPERPARAMETER_TUNING:
    """
    Fine-tune model hyperparameters to improve performance
    
    Concepts:
    - Hyperparameters: Settings we choose (learning rate, tree depth, etc.)
    - Parameters: Learned from data during training (weights, coefficients)
    
    Methods:
    - Grid Search: Try all combinations (exhaustive)
    - Random Search: Try random combinations (faster)
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize with training and test data
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_models = {}
        print("\n[HYPERPARAMETER TUNING INITIALIZED]")
    
    def tune_random_forest_grid(self):
        """
        Fine-tune Random Forest using Grid Search
        
        Parameters tested:
        - n_estimators: Number of trees (10, 50, 100, 200)
        - max_depth: Maximum tree depth (5, 10, 15, None)
        - min_samples_split: Minimum samples to split (2, 5, 10)
        - min_samples_leaf: Minimum samples at leaf (1, 2, 4)
        
        Returns:
            RandomForestClassifier: Best tuned model
        """
        print("\n[Tuning Random Forest - Grid Search]")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        print(f"  Parameter Grid:")
        print(f"    - n_estimators: {param_grid['n_estimators']}")
        print(f"    - max_depth: {param_grid['max_depth']}")
        print(f"    - min_samples_split: {param_grid['min_samples_split']}")
        print(f"    - min_samples_leaf: {param_grid['min_samples_leaf']}")
        
        # Create base model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search
        print("\n  Searching through combinations...")
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=5,  # 5-fold cross-validation
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Best parameters
        print(f"\n  Best Parameters Found:")
        for param, value in grid_search.best_params_.items():
            print(f"    {param}: {value}")
        
        print(f"\n  Best CV Score: {grid_search.best_score_:.4f}")
        
        # Test best model
        best_model = grid_search.best_estimator_
        test_score = best_model.score(self.X_test, self.y_test)
        print(f"  Test Accuracy: {test_score:.4f}")
        
        self.best_models['RandomForest_Grid'] = best_model
        return best_model
    
    def tune_random_forest_random(self, n_iter=20):
        """
        Fine-tune Random Forest using Random Search
        
        What it does:
        - Randomly samples from parameter space
        - Faster than grid search
        - Good for large parameter spaces
        
        Args:
            n_iter: Number of random combinations to try
            
        Returns:
            RandomForestClassifier: Best tuned model
        """
        print(f"\n[Tuning Random Forest - Random Search ({n_iter} iterations)]")
        
        # Define parameter distribution
        param_dist = {
            'n_estimators': [50, 100, 150, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        print(f"  Searching {n_iter} random combinations...")
        
        # Create base model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Random search
        random_search = RandomizedSearchCV(
            rf,
            param_dist,
            n_iter=n_iter,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        # Best parameters
        print(f"\n  Best Parameters Found:")
        for param, value in random_search.best_params_.items():
            print(f"    {param}: {value}")
        
        print(f"\n  Best CV Score: {random_search.best_score_:.4f}")
        
        # Test best model
        best_model = random_search.best_estimator_
        test_score = best_model.score(self.X_test, self.y_test)
        print(f"  Test Accuracy: {test_score:.4f}")
        
        self.best_models['RandomForest_Random'] = best_model
        return best_model
    
    def tune_gradient_boosting(self, n_iter=15):
        """
        Fine-tune Gradient Boosting using Random Search
        
        Parameters tested:
        - n_estimators: Number of boosting stages
        - learning_rate: How fast model learns
        - max_depth: Maximum tree depth
        - subsample: Fraction of samples for training
        
        Args:
            n_iter: Number of random combinations to try
            
        Returns:
            GradientBoostingClassifier: Best tuned model
        """
        print(f"\n[Tuning Gradient Boosting - Random Search ({n_iter} iterations)]")
        
        # Define parameter distribution
        param_dist = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [2, 3, 4, 5],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5]
        }
        
        print(f"  Searching {n_iter} random combinations...")
        
        # Create base model
        gb = GradientBoostingClassifier(random_state=42)
        
        # Random search
        random_search = RandomizedSearchCV(
            gb,
            param_dist,
            n_iter=n_iter,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        # Best parameters
        print(f"\n  Best Parameters Found:")
        for param, value in random_search.best_params_.items():
            print(f"    {param}: {value}")
        
        print(f"\n  Best CV Score: {random_search.best_score_:.4f}")
        
        # Test best model
        best_model = random_search.best_estimator_
        test_score = best_model.score(self.X_test, self.y_test)
        print(f"  Test Accuracy: {test_score:.4f}")
        
        self.best_models['GradientBoosting'] = best_model
        return best_model
    
    def explain_hyperparameters(self):
        """
        Explain what each hyperparameter does
        
        Educational output explaining:
        - What each parameter controls
        - How it affects model behavior
        - Trade-offs to consider
        """
        print("\n[HYPERPARAMETER EXPLANATIONS]")
        print("=" * 60)
        
        explanations = {
            'n_estimators': {
                'what': 'Number of trees to create',
                'effect': 'More trees = potentially better accuracy but slower',
                'tradeoff': 'Balance between accuracy and computation time'
            },
            'max_depth': {
                'what': 'Maximum depth of each tree',
                'effect': 'Deeper trees = more complex patterns, risk of overfitting',
                'tradeoff': 'More depth captures patterns but may overfit'
            },
            'min_samples_split': {
                'what': 'Minimum samples needed to split a node',
                'effect': 'Lower values = more splits, higher = less splits',
                'tradeoff': 'Higher value prevents overfitting but limits learning'
            },
            'min_samples_leaf': {
                'what': 'Minimum samples required at leaf node',
                'effect': 'Controls how pure/specific leaf predictions can be',
                'tradeoff': 'Higher value smooths predictions, prevents overfitting'
            },
            'learning_rate': {
                'what': 'How much each tree contributes to final prediction',
                'effect': 'Lower values = slower learning but more stable',
                'tradeoff': 'Low rate needs more trees, high rate learns faster'
            }
        }
        
        for param, details in explanations.items():
            print(f"\n{param.upper()}")
            print(f"  What: {details['what']}")
            print(f"  Effect: {details['effect']}")
            print(f"  Trade-off: {details['tradeoff']}")
    
    def compare_baseline_vs_tuned(self, baseline_model, tuned_model, model_name='Model'):
        """
        Compare baseline vs tuned model performance
        
        Args:
            baseline_model: Original untuned model
            tuned_model: Tuned model
            model_name: Name of the model for display
            
        Returns:
            dict: Comparison results
        """
        print(f"\n[COMPARISON: Baseline vs Tuned {model_name}]")
        print("=" * 60)
        
        # Evaluate baseline
        baseline_score = baseline_model.score(self.X_test, self.y_test)
        
        # Evaluate tuned
        tuned_score = tuned_model.score(self.X_test, self.y_test)
        
        # Calculate improvement
        improvement = tuned_score - baseline_score
        improvement_pct = (improvement / baseline_score) * 100
        
        print(f"\nBaseline Accuracy: {baseline_score:.4f}")
        print(f"Tuned Accuracy:    {tuned_score:.4f}")
        print(f"Improvement:       {improvement:.4f} ({improvement_pct:.2f}%)")
        
        if improvement > 0:
            print(f"\n[OK] Tuning improved model by {improvement_pct:.2f}%")
        elif improvement < 0:
            print(f"\n[WARNING] Tuning decreased model by {abs(improvement_pct):.2f}%")
        else:
            print(f"\n= No significant change from tuning")
        
        return {
            'baseline_score': baseline_score,
            'tuned_score': tuned_score,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }
    
    def get_best_model(self):
        """
        Return the best tuned model
        
        Returns:
            tuple: (model_name, model_object)
        """
        if self.best_models:
            return list(self.best_models.items())[0]
        else:
            return None, None

"""
Model_Training.py - Model Training Module
========================================
Beginner-friendly training of multiple ML models

Classes:
- MODEL_TRAINING: Train and evaluate various models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

class MODEL_TRAINING:
    """
    Train multiple classification models
    
    Models:
    1. Random Forest - Ensemble of decision trees
    2. Logistic Regression - Linear probability model
    3. Gradient Boosting - Sequential ensemble
    4. SVM - Support Vector Machine
    5. KNN - k-Nearest Neighbors
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize with training and test data
        
        Args:
            X_train: Training features (prepared & balanced)
            X_test: Test features (scaled only)
            y_train: Training target
            y_test: Test target
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
        print("\n[MODEL TRAINING INITIALIZED]")
    
    def train_random_forest(self, n_estimators=100):
        """
        Train Random Forest model
        
        What it does:
        - Creates multiple decision trees
        - Each tree votes on the prediction
        - Combines votes for final prediction
        
        Args:
            n_estimators: Number of trees to create
            
        Returns:
            RandomForestClassifier: Trained model
        """
        print("\n[Training Random Forest Classifier]")
        print(f"  Configuration:")
        print(f"    - Trees: {n_estimators}")
        print(f"    - Max Depth: 15")
        print(f"    - Min Samples: 2")
        
        # Create and train model
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        print("  Training...")
        rf_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = rf_model.predict(self.X_test)
        y_pred_proba = rf_model.predict_proba(self.X_test)[:, 1]
        
        # Store results
        self.models['RandomForest'] = rf_model
        self._evaluate_model('RandomForest', y_pred, y_pred_proba)
        
        print("  [OK] Random Forest training complete")
        return rf_model
    
    def train_logistic_regression(self, max_iter=1000):
        """
        Train Logistic Regression model
        
        What it does:
        - Linear model for binary classification
        - Uses probabilities and weighted features
        - Fast and interpretable
        
        Args:
            max_iter: Maximum iterations for convergence
            
        Returns:
            LogisticRegression: Trained model
        """
        print("\n[Training Logistic Regression]")
        print(f"  Configuration:")
        print(f"    - Solver: lbfgs (good for small-medium data)")
        print(f"    - Max Iterations: {max_iter}")
        
        # Create and train model
        lr_model = LogisticRegression(
            max_iter=max_iter,
            random_state=42,
            solver='lbfgs'
        )
        
        print("  Training...")
        lr_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = lr_model.predict(self.X_test)
        y_pred_proba = lr_model.predict_proba(self.X_test)[:, 1]
        
        # Store results
        self.models['LogisticRegression'] = lr_model
        self._evaluate_model('LogisticRegression', y_pred, y_pred_proba)
        
        print("  [OK] Logistic Regression training complete")
        return lr_model
    
    def train_gradient_boosting(self, n_estimators=100):
        """
        Train Gradient Boosting model
        
        What it does:
        - Sequential ensemble building
        - Each tree corrects previous tree's errors
        - Often very accurate
        
        Args:
            n_estimators: Number of boosting stages
            
        Returns:
            GradientBoostingClassifier: Trained model
        """
        print("\n[Training Gradient Boosting Classifier]")
        print(f"  Configuration:")
        print(f"    - Trees: {n_estimators}")
        print(f"    - Learning Rate: 0.1")
        print(f"    - Max Depth: 3")
        
        # Create and train model
        gb_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        print("  Training...")
        gb_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = gb_model.predict(self.X_test)
        y_pred_proba = gb_model.predict_proba(self.X_test)[:, 1]
        
        # Store results
        self.models['GradientBoosting'] = gb_model
        self._evaluate_model('GradientBoosting', y_pred, y_pred_proba)
        
        print("  [OK] Gradient Boosting training complete")
        return gb_model
    
    def train_svm(self, kernel='rbf'):
        """
        Train Support Vector Machine
        
        What it does:
        - Finds optimal boundary between classes
        - Works well for high-dimensional data
        - Can be slow on large datasets
        
        Args:
            kernel: Type of kernel (rbf, linear, poly)
            
        Returns:
            SVC: Trained SVM model
        """
        print("\n[Training Support Vector Machine (SVM)]")
        print(f"  Configuration:")
        print(f"    - Kernel: {kernel}")
        print(f"    - C (regularization): 1.0")
        
        # Create and train model
        svm_model = SVC(
            kernel=kernel,
            C=1.0,
            probability=True,
            random_state=42
        )
        
        print("  Training...")
        svm_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = svm_model.predict(self.X_test)
        y_pred_proba = svm_model.predict_proba(self.X_test)[:, 1]
        
        # Store results
        self.models['SVM'] = svm_model
        self._evaluate_model('SVM', y_pred, y_pred_proba)
        
        print("  [OK] SVM training complete")
        return svm_model
    
    def train_knn(self, n_neighbors=5):
        """
        Train k-Nearest Neighbors model
        
        What it does:
        - Classifies based on nearest neighbors
        - Simple and intuitive
        - Can be slow on large datasets
        
        Args:
            n_neighbors: Number of neighbors to consider
            
        Returns:
            KNeighborsClassifier: Trained KNN model
        """
        print("\n[Training k-Nearest Neighbors (KNN)]")
        print(f"  Configuration:")
        print(f"    - Neighbors (k): {n_neighbors}")
        print(f"    - Weight: distance")
        
        # Create and train model
        knn_model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance'
        )
        
        print("  Training...")
        knn_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = knn_model.predict(self.X_test)
        y_pred_proba = knn_model.predict_proba(self.X_test)[:, 1]
        
        # Store results
        self.models['KNN'] = knn_model
        self._evaluate_model('KNN', y_pred, y_pred_proba)
        
        print("  [OK] KNN training complete")
        return knn_model
    
    def _evaluate_model(self, model_name, y_pred, y_pred_proba):
        """
        Evaluate model performance
        
        Metrics calculated:
        - Accuracy: Correct predictions / Total
        - Precision: True positives / (True + False positives)
        - Recall: True positives / (True + False negatives)
        - F1: Harmonic mean of Precision & Recall
        - ROC-AUC: Area under ROC curve
        
        Args:
            model_name: Name of the model
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
        """
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Print evaluation
        print(f"\n  Evaluation Metrics:")
        print(f"    Accuracy:  {accuracy:.4f} (Correct predictions)")
        print(f"    Precision: {precision:.4f} (Accurate positive predictions)")
        print(f"    Recall:    {recall:.4f} (Catches positive cases)")
        print(f"    F1-Score:  {f1:.4f} (Balance of Precision & Recall)")
        print(f"    ROC-AUC:   {roc_auc:.4f} (Overall discriminability)")
    
    def train_all_models(self):
        """
        Train all available models
        
        Trains:
        1. Random Forest
        2. Logistic Regression
        3. Gradient Boosting
        4. SVM
        5. KNN
        """
        print("\n[TRAINING ALL MODELS]")
        print("=" * 50)
        
        self.train_random_forest()
        self.train_logistic_regression()
        self.train_gradient_boosting()
        self.train_svm()
        self.train_knn()
        
        self.print_model_comparison()
    
    def print_model_comparison(self):
        """
        Print comparison of all trained models
        
        Shows:
        - Performance metrics for each model
        - Ranked by F1-score
        """
        print("\n[MODEL COMPARISON]")
        print("=" * 70)
        
        # Sort by F1-score
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['f1'],
            reverse=True
        )
        
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("-" * 70)
        
        for model_name, metrics in sorted_models:
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} "
                  f"{metrics['f1']:<12.4f} "
                  f"{metrics['roc_auc']:<12.4f}")
        
        print("\n[OK] Best Model: " + sorted_models[0][0] + f" (F1: {sorted_models[0][1]['f1']:.4f})")
    
    def get_best_model(self):
        """
        Return best performing model
        
        Returns:
            tuple: (best_model_name, model_object)
        """
        best_model = max(self.results.items(), key=lambda x: x[1]['f1'])
        return best_model[0], self.models[best_model[0]]
    
    def get_all_models(self):
        """
        Return all trained models
        
        Returns:
            dict: All trained models
        """
        return self.models

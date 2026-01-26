"""
Prediction_Export.py - Prediction & Export Module
========================================
Beginner-friendly final predictions and result export

Classes:
- FINAL_PREDICTIONS: Make predictions on test set and export results
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, auc)
import json
import warnings
warnings.filterwarnings('ignore')

class FINAL_PREDICTIONS:
    """
    Generate final predictions and export results
    
    Features:
    - Make predictions on test set
    - Generate detailed reports
    - Export predictions to CSV
    - Visualize results
    """
    
    def __init__(self, model, X_test, y_test, feature_names=None):
        """
        Initialize with trained model and test data
        
        Args:
            model: Trained ML model with predict() method
            X_test: Test features
            y_test: Test target
            feature_names: Names of features (for interpretability)
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or list(X_test.columns)
        
        self.predictions = None
        self.probabilities = None
        self.results_df = None
        print("\n[FINAL PREDICTIONS INITIALIZED]")
    
    def make_predictions(self):
        """
        Make predictions on test set
        
        Returns:
            tuple: (predictions, probabilities)
        """
        print("\n[Making Final Predictions]")
        
        # Get predictions
        self.predictions = self.model.predict(self.X_test)
        
        # Get prediction probabilities (if available)
        try:
            self.probabilities = self.model.predict_proba(self.X_test)[:, 1]
            print("  [OK] Predictions with probabilities obtained")
        except:
            self.probabilities = None
            print("  [OK] Predictions obtained (probabilities not available)")
        
        return self.predictions, self.probabilities
    
    def generate_classification_report(self):
        """
        Generate detailed classification report
        
        Includes:
        - Precision, Recall, F1 for each class
        - Macro and weighted averages
        - Support (number of samples)
        
        Returns:
            str: Classification report
        """
        if self.predictions is None:
            self.make_predictions()
        
        print("\n[Classification Report]")
        print("=" * 70)
        
        report = classification_report(
            self.y_test,
            self.predictions,
            target_names=['Not Churned', 'Churned'],
            digits=4
        )
        
        print(report)
        return report
    
    def generate_confusion_matrix(self):
        """
        Generate confusion matrix
        
        Confusion Matrix:
        - True Negatives: Correctly predicted not churned
        - False Positives: Incorrectly predicted as churned
        - False Negatives: Missed churning customers
        - True Positives: Correctly predicted churned
        
        Returns:
            array: 2x2 confusion matrix
        """
        if self.predictions is None:
            self.make_predictions()
        
        print("\n[Confusion Matrix]")
        print("=" * 50)
        
        cm = confusion_matrix(self.y_test, self.predictions)
        
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n                 Predicted")
        print(f"                Not Churned  Churned")
        print(f"Actual Not Churned    {tn:5d}      {fp:5d}")
        print(f"       Churned        {fn:5d}      {tp:5d}")
        
        print(f"\nInterpretation:")
        print(f"  True Negatives (TN):  {tn} - Correct 'Not Churned' predictions")
        print(f"  False Positives (FP): {fp} - Wrong 'Churned' predictions")
        print(f"  False Negatives (FN): {fn} - Missed 'Churned' cases")
        print(f"  True Positives (TP):  {tp} - Correct 'Churned' predictions")
        
        return cm
    
    def calculate_detailed_metrics(self):
        """
        Calculate detailed evaluation metrics
        
        Metrics:
        - Accuracy: Overall correctness
        - Precision: Accuracy of positive predictions
        - Recall: Ability to find positives
        - F1: Balance between Precision and Recall
        - ROC-AUC: Overall discriminability
        - Specificity: Ability to find negatives
        
        Returns:
            dict: All metrics
        """
        if self.predictions is None:
            self.make_predictions()
        
        print("\n[Detailed Evaluation Metrics]")
        print("=" * 70)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions)
        recall = recall_score(self.y_test, self.predictions)
        f1 = f1_score(self.y_test, self.predictions)
        
        # Confusion matrix for specificity
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.predictions).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC-AUC if probabilities available
        roc_auc = None
        if self.probabilities is not None:
            roc_auc = roc_auc_score(self.y_test, self.probabilities)
        
        # Print metrics
        print(f"\n1. ACCURACY: {accuracy:.4f}")
        print(f"   What: Percentage of correct predictions")
        print(f"   Formula: (TP + TN) / Total")
        print(f"   Meaning: {accuracy * 100:.2f}% of all predictions were correct")
        
        print(f"\n2. PRECISION: {precision:.4f}")
        print(f"   What: Accuracy of 'Churned' predictions")
        print(f"   Formula: TP / (TP + FP)")
        print(f"   Meaning: {precision * 100:.2f}% of predicted churned were actually churned")
        
        print(f"\n3. RECALL: {recall:.4f}")
        print(f"   What: Ability to find actual churned customers")
        print(f"   Formula: TP / (TP + FN)")
        print(f"   Meaning: Caught {recall * 100:.2f}% of actual churned customers")
        
        print(f"\n4. F1-SCORE: {f1:.4f}")
        print(f"   What: Balance between Precision and Recall")
        print(f"   Formula: 2 * (Precision * Recall) / (Precision + Recall)")
        print(f"   Meaning: Balanced metric = {f1:.4f}")
        
        print(f"\n5. SPECIFICITY: {specificity:.4f}")
        print(f"   What: Ability to correctly identify non-churned")
        print(f"   Formula: TN / (TN + FP)")
        print(f"   Meaning: {specificity * 100:.2f}% of non-churned correctly identified")
        
        if roc_auc:
            print(f"\n6. ROC-AUC: {roc_auc:.4f}")
            print(f"   What: Overall discriminability (0.5-1.0)")
            print(f"   Meaning: Model's ability to distinguish classes = {roc_auc:.4f}")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'roc_auc': roc_auc
        }
        
        return metrics
    
    def export_predictions_csv(self, filename='predictions.csv'):
        """
        Export predictions to CSV file
        
        Args:
            filename: Output CSV filename
            
        Returns:
            str: Path to saved file
        """
        if self.predictions is None:
            self.make_predictions()
        
        print(f"\n[Exporting Predictions to {filename}]")
        
        # Create results dataframe
        results = pd.DataFrame({
            'Actual': self.y_test.values,
            'Predicted': self.predictions,
            'Correct': self.predictions == self.y_test.values
        })
        
        # Add probabilities if available
        if self.probabilities is not None:
            results['Probability_Churned'] = self.probabilities
        
        # Save to CSV
        results.to_csv(filename, index=False)
        
        print(f"  [OK] Predictions exported to {filename}")
        print(f"  [OK] Total predictions: {len(results)}")
        print(f"  [OK] Correct predictions: {results['Correct'].sum()}")
        print(f"  [OK] Accuracy: {results['Correct'].sum() / len(results):.4f}")
        
        return filename
    
    def feature_importance(self):
        """
        Extract and display feature importance
        
        Shows which features were most important for predictions
        
        Returns:
            pd.DataFrame: Features ranked by importance
        """
        print("\n[Feature Importance]")
        
        try:
            # Get feature importances from model
            importances = self.model.feature_importances_
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print("=" * 50)
            for idx, row in importance_df.head(10).iterrows():
                bar_length = int(row['Importance'] * 50)
                bar = '█' * bar_length
                print(f"{row['Feature']:<25} {row['Importance']:.4f}  {bar}")
            
            return importance_df
        
        except AttributeError:
            print("  [WARNING] Model does not support feature importance")
            return None
    
    def prediction_summary(self):
        """
        Print summary of predictions
        
        Returns:
            dict: Summary statistics
        """
        if self.predictions is None:
            self.make_predictions()
        
        print("\n[Prediction Summary]")
        print("=" * 50)
        
        correct = (self.predictions == self.y_test.values).sum()
        total = len(self.y_test)
        accuracy = correct / total
        
        not_churned_pred = (self.predictions == 0).sum()
        churned_pred = (self.predictions == 1).sum()
        
        print(f"\nTotal Test Samples: {total}")
        print(f"Correct Predictions: {correct}")
        print(f"Incorrect Predictions: {total - correct}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        
        print(f"\nPrediction Distribution:")
        print(f"  Not Churned: {not_churned_pred} ({not_churned_pred/total*100:.2f}%)")
        print(f"  Churned: {churned_pred} ({churned_pred/total*100:.2f}%)")
        
        actual_not_churned = (self.y_test.values == 0).sum()
        actual_churned = (self.y_test.values == 1).sum()
        
        print(f"\nActual Distribution:")
        print(f"  Not Churned: {actual_not_churned} ({actual_not_churned/total*100:.2f}%)")
        print(f"  Churned: {actual_churned} ({actual_churned/total*100:.2f}%)")
        
        summary = {
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'not_churned_predictions': not_churned_pred,
            'churned_predictions': churned_pred
        }
        
        return summary

"""
Prediction_pipeline.py - Production Prediction Pipeline
========================================
Beginner-friendly production-ready prediction pipeline

Classes:
- PredictionPipeline: Ready-to-use prediction for new data
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class PredictionPipeline:
    """
    Production-ready pipeline for making predictions on new data
    
    Workflow:
    1. Load trained model and preprocessing objects
    2. Prepare new data (scale, encode, etc.)
    3. Make predictions
    4. Format results for deployment
    """
    
    def __init__(self, model=None, scaler=None, feature_names=None):
        """
        Initialize production pipeline
        
        Args:
            model: Trained ML model
            scaler: Fitted scaler for preprocessing
            feature_names: List of feature names in correct order
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.label_encoders = {}
        self.categorical_mappings = {}
        
        print("\n[PREDICTION PIPELINE INITIALIZED]")
    
    def load_model(self, model_path):
        """
        Load saved model from disk
        
        Args:
            model_path: Path to pickle file containing model
            
        Returns:
            model: Loaded model
        """
        print(f"\n[Loading Model from {model_path}]")
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("  [OK] Model loaded successfully")
            return self.model
        except Exception as e:
            print(f"  [FAILED] Error loading model: {str(e)}")
            return None
    
    def load_scaler(self, scaler_path):
        """
        Load fitted scaler from disk
        
        Args:
            scaler_path: Path to pickle file containing scaler
            
        Returns:
            scaler: Loaded scaler
        """
        print(f"\n[Loading Scaler from {scaler_path}]")
        
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("  [OK] Scaler loaded successfully")
            return self.scaler
        except Exception as e:
            print(f"  [FAILED] Error loading scaler: {str(e)}")
            return None
    
    def load_feature_names(self, features_path):
        """
        Load feature names from JSON file
        
        Args:
            features_path: Path to JSON file with feature names
            
        Returns:
            list: Feature names
        """
        print(f"\n[Loading Feature Names from {features_path}]")
        
        try:
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
            print(f"  [OK] Loaded {len(self.feature_names)} feature names")
            return self.feature_names
        except Exception as e:
            print(f"  [FAILED] Error loading features: {str(e)}")
            return None
    
    def preprocess_single_row(self, data_dict):
        """
        Preprocess single data point for prediction
        
        Args:
            data_dict: Dictionary with feature values
                      {feature_name: value}
            
        Returns:
            np.array: Preprocessed features ready for model
        """
        print("\n[Preprocessing Single Data Point]")
        
        # Create dataframe from dictionary
        df = pd.DataFrame([data_dict])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only required features in correct order
        df = df[self.feature_names]
        
        print(f"  [OK] Features extracted: {len(self.feature_names)}")
        
        # Scale features
        if self.scaler:
            df_scaled = self.scaler.transform(df)
            print("  [OK] Features scaled")
            return df_scaled
        else:
            return df.values
    
    def preprocess_batch(self, data_df):
        """
        Preprocess batch of data points
        
        Args:
            data_df: Dataframe with features
            
        Returns:
            np.array: Preprocessed features
        """
        print(f"\n[Preprocessing Batch of {len(data_df)} Samples]")
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in data_df.columns:
                data_df[feature] = 0
        
        # Select only required features in correct order
        data_df = data_df[self.feature_names]
        
        print(f"  [OK] Features extracted: {len(self.feature_names)}")
        
        # Scale features
        if self.scaler:
            data_scaled = self.scaler.transform(data_df)
            print("  [OK] Features scaled")
            return data_scaled
        else:
            return data_df.values
    
    def predict_single(self, data_dict):
        """
        Make prediction for single data point
        
        Args:
            data_dict: Dictionary with feature values
            
        Returns:
            dict: Prediction with confidence
        """
        if not self.model:
            print("[FAILED] Model not loaded")
            return None
        
        print("\n[Making Single Prediction]")
        
        # Preprocess data
        preprocessed = self.preprocess_single_row(data_dict)
        
        # Make prediction
        prediction = self.model.predict(preprocessed)[0]
        
        # Get probability if available
        probability = None
        try:
            probability = self.model.predict_proba(preprocessed)[0]
        except:
            pass
        
        # Format result
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Churned' if prediction == 1 else 'Not Churned',
            'confidence': None
        }
        
        if probability is not None:
            result['confidence'] = float(probability[prediction])
            result['probability_not_churned'] = float(probability[0])
            result['probability_churned'] = float(probability[1])
        
        print(f"  [OK] Prediction: {result['prediction_label']}")
        if result['confidence']:
            print(f"  [OK] Confidence: {result['confidence']:.2%}")
        
        return result
    
    def predict_batch(self, data_df):
        """
        Make predictions for batch of data
        
        Args:
            data_df: Dataframe with features
            
        Returns:
            pd.DataFrame: Predictions with confidence
        """
        if not self.model:
            print("[FAILED] Model not loaded")
            return None
        
        print(f"\n[Making Batch Predictions for {len(data_df)} Samples]")
        
        # Preprocess data
        preprocessed = self.preprocess_batch(data_df)
        
        # Make predictions
        predictions = self.model.predict(preprocessed)
        
        # Get probabilities if available
        probabilities = None
        try:
            probabilities = self.model.predict_proba(preprocessed)
        except:
            pass
        
        # Create results dataframe
        results = pd.DataFrame({
            'prediction': predictions,
            'prediction_label': ['Churned' if p == 1 else 'Not Churned' for p in predictions]
        })
        
        if probabilities is not None:
            results['probability_not_churned'] = probabilities[:, 0]
            results['probability_churned'] = probabilities[:, 1]
            results['confidence'] = np.max(probabilities, axis=1)
        
        print(f"  [OK] Made {len(results)} predictions")
        print(f"  Churned: {(predictions == 1).sum()}")
        print(f"  Not Churned: {(predictions == 0).sum()}")
        
        return results
    
    def predict_with_explanation(self, data_dict):
        """
        Make prediction with explanation
        
        Args:
            data_dict: Dictionary with feature values
            
        Returns:
            dict: Prediction with explanation
        """
        print("\n[Prediction with Explanation]")
        
        # Get prediction
        result = self.predict_single(data_dict)
        
        # Add explanation
        explanation = {
            'prediction': result['prediction_label'],
            'confidence': result['confidence'],
            'interpretation': None,
            'risk_level': None
        }
        
        if result['confidence']:
            if result['confidence'] >= 0.8:
                explanation['risk_level'] = 'High Risk' if result['prediction'] == 1 else 'Stable'
            elif result['confidence'] >= 0.6:
                explanation['risk_level'] = 'Moderate Risk' if result['prediction'] == 1 else 'Likely Stable'
            else:
                explanation['risk_level'] = 'Uncertain'
        
        if result['prediction'] == 1:
            explanation['interpretation'] = 'Customer is likely to churn. Consider retention strategies.'
        else:
            explanation['interpretation'] = 'Customer is unlikely to churn.'
        
        print(f"  Risk Level: {explanation['risk_level']}")
        print(f"  Interpretation: {explanation['interpretation']}")
        
        result['explanation'] = explanation
        return result
    
    def save_model(self, model_path):
        """
        Save trained model to disk
        
        Args:
            model_path: Path to save pickle file
        """
        if self.model:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"[OK] Model saved to {model_path}")
    
    def save_scaler(self, scaler_path):
        """
        Save fitted scaler to disk
        
        Args:
            scaler_path: Path to save pickle file
        """
        if self.scaler:
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"[OK] Scaler saved to {scaler_path}")
    
    def save_feature_names(self, features_path):
        """
        Save feature names to JSON
        
        Args:
            features_path: Path to save JSON file
        """
        if self.feature_names:
            with open(features_path, 'w') as f:
                json.dump(self.feature_names, f, indent=2)
            print(f"[OK] Feature names saved to {features_path}")
    
    def get_model_info(self):
        """
        Get information about loaded model
        
        Returns:
            dict: Model information
        """
        info = {
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names[:5] + ['...'] if len(self.feature_names) > 5 else self.feature_names
        }
        
        return info

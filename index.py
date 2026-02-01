"""
Flask Web Application - Customer Churn Prediction
Vercel Deployment Entry Point
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json
from pathlib import Path

# Get the project root
PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / 'app'

# Create Flask app with proper paths
app = Flask(__name__, 
            template_folder=str(APP_DIR / 'templates'),
            static_folder=str(APP_DIR / 'static'))
app.secret_key = 'telecom_churn_prediction_2025'

# Load model and scaler
MODEL_PATH = PROJECT_ROOT / 'models' / 'random_forest_tuned.pkl'
SCALER_PATH = PROJECT_ROOT / 'scalers' / 'Robust_Scaler.pkl'

model = None
scaler = None
FEATURE_ORDER = []

# Load model
try:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
except Exception as e:
    print(f"Model error: {e}")

# Load scaler
try:
    if SCALER_PATH.exists():
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
except Exception as e:
    print(f"Scaler error: {e}")

# Load feature order
feature_file = PROJECT_ROOT / 'models' / 'feature_names.json'
if feature_file.exists():
    try:
        with open(feature_file, 'r') as f:
            feature_data = json.load(f)
        if isinstance(feature_data, list):
            FEATURE_ORDER = feature_data
        elif isinstance(feature_data, dict) and 'features' in feature_data:
            FEATURE_ORDER = feature_data['features']
        else:
            FEATURE_ORDER = feature_data
    except Exception as e:
        print(f"Feature order error: {e}")

FEATURE_DEFINITIONS = {
    'numeric_features': ['tenure', 'MonthlyCharges', 'TotalCharges'],
    'categorical_features': [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/overview')
def overview():
    return render_template('overview.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html', features=FEATURE_DEFINITIONS)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not model:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Model file not found or could not be loaded',
                'success': False
            }), 503
        
        data = request.json
        
        try:
            all_features = []
            order = FEATURE_ORDER if FEATURE_ORDER else [
                'customerID','gender','SeniorCitizen','Partner','Dependents','tenure',
                'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
                'PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'
            ]

            for feat in order:
                try:
                    if feat == 'customerID':
                        raw_id = data.get('customerID', '')
                        val = abs(hash(str(raw_id))) % 10000 if raw_id else 0
                    elif feat == 'gender':
                        val = 1 if data.get('gender') == 'Male' else 0
                    elif feat == 'SeniorCitizen':
                        val = int(data.get('SeniorCitizen', 0))
                    elif feat in ['Partner','Dependents','PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']:
                        val = 1 if data.get(feat) == 'Yes' else 0
                    elif feat == 'MultipleLines':
                        val = 1 if data.get('MultipleLines') == 'Yes' else 0
                    elif feat == 'InternetService':
                        internet = data.get('InternetService', 'DSL')
                        val = 0 if internet == 'DSL' else (1 if internet == 'Fiber optic' else 2)
                    elif feat == 'Contract':
                        contract = data.get('Contract', 'Month-to-month')
                        val = 0 if contract == 'Month-to-month' else (1 if contract == 'One year' else 2)
                    elif feat == 'PaperlessBilling':
                        val = 1 if data.get('PaperlessBilling') == 'Yes' else 0
                    elif feat == 'PaymentMethod':
                        pm = data.get('PaymentMethod', 'Electronic check')
                        pm_map = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
                        val = pm_map.get(pm, 0)
                    elif feat in ['tenure','MonthlyCharges','TotalCharges']:
                        val = float(data.get(feat, 0))
                    else:
                        val = float(data.get(feat, 0)) if data.get(feat) is not None else 0
                    all_features.append(val)
                except Exception:
                    all_features.append(0)

            X = np.array([all_features])
            
            if scaler:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]
            
            if probability < 0.3:
                risk_category = "LOW"
                color = "#4CAF50"
            elif probability < 0.6:
                risk_category = "MEDIUM"
                color = "#FF9800"
            else:
                risk_category = "HIGH"
                color = "#F44336"
            
            suggestions = {
                "LOW": "Customer is stable. Continue regular service & maintain satisfaction.",
                "MEDIUM": "Monitor closely. Consider personalized retention offers and discounts.",
                "HIGH": "Urgent action needed. Offer special promotions, upgrades, or premium support."
            }
            
            return jsonify({
                'churn_probability': float(probability),
                'risk_category': risk_category,
                'color': color,
                'suggestion': suggestions[risk_category],
                'success': True,
                'model_accuracy': '87.5%'
            })
        
        except Exception as e:
            return jsonify({
                'error': 'Feature processing failed',
                'details': str(e),
                'success': False
            }), 400
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'success': False
        }), 500

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 200

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

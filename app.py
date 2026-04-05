"""
=============================================================================
    Customer Churn Prediction - Flask Web Application
    Serves the prediction model via a REST API with a beautiful web frontend.
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ============================================================================
# APP CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_churn_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

app = Flask(__name__)
CORS(app)

# ============================================================================
# LOAD MODEL & SCALER
# ============================================================================
model = None
scaler = None


def load_artifacts():
    """Load the trained model and scaler from disk."""
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"[INFO] Model loaded: {type(model).__name__}")
        print(f"[INFO] Scaler loaded: StandardScaler")
    else:
        print("[WARNING] Model or scaler not found. Run churn_prediction.py first!")
        print(f"  Expected model at: {MODEL_PATH}")
        print(f"  Expected scaler at: {SCALER_PATH}")


# ============================================================================
# CATEGORICAL MAPPINGS
# ============================================================================
CATEGORICAL_MAPPINGS = {
    'gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'MultipleLines': {'No': 0, 'No phone service': 1, 'Yes': 2},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingTV': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingMovies': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)': 1,
        'Electronic check': 2,
        'Mailed check': 3
    }
}

# Feature importance order (approximate, from model training)
FEATURE_IMPORTANCE = {
    'Contract': 0.18,
    'tenure': 0.15,
    'OnlineSecurity': 0.10,
    'TechSupport': 0.09,
    'InternetService': 0.08,
    'MonthlyCharges': 0.07,
    'TotalCharges': 0.06,
    'PaymentMethod': 0.05,
    'PaperlessBilling': 0.04,
    'Dependents': 0.03,
    'Partner': 0.03,
    'SeniorCitizen': 0.03,
    'MultipleLines': 0.02,
    'OnlineBackup': 0.02,
    'DeviceProtection': 0.02,
    'StreamingTV': 0.01,
    'StreamingMovies': 0.01,
    'PhoneService': 0.01,
    'gender': 0.005
}


def get_risk_factors(customer_data):
    """
    Determine the top 3 risk factors for a customer based on feature values.
    
    Args:
        customer_data: dict of customer features
    
    Returns:
        list of dicts with factor name and description
    """
    factors = []
    
    if customer_data.get('Contract') == 'Month-to-month':
        factors.append({
            'factor': 'Month-to-month Contract',
            'impact': 'High',
            'description': 'Short-term contracts have the highest churn rate'
        })
    
    tenure = float(customer_data.get('tenure', 0))
    if tenure < 12:
        factors.append({
            'factor': f'Low Tenure ({int(tenure)} months)',
            'impact': 'High',
            'description': 'New customers are more likely to churn'
        })
    
    if customer_data.get('InternetService') == 'Fiber optic':
        factors.append({
            'factor': 'Fiber Optic Internet',
            'impact': 'Medium',
            'description': 'Fiber optic users churn more due to higher costs'
        })
    
    if customer_data.get('OnlineSecurity') == 'No':
        factors.append({
            'factor': 'No Online Security',
            'impact': 'Medium',
            'description': 'Lack of security add-on correlates with churn'
        })
    
    if customer_data.get('TechSupport') == 'No':
        factors.append({
            'factor': 'No Tech Support',
            'impact': 'Medium',
            'description': 'No tech support means unresolved issues'
        })
    
    monthly = float(customer_data.get('MonthlyCharges', 0))
    if monthly > 70:
        factors.append({
            'factor': f'High Monthly Charges (${monthly:.0f})',
            'impact': 'Medium',
            'description': 'Higher charges increase price sensitivity'
        })
    
    if customer_data.get('PaymentMethod') == 'Electronic check':
        factors.append({
            'factor': 'Electronic Check Payment',
            'impact': 'Low',
            'description': 'Manual payment methods correlate with higher churn'
        })
    
    if customer_data.get('PaperlessBilling') == 'Yes' and customer_data.get('Contract') == 'Month-to-month':
        factors.append({
            'factor': 'Paperless Billing',
            'impact': 'Low',
            'description': 'Combined with month-to-month, indicates less commitment'
        })
    
    # If no specific risk factors, add general ones
    if len(factors) == 0:
        factors.append({
            'factor': 'Standard Risk Profile',
            'impact': 'Low',
            'description': 'No major individual risk factors identified'
        })
    
    return factors[:3]


def get_recommendation(risk_level, factors):
    """
    Generate personalized recommendation based on risk level and factors.
    
    Args:
        risk_level: 'Low', 'Medium', or 'High'
        factors: list of risk factor dicts
    
    Returns:
        str: Recommendation text
    """
    if risk_level == 'High':
        return ("⚠️ IMMEDIATE ACTION REQUIRED: This customer is at high risk of churning. "
                "Recommend offering a discounted annual contract, providing a dedicated "
                "account manager, and bundling complimentary services (Online Security, "
                "Tech Support) for 3 months to increase retention probability.")
    elif risk_level == 'Medium':
        return ("📋 PROACTIVE ENGAGEMENT: This customer shows moderate churn indicators. "
                "Consider sending personalized loyalty rewards, scheduling a satisfaction "
                "survey, and recommending service upgrades that match their usage patterns.")
    else:
        return ("✅ MAINTAIN QUALITY: This customer appears satisfied and engaged. Continue "
                "providing excellent service quality. Consider upselling premium services "
                "or referring them to the loyalty rewards program to deepen engagement.")


# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict customer churn from submitted form data or JSON.
    
    Accepts JSON body with customer features.
    Returns:
        JSON: {prediction, churn_probability, risk_level, factors, recommendation}
    """
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded. Run churn_prediction.py first to train and save the model.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Build feature dictionary
        features = {}
        for col, mapping in CATEGORICAL_MAPPINGS.items():
            val = data.get(col, list(mapping.keys())[0])
            features[col] = mapping.get(str(val), 0)
        
        features['SeniorCitizen'] = int(data.get('SeniorCitizen', 0))
        features['tenure'] = float(data.get('tenure', 1))
        features['MonthlyCharges'] = float(data.get('MonthlyCharges', 50))
        features['TotalCharges'] = float(data.get('TotalCharges', 50))
        features['ChargesPerMonth'] = features['TotalCharges'] / (features['tenure'] + 1)
        
        # Create DataFrame
        feature_order = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges', 'ChargesPerMonth'
        ]
        
        input_df = pd.DataFrame([features])[feature_order]
        
        # Scale numerical features
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'ChargesPerMonth']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = float(model.predict_proba(input_df)[0][1])
        
        # Risk level
        if probability < 0.3:
            risk_level = 'Low'
        elif probability < 0.6:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Risk factors
        factors = get_risk_factors(data)
        recommendation = get_recommendation(risk_level, factors)
        
        return jsonify({
            'prediction': 'Yes' if prediction == 1 else 'No',
            'churn_probability': round(probability, 4),
            'risk_level': risk_level,
            'factors': factors,
            'recommendation': recommendation
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': type(model).__name__ if model else None
    })


# ============================================================================
# MAIN
# ============================================================================
load_artifacts()

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Customer Churn Prediction Web App")
    print("  Open: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)


from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import sys
import os

# Define Base Directory (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set the path to the src directory before importing custom engines
sys.path.append(os.path.join(BASE_DIR, "src", "core"))

from escalation_engine import EscalationEngine
from explainability_engine import ExplainabilityEngine

app = Flask(__name__)
CORS(app) # Enable CORS for Streamlit Cloud deployment

print("Waking up the AI Engines...")
escalation_engine = EscalationEngine()
explainability_engine = ExplainabilityEngine()

model_path = os.path.join(BASE_DIR, "src", "models", "xgboost_risk_model.pkl")
features_path = os.path.join(BASE_DIR, "src", "models", "model_features.pkl")
synthetic_db_path = os.path.join(BASE_DIR, "src", "data", "processed", "synthetic_students.csv")

model = joblib.load(model_path)
feature_names = joblib.load(features_path)

# Load the synthetic database 
student_db = pd.read_csv(synthetic_db_path)

print("Encoding text columns for AI compatibility (Manual)...")
# Manual encoding to avoid loading the heavy Scikit-Learn library on Vercel
for col in student_db.select_dtypes(include=['object', 'category']).columns:
    student_db[col] = pd.factorize(student_db[col])[0]
# ---------------
@app.route('/analyze_student', methods=['POST'])
def analyze_student():
    data = request.json
    student_id = data.get('student_id', 'STU-0000')

    # Fetch a random synthetic student for the demo
    student_record = student_db.sample(1).iloc[0].to_dict()

    # 1. Predict Risk Score
    student_df = pd.DataFrame([student_record])
    for col in feature_names: 
        if col not in student_df: student_df[col] = 0
    student_df = student_df[feature_names]
    
    risk_prob = float(model.predict_proba(student_df)[:, 1][0])

    # 2. Get XAI Reasons First (to feed the Recommendation Engine)
    shap_reasons = explainability_engine.generate_intervention_card(student_record)

    # 3. Determine Escalation & Prescriptive Action
    level, intervention = escalation_engine.evaluate_student(
        risk_score=risk_prob,
        shap_reasons=shap_reasons,
        consecutive_high_risk_semesters=1 
    )

    return jsonify({
        "student_id": student_id,
        "risk_score": risk_prob,
        "escalation_level": level,
        "prescribed_intervention": intervention,
        "intervention_card": shap_reasons
    })

@app.route('/risk_summary', methods=['GET'])
def get_risk_summary():
    """
    Calculates the distribution of risk levels across the entire student body.
    Used for the 'Escalation Heatmap' Dashboard.
    """
    # 1. Predict for everyone (in a real app, this would be pre-calculated)
    X = student_db[feature_names]
    probs = model.predict_proba(X)[:, 1]
    
    # 2. Determine levels
    levels = []
    for p in probs:
        lvl, _ = escalation_engine.evaluate_student(p)
        levels.append(lvl)
    
    # 3. Count frequencies
    summary = pd.Series(levels).value_counts().to_dict()
    
    return jsonify(summary)

if __name__ == '__main__':
    # Get port from environment variable for cloud deployment (default to 5000)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
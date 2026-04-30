
import shap
import joblib
import pandas as pd
import numpy as np
import os

class ExplainabilityEngine:
    def __init__(self):
        
        print("Loading Model and Explainer...")
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(BASE_DIR, "src", "models", "xgboost_risk_model.pkl")
        features_path = os.path.join(BASE_DIR, "src", "models", "model_features.pkl")
        
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)
        
        
        self.explainer = shap.TreeExplainer(self.model)

    def generate_intervention_card(self, student_data_dict):
        """
        Calculates the top risk drivers for a single student to populate the UI Intervention Card.
        """
        
        student_df = pd.DataFrame([student_data_dict])
        
       
        for col in self.feature_names:
            if col not in student_df.columns:
                student_df[col] = 0
        student_df = student_df[self.feature_names]

        shap_values = self.explainer.shap_values(student_df)
        
        feature_impacts = list(zip(self.feature_names, shap_values[0]))
        
       
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

       
        top_reasons = []
        for feature, impact in feature_impacts[:3]: 
            if impact > 0:
                effect_text = "Drives Risk UP (Needs Attention)"
            else:
                effect_text = "Drives Risk DOWN (Protective Factor)"
                
            top_reasons.append({
                "metric": feature,
                "shap_weight": float(impact),
                "explanation": effect_text
            })

        return top_reasons


if __name__ == '__main__':
    engine = ExplainabilityEngine()
    
    
    mock_student = {
        'gender': 1, 'highest_education': 2, 'disability': 0,
        'total_lms_clicks': 12, 'average_score': 45.5, 'assessments_completed': 2
    }
    
    print("\nGenerating Intervention Card for Mock Student...")
    reasons = engine.generate_intervention_card(mock_student)
    
    for rank, reason in enumerate(reasons, 1):
        print(f"{rank}. {reason['metric'].upper()}: {reason['shap_weight']:.3f} -> {reason['explanation']}")
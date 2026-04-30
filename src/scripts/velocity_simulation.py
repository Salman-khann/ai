
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Define Base Directory (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src", "core"))
from escalation_engine import EscalationEngine

def simulate_velocity(n_students=100, n_semesters=4):
    """
    Simulates how the distribution of risk levels changes over multiple semesters.
    Used for 'Velocity of Risk' reporting.
    """
    print(f"--- Simulating Risk Velocity for {n_students} students over {n_semesters} semesters ---")
    
    # Define Base Directory (Project Root)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    model_path = os.path.join(BASE_DIR, "src", "models", "xgboost_risk_model.pkl")
    features_path = os.path.join(BASE_DIR, "src", "models", "model_features.pkl")
    data_path = os.path.join(BASE_DIR, "src", "data", "processed", "synthetic_students.csv")
    output_path = os.path.join(BASE_DIR, "src", "data", "processed", "risk_velocity.csv")

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    student_db = pd.read_csv(data_path).sample(n_students)
    
    engine = EscalationEngine()
    
    history = []

    for sem in range(1, n_semesters + 1):
        # Simulate slight changes in student behavior each semester
        # (e.g., scores drop, clicks fluctuate)
        if sem > 1:
            student_db['average_score'] *= np.random.uniform(0.8, 1.1, size=n_students)
            student_db['total_lms_clicks'] *= np.random.uniform(0.7, 1.2, size=n_students)
            student_db['average_score'] = student_db['average_score'].clip(0, 100)

        # Predict Risk
        X = student_db[feature_names].copy()
        
        # Encode categorical columns
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = le.fit_transform(X[col].astype(str))

        probs = model.predict_proba(X)[:, 1]
        
        counts = {"Level 0": 0, "Level 1": 0, "Level 2": 0, "Level 3": 0}
        
        for p in probs:
            lvl, _ = engine.evaluate_student(p)
            if "Level 0" in lvl: counts["Level 0"] += 1
            elif "Level 1" in lvl: counts["Level 1"] += 1
            elif "Level 2" in lvl: counts["Level 2"] += 1
            elif "Level 3" in lvl: counts["Level 3"] += 1
            
        counts["Semester"] = f"Sem {sem}"
        history.append(counts)
        
    velocity_df = pd.DataFrame(history)
    print("\nSimulation Complete. Velocity Matrix:")
    print(velocity_df)
    
    velocity_df.to_csv(output_path, index=False)
    print(f"\nVelocity data saved to {output_path}")

if __name__ == '__main__':
    simulate_velocity()

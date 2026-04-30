
import pandas as pd
import joblib
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.preprocessing import LabelEncoder

def run_fairness_audit():
    print("--- Starting AI Fairness Audit (AIF360) ---")
    
    # Define Base Directory (Project Root)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Load Data and Model
    data_path = os.path.join(BASE_DIR, "src", "data", "processed", "oulad_seed.csv")
    model_path = os.path.join(BASE_DIR, "src", "models", "xgboost_risk_model.pkl")
    features_path = os.path.join(BASE_DIR, "src", "models", "model_features.pkl")
    
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    
    # 2. Preprocess for AIF360
    # AIF360 requires numeric data and specific column mappings
    audit_df = df.copy()
    le = LabelEncoder()
    for col in ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']:
        audit_df[col] = le.fit_transform(audit_df[col].astype(str))
    
    # Define 'is_at_risk' (Target)
    audit_df['is_at_risk'] = audit_df['final_result'].apply(lambda x: 1 if x in ['Fail', 'Withdrawn'] else 0)
    
    # 3. Create AIF360 Dataset
    # We will audit 'gender' (Assume 1 is privileged, 0 is unprivileged for this demo)
    dataset = BinaryLabelDataset(
        df=audit_df[feature_names + ['is_at_risk']],
        label_names=['is_at_risk'],
        favorable_label=0,   # 0 = Low Risk (Favorable outcome)
        unfavorable_label=1, # 1 = High Risk
        protected_attribute_names=['gender']
    )
    
    # 4. Calculate Baseline Metrics (Before Predictions)
    metric_orig = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=[{'gender': 0}],
        privileged_groups=[{'gender': 1}]
    )
    print(f"Statistical Parity Difference (Data): {metric_orig.statistical_parity_difference():.4f}")
    
    # 5. Run Model and Audit Predictions
    X = audit_df[feature_names]
    audit_df['predicted_risk'] = model.predict(X)
    
    dataset_pred = dataset.copy()
    dataset_pred.labels = audit_df[['predicted_risk']].values
    
    metric_pred = ClassificationMetric(
        dataset, dataset_pred,
        unprivileged_groups=[{'gender': 0}],
        privileged_groups=[{'gender': 1}]
    )
    
    print(f"Disparate Impact (Ratio): {metric_pred.disparate_impact():.4f}")
    print(f"Equal Opportunity Difference: {metric_pred.equal_opportunity_difference():.4f}")
    
    # Interpretation for the Journal
    di = metric_pred.disparate_impact()
    if 0.8 <= di <= 1.25:
        print("\nRESULT: Model meets the '80% Rule' for fairness.")
    else:
        print("\nWARNING: Potential bias detected. Mitigation (re-weighing) recommended for Q1 submission.")

    print("--- Audit Complete ---\n")

if __name__ == '__main__':
    run_fairness_audit()

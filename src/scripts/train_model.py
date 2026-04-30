import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

def train_optimized_xgboost_model():
    print("1. Loading Synthetic Digital Twin Data...")
    try:
        df = pd.read_csv("C:\\Users\\salman\\Desktop\\ai\\src\\data\\processed\\oulad_seed.csv")
    except FileNotFoundError:
        print("Error: Could not find synthetic_students.csv.")
        return

    print("2. Preprocessing Data...")
    le = LabelEncoder()
    # List of categorical columns to encode
    categorical_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Define Target Variable
    df['is_at_risk'] = df['final_result'].apply(lambda x: 1 if x in ['Fail', 'Withdrawn'] else 0)
    
    X = df.drop(columns=['final_result', 'id_student', 'is_at_risk'])
    y = df['is_at_risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calculate class imbalance
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"3. Searching for optimal hyperparameters...")
    base_model = xgb.XGBClassifier(
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        eval_metric='logloss'
    )

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=base_model, 
        param_grid=param_grid, 
        scoring='f1', 
        cv=3, 
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:")
    print(grid_search.best_params_)

    print("\n4. Evaluating the Optimized Model...")
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n--- Optimized Model Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("----------------------------------\n")

    print("5. Saving the Optimized Model...")
    model_path = "C:\\Users\\salman\\Desktop\\ai\\src\\models\\xgboost_risk_model.pkl"
    features_path = "C:\\Users\\salman\\Desktop\\ai\\src\\models\\model_features.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(X_train.columns.tolist(), features_path)
    
    print(f"Success! Optimized model saved to {model_path}")

if __name__ == '__main__':
    train_optimized_xgboost_model()
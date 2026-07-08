# scripts/train_model.py
import os
import json
import joblib
import pandas as pd
import shap
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Paths
ROOT = Path(__file__).resolve().parent.parent  # adjust if running from repo root
DATA_PATH = ROOT / "diabetes_prediction_dataset.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "model_pipeline.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# 1. Load data
df = pd.read_csv(DATA_PATH)

# Real Kaggle dataset: mix of numeric + categorical, target is 'diabetes'
numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi',
                'HbA1c_level', 'blood_glucose_level']
categorical_cols = ['gender', 'smoking_history']
feature_cols = numeric_cols + categorical_cols

X = df[feature_cols]
y = df['diabetes']

# 2. Train-test split, stratified (only ~8.5% positives)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale_pos_weight for xgboost, to counter the class imbalance
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
pos_weight = neg / pos

# 3. Shared preprocessing: scale the numerics, one-hot the categoricals.
# sparse_output=False so the transformed matrix stays dense for SHAP.
def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])

pipeline_lr = Pipeline([
    ("pre", make_preprocessor()),
    ("clf", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))
])

pipeline_rf = Pipeline([
    ("pre", make_preprocessor()),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"))
])

pipeline_xgb = Pipeline([
    ("pre", make_preprocessor()),
    ("clf", XGBClassifier(random_state=42, eval_metric="logloss",
                          scale_pos_weight=pos_weight, n_jobs=-1))
])

# 4. Set up param grids for EACH model
param_grid_lr = {
    "clf__C": [0.1, 1.0, 10]
}

param_grid_rf = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [6, 10],
}

param_grid_xgb = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5],
    "clf__learning_rate": [0.05, 0.1]
}

# 5. Create a list of models to loop through
models_to_run = [
    ("LogisticRegression", pipeline_lr, param_grid_lr),
    ("RandomForest", pipeline_rf, param_grid_rf),
    ("XGBoost", pipeline_xgb, param_grid_xgb)
]

# 6. Loop, train, and evaluate
all_metrics = {}
best_model = None
best_model_name = ""
best_roc_auc = -1.0

for name, pipeline, param_grid in models_to_run:
    print(f"--- Training {name} ---")
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1) # Score on roc_auc
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, best_pipeline.predict_proba(X_test)[:, 1])

    # Store metrics
    all_metrics[name] = {
        "best_params": grid.best_params_,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_mat,
        "classification_report": class_report
    }

    # Check if this is the best model
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_model = best_pipeline
        best_model_name = name

# 7. Save the BEST pipeline and ALL metrics
print(f"Best model found: {best_model_name} with ROC AUC: {best_roc_auc}")
joblib.dump(best_model, MODEL_PATH)  # Saves the best one as 'model_pipeline.pkl'
print("Best model saved to:", MODEL_PATH)

# --- SHAP background data (summarize the processed training set) ---
print("Creating SHAP background data...")
try:
    # preprocessing steps only (scaler + one-hot), then summarize with k-means
    preprocessor = best_model[:-1]
    X_train_processed = preprocessor.transform(X_train)

    background_summary = shap.kmeans(X_train_processed, 100)

    BACKGROUND_PATH = MODELS_DIR / "shap_background.npy"
    np.save(BACKGROUND_PATH, background_summary.data)
    print("SHAP background data saved to:", BACKGROUND_PATH)

except Exception as e:
    print(f"Warning: Could not create SHAP background data. {e}")
    print("SHAP plots for linear models may not work.")

with open(METRICS_PATH, "w") as f:
    json.dump(all_metrics, f, indent=4) # Saves all metrics
print("All model metrics saved to:", METRICS_PATH)

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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Paths
ROOT = Path(__file__).resolve().parent.parent  # adjust if running from repo root
DATA_PATH = ROOT / "diabetes.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "model_pipeline.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# 1. Load data
df = pd.read_csv(DATA_PATH)

# If you used 'Outcome' as target and the dataset has only 0/1:
feature_cols = ['HbA1c_level','Pregnancies','Glucose','BloodPressure',
                'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = df[feature_cols]
y = df['Outcome']

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build pipelines for EACH model
pipeline_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=42, max_iter=500))
])

pipeline_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

pipeline_xgb = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1))
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
    "clf__learning_rate": [0.01, 0.1]
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
    roc_auc = roc_auc_score(y_test, best_pipeline.predict_proba(X_test)[:,1])
    
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

# --- NEW: Create SHAP Background Data ---
print("Creating SHAP background data...")
try:
    # 1. Get the preprocessing steps (imputer + scaler) from the best model
    preprocessor = best_model[:-1]
    
    # 2. Process the X_train data (this applies imputation and scaling)
    X_train_processed = preprocessor.transform(X_train)
    
    # 3. Summarize the processed data into 100 representative samples
    # We use shap.kmeans (which is a weighted k-means)
    background_summary = shap.kmeans(X_train_processed, 100)
    
    # 4. Save the summary
    BACKGROUND_PATH = MODELS_DIR / "shap_background.npy"
    np.save(BACKGROUND_PATH, background_summary.data)
    print("SHAP background data saved to:", BACKGROUND_PATH)
    
except Exception as e:
    print(f"Warning: Could not create SHAP background data. {e}")
    print("SHAP plots for linear models may not work.")
# --- END OF NEW BLOCK ---

with open(METRICS_PATH, "w") as f:
    json.dump(all_metrics, f, indent=4) # Saves all metrics
print("All model metrics saved to:", METRICS_PATH)
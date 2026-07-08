# web_functions.py
# ----------------------------------------
# Handles model loading, prediction, and metrics
# ----------------------------------------

import joblib
import numpy as np
import pandas as pd
import os
import json
import streamlit as st # type: ignore # Added for caching

# ----------------------------------------
# Paths (Relative to the app's root)
# ----------------------------------------
# We assume the app is run from the project's root directory (where main.py is)
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model_pipeline.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# ----------------------------------------
# Model Loading
# ----------------------------------------
@st.cache_resource  # Cache the loaded model for performance
def load_model():
    """
    Loads the saved model pipeline from the .pkl file.
    Uses @st.cache_resource to avoid reloading on every script run.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found. Expected at: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading model from {MODEL_PATH}: {e}")
    return model

# ----------------------------------------
# Feature Order (so inputs match training order)
# ----------------------------------------
def get_feature_order():
    """
    Returns the raw input features the pipeline expects (before one-hot
    encoding). MUST match the columns used in train_model.py.
    """
    return [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "gender",
        "smoking_history",
    ]

def get_encoded_feature_names(model):
    """
    Names of the features AFTER preprocessing (scaled numerics + one-hot
    categoricals). Used to label SHAP plots and feature importances, since
    those live in the encoded space, not the raw input space.
    """
    return list(model[:-1].get_feature_names_out())

# ----------------------------------------
# Prediction
# ----------------------------------------
def predict_diabetes(input_data: dict):
    """
    input_data: A dictionary with feature names as keys and
                user-provided values.

    Returns: A dict containing prediction (0 or 1) and probability.
    """
    try:
        # Load the entire pipeline (preprocessing + classifier)
        model = load_model()

        # Build a one-row frame with the raw columns; the pipeline's
        # ColumnTransformer picks columns by name, so order isn't critical,
        # but we keep it fixed for consistency.
        feature_order = get_feature_order()
        input_df = pd.DataFrame([input_data])[feature_order]

        # The pipeline handles scaling + one-hot encoding + prediction.
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        return {"prediction": prediction, "probability": probability}

    except Exception as e:
        return {"error": str(e)}

# ----------------------------------------
# Load Metrics
# ----------------------------------------
@st.cache_data  # Cache this data
def load_metrics():
    """
    Loads model evaluation metrics from metrics.json (if available).
    """
    if not os.path.exists(METRICS_PATH):
        return {"error": f"Metrics file not found at {METRICS_PATH}"}

    try:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        return {"error": f"Error loading metrics: {e}"}

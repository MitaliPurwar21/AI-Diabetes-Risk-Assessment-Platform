# web_functions.py
# ----------------------------------------
# Handles model loading, prediction, and metrics
# ----------------------------------------

import joblib
import numpy as np
import pandas as pd
import os
import json
import streamlit as st # Added for caching

# ----------------------------------------
# Paths (Relative to the app's root)
# ----------------------------------------
# We assume the app is run from the project's root directory
# (where main.py is)
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
    Returns the exact order of features used during training.
    This MUST match the order in your train_model.py script.
    """
    return [
        "HbA1c_level",
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]

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
        # Load the entire pipeline (imputer, scaler, classifier)
        model = load_model()
        
        # Get the correct feature order
        feature_order = get_feature_order()
        
        # Create a DataFrame from the input_data,
        # ensuring the columns are in the correct order.
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_order] 

        # The pipeline handles EVERYTHING:
        # 1. Imputation (e.g., SimpleImputer)
        # 2. Scaling (e.g., StandardScaler)
        # 3. Prediction (e.g., RandomForestClassifier)
        
        # Use .predict() for the final class (0 or 1)
        prediction = int(model.predict(input_df)[0])
        
        # Use .predict_proba() for the probabilities
        # [0][1] gets the probability of the positive class (1)
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
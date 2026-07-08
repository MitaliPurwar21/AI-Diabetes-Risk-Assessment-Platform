# Tabs/diagnosis.py
import streamlit as st
from web_functions import predict_diabetes, load_metrics, get_feature_order, get_encoded_feature_names, load_model
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import io
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# -------------------------------
# 🔑 Load environment and API key
# -------------------------------
load_dotenv()
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)

if not GEMINI_API_KEY:
    st.warning("⚠️ Gemini API key missing. AI-assisted medical recommendations will not work.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------
# 🧠 Main App Function
# -------------------------------
def app():
    DATA_PATH = "./diabetes_prediction_dataset.csv"
    df = pd.read_csv(DATA_PATH)
    """Streamlit app for diabetes diagnosis and AI-assisted medical recommendation"""

    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 24px;
        color: #0000cc;
    }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Diagnosis 🩺", "Medication 💊", "Data Source 🛢️"])

    # ---------------- Tab 1: Diagnosis ----------------
    with tab1:
        st.title("Diagnosis Page")
        st.write("Detect diabetes risk levels from clinical data using the best-performing trained model.")

        # Inputs for the real dataset's features
        user_input = {}
        col1, col2 = st.columns(2)

        with col1:
            user_input["age"] = st.slider("Age", 0, 100, 40)
            user_input["bmi"] = st.slider("BMI", 10.0, 60.0, 27.0)
            user_input["HbA1c_level"] = st.slider("HbA1c level", 3.5, 9.0, 5.5)
            user_input["blood_glucose_level"] = st.slider("Blood glucose level", 80, 300, 120)

        with col2:
            user_input["gender"] = st.selectbox("Gender", ["Female", "Male", "Other"])
            user_input["smoking_history"] = st.selectbox(
                "Smoking history",
                ["never", "No Info", "current", "former", "ever", "not current"]
            )
            user_input["hypertension"] = 1 if st.checkbox("Hypertension") else 0
            user_input["heart_disease"] = 1 if st.checkbox("Heart disease") else 0

        # Display selected values
        st.subheader("Selected Values:")
        st.table(pd.DataFrame([(k, str(v)) for k, v in user_input.items()], columns=["Feature", "Value"]))

        # Predict
        if st.button("Predict"):
            try:
                result = predict_diabetes(user_input)

                if "error" in result:
                    st.error(f"Prediction failed: {result['error']}")
                else:
                    prediction = result["prediction"]
                    prob = result["probability"]

                    if prediction == 1:
                        msg = f"⚠️ The model predicts **Diabetes** with probability {prob * 100:.2f}%"
                        st.warning(msg)
                    else:
                        msg = f"✅ The model predicts **No Diabetes** with probability {(1 - prob) * 100:.2f}%"
                        st.success(msg)

                    # Save for PDF
                    st.session_state['prediction_result'] = msg
                    st.session_state['predicted_probability'] = f"{prob * 100:.2f}%"
                    st.session_state['user_input_data'] = user_input

                    # --- SHAP EXPLANATION ---
                    st.subheader("🔬 Prediction Explained")

                    try:
                        # Load model, split off the classifier + preprocessor
                        model = load_model()
                        clf = model.named_steps['clf']
                        preprocessor = model[:-1]

                        # Preprocess the user's row into the encoded space
                        feature_order = get_feature_order()
                        input_df = pd.DataFrame([user_input])[feature_order]
                        input_processed = preprocessor.transform(input_df)
                        enc_names = get_encoded_feature_names(model)

                        # SHAP values for this row (tree vs linear explainer)
                        raw = None
                        if isinstance(clf, (RandomForestClassifier, XGBClassifier)):
                            raw = shap.TreeExplainer(clf).shap_values(input_processed)
                        elif isinstance(clf, LogisticRegression):
                            background_data = np.load("models/shap_background.npy")
                            raw = shap.LinearExplainer(clf, background_data).shap_values(input_processed)
                        else:
                            st.info(f"SHAP explanations are not configured for {type(clf).__name__}.")

                        if raw is not None:
                            # squeeze whatever shape the explainer returns down to one
                            # value per feature for the positive (diabetes) class
                            if isinstance(raw, list):
                                arr = np.array(raw[1] if len(raw) > 1 else raw[0])
                            else:
                                arr = np.array(raw)
                            if arr.ndim == 3:      # (samples, features, classes)
                                row = arr[0, :, -1]
                            elif arr.ndim == 2:    # (samples, features)
                                row = arr[0]
                            else:
                                row = arr

                            contrib = pd.DataFrame({"feature": enc_names, "shap": row})
                            contrib = contrib[contrib["shap"].abs() > 1e-6].sort_values("shap")

                            st.write("Each bar shows how that feature pushed this prediction. Red pushes toward diabetes, blue pushes away.")
                            fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(contrib))))
                            colors = ["#FF4B4B" if v > 0 else "#4B8BFF" for v in contrib["shap"]]
                            ax.barh(contrib["feature"], contrib["shap"], color=colors)
                            ax.axvline(0, color="grey", linewidth=0.8)
                            ax.set_xlabel("SHAP value  (→ higher diabetes risk)")
                            ax.set_title("Feature contributions to this prediction")
                            plt.tight_layout()
                            st.pyplot(fig)

                    except Exception as e:
                        st.error(f"An error occurred during SHAP analysis: {e}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # PDF Download
        st.subheader("Download Test Report")
        user_name = st.text_input("Enter your name (required for download):")

        if user_name:
            if 'prediction_result' in st.session_state:
                # --- PDF Generation ---
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, txt="Diabetes Risk Assessment Report", ln=True, align='C')
                pdf.ln(10)

                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"User Name: {user_name}", ln=True)
                pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                pdf.ln(10)

                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt="Prediction Result:", ln=True)
                pdf.set_font("Arial", size=12)

                import re
                clean_text = re.sub(r'[^\x00-\x7F]+', '', st.session_state['prediction_result'])
                pdf.multi_cell(0, 10, clean_text)
                pdf.ln(5)

                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt="Predicted Probability:", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=st.session_state['predicted_probability'], ln=True)
                pdf.ln(10)

                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt="Measurements:", ln=True)
                pdf.set_font("Arial", size=12)
                for feature, value in user_input.items():
                    pdf.cell(100, 10, txt=f"{feature}:", ln=False)
                    pdf.cell(100, 10, txt=f"{value}", ln=True)

                pdf_bytes = pdf.output(dest='S').encode('latin-1')

                # --- CSV Generation ---
                slider_df = pd.DataFrame(user_input.items(), columns=["Feature", "Value"])
                csv_buffer = io.StringIO()
                slider_df.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue()

                # --- Download Buttons (in columns) ---
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{user_name}_diabetes_report.pdf",
                        mime="application/pdf",
                    )

                with col2:
                    st.download_button(
                        label="💾 Download CSV Data",
                        data=csv_bytes,
                        file_name=f"{user_name}_diabetes_data.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Run prediction first to generate a report.")
        else:
            st.info("Please enter your name to enable downloads.")

    # ---------------- Tab 2: Medication ----------------
    with tab2:
        st.title("AI-Assisted Medical Recommendations")

        def get_gemini_medication_recommendation(disease_type, patient_data):
            if not GEMINI_API_KEY:
                return "Gemini API key not configured."
            prompt = f"""
            You are a medical expert. Based on this disease diagnosis, suggest medications and lifestyle recommendations:
            Disease: {disease_type}
            Patient Data: {patient_data}
            """
            try:
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"⚠️ The AI service is unavailable right now (rate limit or API quota). [{type(e).__name__}]"

        st.caption("CSV format: two columns (feature, value), one row per feature. "
                   "Features: " + ", ".join(get_feature_order()))
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                st.dataframe(df_uploaded)

                if df_uploaded.shape[1] < 2:
                    st.error("CSV must have at least two columns: parameter and value.")
                    st.stop()

                df_processed = pd.DataFrame([{p: v for p, v in zip(df_uploaded.iloc[:, 0], df_uploaded.iloc[:, 1])}])

                required_cols = get_feature_order()
                missing = [c for c in required_cols if c not in df_processed.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                    st.stop()

                features = {f: df_processed[f].iloc[0] for f in required_cols}

                result = predict_diabetes(features)
                prediction = result["prediction"]
                prob = result["probability"]

                disease = "Diabetes Detected" if prediction == 1 else "No Diabetes"

                st.subheader("Patient Recommendation:")
                if disease == "Diabetes Detected":
                    st.warning(disease)
                    recommendation = get_gemini_medication_recommendation(disease, df_processed.to_dict())
                    st.info("Gemini AI Recommended Medication:")
                    st.write(recommendation)
                else:
                    st.success("No diabetes detected.")
                    st.info("Maintain a healthy lifestyle.")

            except Exception as e:
                st.error(f"Error: {e}")

    # ---------------- Tab 3: Data Source ----------------
    with tab3:
        st.title("Data Info Page")
        with st.expander("View data"):
            st.dataframe(df)

        st.subheader("Columns Description:")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.checkbox("Column Names"):
                st.dataframe(df.columns)

        with col2:
            if st.checkbox("View Summary"):
                st.dataframe(df.describe())

        with col3:
            if st.checkbox("Columns Data"):
                col = st.selectbox("Column Name", list(df.columns))
                st.dataframe(df[col])

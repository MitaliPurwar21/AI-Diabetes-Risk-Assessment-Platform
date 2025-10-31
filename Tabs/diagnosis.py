# Tabs/diagnosis.py
import streamlit as st
from web_functions import predict_diabetes, load_metrics, get_feature_order, load_model
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
    DATA_PATH = "./diabetes.csv"
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

        # --- FIXED: Removed the broken sidebar metrics ---
        # (The result.py tab now handles this correctly)

        # Dynamic feature inputs
        feature_order = get_feature_order()
        user_input = {}

        for feature in feature_order:
            if feature == "Pregnancies" or feature == "Age":
                user_input[feature] = st.slider(feature, 0, 20 if feature == "Pregnancies" else 90, 1)
            elif feature == "Glucose":
                user_input[feature] = st.slider(feature, 40, 250, 120)
            elif feature == "BloodPressure":
                user_input[feature] = st.slider(feature, 30, 140, 70)
            elif feature == "SkinThickness":
                user_input[feature] = st.slider(feature, 0, 99, 20)
            elif feature == "Insulin":
                user_input[feature] = st.slider(feature, 0, 900, 100)
            elif feature == "BMI":
                user_input[feature] = st.slider(feature, 15.0, 60.0, 25.0)
            elif feature == "DiabetesPedigreeFunction":
                user_input[feature] = st.slider(feature, 0.1, 2.5, 0.5)
            elif feature == "HbA1c_level":
                user_input[feature] = st.slider(feature, 4.0, 14.0, 6.0)

        # Display selected values
        st.subheader("Selected Values:")
        st.table(pd.DataFrame(user_input.items(), columns=["Feature", "Value"]))

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
                    
                    # --- NEW: SHAP EXPLANATION (Corrected Plotting) ---
                    st.subheader("🔬 Prediction Explained")
                    
                    try:
                        # 1. Load model and get components
                        model = load_model() 
                        clf = model.named_steps['clf']
                        preprocessor = model[:-1] # Get the full preprocessor
                        
                        # 3. Prepare user input
                        feature_order = get_feature_order()
                        input_df = pd.DataFrame([user_input])[feature_order]
                        
                        # 4. Apply the FULL preprocessing pipeline
                        input_processed = preprocessor.transform(input_df)

                        # 5. Create SHAP explainer based on model type
                        
                        # --- Handle Tree Models ---
                        if isinstance(clf, (RandomForestClassifier, XGBClassifier)):
                            explainer = shap.TreeExplainer(clf)
                            shap_values = explainer.shap_values(input_processed)
                            expected_value = explainer.expected_value

                            shap_vals_for_plot = None
                            expected_val_for_plot = None

                            if isinstance(shap_values, list) and len(shap_values) == 2:
                                shap_vals_for_plot = shap_values[1][0] # Use class 1
                                expected_val_for_plot = expected_value[1]
                            else:
                                shap_vals_for_plot = shap_values[0]
                                expected_val_for_plot = expected_value

                            if shap_vals_for_plot is not None:
                                st.write("This chart shows how each feature *pushed* the prediction from the 'base' value (average prediction) to the final output. Red features increase the risk, blue features decrease it.")
                                
                                # --- FIXED PLOTTING ---
                                # 1. Remove the bad plt.subplots() line
                                # 2. Capture the figure returned by shap.force_plot
                                force_plot_fig = shap.force_plot(
                                    expected_val_for_plot, 
                                    shap_vals_for_plot, 
                                    input_df, 
                                    matplotlib=True, 
                                    show=False,
                                    figsize=(20, 3) # Pass figsize here
                                )
                                # 3. Plot the correct figure
                                st.pyplot(force_plot_fig, bbox_inches='tight')
                                                            
                        # --- Handle Linear Models ---
                        elif isinstance(clf, LogisticRegression):
                            background_data = np.load("models/shap_background.npy")
                            explainer = shap.LinearExplainer(clf, background_data)
                            shap_values = explainer.shap_values(input_processed)
                            
                            st.write("This chart shows how each feature *pushed* the prediction from the 'base' value (average prediction) to the final output. Red features increase the risk, blue features decrease it.")
                            
                            # --- FIXED PLOTTING ---
                            # 1. Remove the bad plt.subplots() line
                            # 2. Capture the figure returned by shap.force_plot
                            force_plot_fig = shap.force_plot(
                                explainer.expected_value, 
                                shap_values[0], 
                                input_df, 
                                matplotlib=True, 
                                show=False,
                                figsize=(20, 3) # Pass figsize here
                            )
                            # 3. Plot the correct figure
                            st.pyplot(force_plot_fig, bbox_inches='tight')
                            
                        # --- Handle other models ---
                        else:
                            st.info(f"SHAP explanations are not currently configured for the winning model type ({type(clf).__name__}).")

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
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text

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
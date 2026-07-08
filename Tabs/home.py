import streamlit as st

def app():
    st.title('AI-Powered Diabetes Risk Assessment Platform and Advisor')
    st.image('./images/diabetic.png')

    st.markdown(
    """
    <p style="font-size:20px;">
    Diabetes is a chronic health condition that affects how your body turns food into energy. This project explores how modern data science techniques can be used to build a robust and transparent tool for diabetes risk assessment.
    </p>
    """, unsafe_allow_html=True)
    
    st.subheader("Key Project Features:")
    
    st.markdown(
    """
    * **Robust Model Training:** This app is trained on the **100,000-record Kaggle Diabetes Prediction Dataset** of real clinical records with age, BMI, HbA1c, blood glucose, and smoking history.

    * **Automated Model Selection:** The prediction model wasn't just *chosen*, it *competed*. The training script compares **Logistic Regression, Random Forest, and XGBoost** (handling the ~8.5% positive-class imbalance), automatically selecting the best-performing model (based on ROC AUC) to serve in this app.

    * **Explainable AI (XAI):** This tool doesn't just give a prediction; it explains *why*. Using **SHAP plots** on the 'Diagnosis' page, you can see exactly how each factor (like HbA1c or blood glucose) contributed to the final risk score.
    """)

    st.markdown(
    """
    <p style="font-size:20px; margin-top: 20px;">
    Use the 'Diagnosis' tab to get your risk assessment, or visit the 'Knowledge Center' tab for a full technical breakdown.
    </p>
    """, unsafe_allow_html=True)
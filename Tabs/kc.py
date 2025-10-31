import streamlit as st

def app():
    st.markdown('''<h1><center>About This Project</center></h1>''', unsafe_allow_html=True)

    # --- Paragraph 1: Diabetes Detection ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("./images/1.png", caption="ML Risk Classification", width=200)
    with col2:
        st.markdown('''
            **1. Machine Learning Risk Classification**
            
            This project explores how machine learning can be used for diabetes risk classification. It uses a **synthetic dataset of 100,000 patient records** (generated from the original PIMA dataset's statistics) to train and compare three different models: Logistic Regression, Random Forest, and XGBoost.
            
            The system automatically selects the best-performing model (based on ROC AUC score) from the training script to provide predictions.
        ''')

    # --- Paragraph 2: Explainable AI (XAI) ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('''
            **2. Explainable AI (XAI) with SHAP**
            
            A key goal of this project is to move beyond "black box" predictions. When a prediction is made on the 'Diagnosis' page, the app generates a **SHAP (SHapley Additive exPlanations) force plot**.
            
            This chart shows exactly how each patient feature (like 'Glucose' or 'BMI') contributed to the final risk score, providing transparency and helping to build trust in the model's decision.
        ''')
    with col2:
        st.image("./images/2.png", caption="Explainable AI (SHAP)", width=200)

    # --- Paragraph 3: AI-Assisted Q&A ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("./images/3.png", caption="AI-Assisted Q&A", width=200)
    with col2:
        st.markdown('''
            **3. AI-Assisted Q&A**
            
            The 'Ask Queries' tab features an AI-assisted Q&A tool. This is **not a custom-trained chatbot**, but rather a direct interface with **Google's Gemini API**.
            
            It is system-prompted to act as a medical assistant and can answer general questions about diabetes. This demonstrates the integration of large language models (LLMs) into a data science application.
        ''')

    # --- Paragraph 4: Analytical Dashboard ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('''
            **4. Analytical Dashboard**
            
            The 'Result' tab functions as an analytical dashboard for the project. Instead of tracking "global trends," this dashboard provides a **deep dive into the project's data and models**.
            
            It includes a model comparison table, a confusion matrix for the winning model, feature importance plots, and exploratory data analysis (EDA) visualizations like correlation heatmaps from the training data.
        ''')
    with col2:
        st.image("./images/4.png", caption="Analytical Dashboard", width=200)

    # --- Paragraph 5: Streamlit Integration ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("./images/5.png", caption="Streamlit Integration", width=200)
    with col2:
        st.markdown('''
            **5. Project Architecture**
            
            This entire application is built in Python and served using **Streamlit**. The project follows modern development practices by separating concerns:
            
            * **`scripts/`**: A separate script handles all heavy model training and evaluation.
            * **`models/`**: The trained model, metrics, and SHAP data are saved as artifacts.
            * **`Tabs/`**: The Streamlit app is modularized, with each page as its own file.
        ''')

# Run the app
if __name__ == "__main__":
    app()
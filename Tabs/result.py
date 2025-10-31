# Tabs/result.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from web_functions import load_model, load_metrics, get_feature_order
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Set global style for neon-like effect
plt.style.use("dark_background")

def app():
    st.title("📊 Analytical Dashboard")
    st.write("This dashboard provides an analytical overview of the model's performance and the data it was trained on.")

    # --- Load Data and Models ---
    try:
        # Load data for correlations and population means
        df = pd.read_csv("diabetes.csv")
        
        # Load model for feature importances
        model = load_model()
        
        # Load all metrics
        all_metrics = load_metrics() # This is now a dictionary of models

        if "error" in all_metrics:
            raise FileNotFoundError(all_metrics["error"])
        
    except FileNotFoundError:
        st.error("Error: Could not find necessary files (diabetes.csv, model_pipeline.pkl, metrics.json). Please run the training script.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return

    # --- Tab Layout ---
    tab1, tab2, tab3 = st.tabs(["Model Performance", "Data Analysis", "Patient vs. Population"])

    # ----------------------------------------
    # Tab 1: Model Performance
    # ----------------------------------------
    with tab1:
        st.header("Model Performance Metrics")
        
        # Model Comparison Table
        st.subheader("Model Comparison")
        try:
            # Create a summary DataFrame
            comparison_data = []
            for model_name, metrics in all_metrics.items():
                comparison_data.append({
                    "Model": model_name,
                    "Accuracy": metrics.get('accuracy'),
                    "ROC AUC": metrics.get('roc_auc')
                })
            df_comparison = pd.DataFrame(comparison_data).sort_values(by="ROC AUC", ascending=False)
            df_comparison["Accuracy"] = df_comparison["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
            df_comparison["ROC AUC"] = df_comparison["ROC AUC"].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(df_comparison.set_index("Model"))
            st.info("The best model (based on ROC AUC) is automatically selected for prediction and analysis below.")
            
        except Exception as e:
            st.error(f"Could not generate model comparison table: {e}")
            
        # --- Analysis of the BEST Model ---
        # Find the best model name from metrics (the one that won)
        best_model_name = ""
        best_roc_auc = -1
        for model_name, metrics in all_metrics.items():
            if metrics.get('roc_auc', -1) > best_roc_auc:
                best_roc_auc = metrics.get('roc_auc')
                best_model_name = model_name
                
        if not best_model_name:
             st.error("Could not identify the best model from metrics.json")
             return

        st.header(f"Detailed Analysis: {best_model_name}")
        best_model_metrics = all_metrics[best_model_name]

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy & ROC AUC")
            if "accuracy" in best_model_metrics:
                st.metric("Test Accuracy", f"{best_model_metrics['accuracy'] * 100:.2f}%")
            if "roc_auc" in best_model_metrics:
                st.metric("Test ROC AUC", f"{best_model_metrics['roc_auc']:.4f}")

            st.subheader("Feature Importances")
            try:
                clf = model.named_steps['clf']
                feature_names = get_feature_order()
                importances = None

                # --- NEW: Check model type ---
                if isinstance(clf, (RandomForestClassifier, XGBClassifier)):
                    importances = clf.feature_importances_
                elif isinstance(clf, LogisticRegression):
                    # For linear models, we use the absolute value of the coefficients
                    importances = np.abs(clf.coef_[0])
                # --- END OF NEW BLOCK ---

                if importances is not None:
                    # Create a DataFrame
                    df_importances = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=True)

                    # Create bar chart
                    fig, ax = plt.subplots()
                    ax.set_facecolor("black")
                    ax.barh(df_importances['Feature'], df_importances['Importance'], color='#00FFFF')
                    ax.set_title("Feature Importances", color='white')
                    ax.set_xlabel("Importance (Magnitude)", color='white')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    # This message will now only show if the model is unknown
                    st.info("Feature importances are not configured for this model type.")

            except Exception as e:
                st.error(f"Could not plot feature importances: {e}")

        with col2:
            st.subheader("Confusion Matrix (Test Set)")
            if "confusion_matrix" in best_model_metrics:
                cm = best_model_metrics['confusion_matrix']
                
                # --- ✅ START OF FIXED CODE 2 ---
                fig, ax = plt.subplots()
                ax.set_facecolor("black")
                sns.heatmap(cm, annot=True, fmt='d', cmap="coolwarm", ax=ax,
                            annot_kws={"color": "white"}, cbar_kws={'label': 'Count'})
                ax.set_title("Confusion Matrix", color='white')
                ax.set_xlabel("Predicted Label", color='white')
                ax.set_ylabel("True Label", color='white')
                ax.set_xticklabels(['No Diabetes (0)', 'Diabetes (1)'], color='white')
                ax.set_yticklabels(['No Diabetes (0)', 'Diabetes (1)'], color='white', rotation=0)
                st.pyplot(fig)
                # --- ✅ END OF FIXED CODE 2 ---
            
            else:
                st.info("Confusion matrix not found.")

    # ----------------------------------------
    # Tab 2: Data Analysis
    # ----------------------------------------
    with tab2:
        st.header("Exploratory Data Analysis")
        
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor("black")
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='viridis', ax=ax,
                    annot_kws={"color": "black", "size": 8})
        ax.set_title("Feature Correlation", color='white')
        plt.setp(ax.get_xticklabels(), color='white', rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), color='white', rotation=0)
        st.pyplot(fig)
        
        st.subheader("Feature Distributions by Outcome")
        feature = st.selectbox("Select Feature to Visualize", get_feature_order())
        
        fig, ax = plt.subplots()
        ax.set_facecolor("black")
        # Histogram
        sns.histplot(data=df, x=feature, hue='Outcome', kde=True, ax=ax, palette=['#00FFFF', '#FF00FF'])
        ax.set_title(f"{feature} Distribution by Diabetes Outcome", color='white')
        ax.set_xlabel(feature, color='white')
        ax.set_ylabel("Count", color='white')
        st.pyplot(fig)

    # ----------------------------------------
    # Tab 3: Patient vs. Population
    # ----------------------------------------
    with tab3:
        st.header("Your Vitals vs. Population Average")
        
        if 'user_input_data' in st.session_state:
            user_data = st.session_state['user_input_data']
            
            # Calculate population means
            pop_means = df[get_feature_order()].mean()
            
            # Create comparison DataFrame
            df_compare = pd.DataFrame({
                'Your Value': user_data,
                'Population Mean': pop_means
            })

            # Create grouped bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_facecolor("black")
            df_compare.plot(kind='bar', ax=ax, color=['#00FFFF', '#FF00FF'])
            ax.set_title("Your Vitals vs. Population Mean", color='white')
            ax.set_ylabel("Value", color='white')
            plt.setp(ax.get_xticklabels(), color='white', rotation=45, ha='right')
            ax.legend()
            st.pyplot(fig)
            
            st.dataframe(df_compare.T.round(2)) # Show the data as well
            
        else:
            st.info("ℹ️ Please run a prediction on the 'Diagnosis' page to see your personalized comparison.")

# Run the app
if __name__ == "__main__":
    app()
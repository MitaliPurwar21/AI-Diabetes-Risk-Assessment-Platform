# AI-Powered Diabetes Risk Assessment Platform and Advisor

This project is a complete, end-to-end data science application built in Python and Streamlit. It moves beyond a simple "black box" prediction by comparing multiple ML models, selecting the best one, and providing in-depth, explainable AI (XAI) analysis for each prediction using SHAP.

**Link to the deployed app:** https://ai-powered-diabetes-risk-assessment-platform.streamlit.app/

---

## Key Features

* **✨ Automated Model Selection:** The app doesn't just use one model. The training script (`scripts/train_model.py`) compares `LogisticRegression`, `RandomForest`, and `XGBoost` and automatically saves the best-performing model (based on ROC AUC) for the app to use.
* **🔬 Explainable Predictions (XAI):** Built with `SHAP` (SHapley Additive exPlanations). After making a prediction, the app generates a "force plot" that shows *why* the model made its decision, detailing how each feature (like Glucose or BMI) "pushed" the risk score. This logic correctly handles *both* linear models (`LogisticRegression`) and tree-based models (`XGBoost`, `RandomForest`).
* **📊 Synthetic Data Generation:** To avoid overfitting on the small, common PIMA dataset, this project uses a 100,000-row synthetic dataset. The script (`diabetes_synthetic.py`) generates this data by modeling feature distributions and using a logistic function to create a probabilistic outcome.
* **📈 Analytical Dashboard:** A dedicated "Result" tab shows a full breakdown of the model training process, including:
    * A performance comparison table for all 3 models.
    * The confusion matrix for the winning model.
    * Feature importance plots.
    * Exploratory Data Analysis (EDA) of the training data.
    * An analytical overview of the model's performance and the data it was trained on.
* **🤖 AI-Assisted Q&A:** Integrates Google's Gemini API with a specific system prompt, allowing users to ask general medical questions about diabetes.
* **📄 PDF & CSV Report:** Users can download their diagnosis and the data they entered as a PDF or CSV file.

---

## 🛠 Tech Stack

* **Python 3.10**
* **Streamlit:** For the web app interface.
* **Scikit-learn:** For ML pipelines, model training (`LogisticRegression`, `RandomForest`), and metrics.
* **XGBoost:** For the `XGBClassifier` model.
* **SHAP:** For generating all explainable AI plots.
* **Pandas:** For data manipulation.
* **Google Gemini API:** For the AI-Assisted Q&A feature.
* **Matplotlib & Seaborn:** For data visualizations.
* **Utilities:** Joblib, FPDF

---

## 🚀 How to Run Locally

### Prerequisites

* Python 3.9+
* Homebrew (for macOS users) to install system dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/MitaliPurwar21/AI-Diabetes-Risk-Assessment-Platform.git
cd AI-Diabetes-Risk-Assessment-Platform
```

### 2. Set Up the Environment
On macOS (for XGBoost): You must install the libomp library first.

```bash
brew install libomp
```

### 3. Create and Activate a Virtual Environment

#### For Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### For Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up API keys

This project requires a Google Gemini API Key for the chatbot and recommendation features.

* Create a folder .streamlit in the project's root directory.
* Inside this folder, create a file named secrets.toml.
* Add your API key to this file:
```toml
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

### 6. Train the Model
Run the training script. This will train all the models, run the comparison, and save the best pipeline (model_pipeline.pkl), metrics (metrics.json), and SHAP data (shap_background.npy) to the models/ folder.

```bash
python scripts/train_model.py
```

### 7. Run the Streamlit App

```bash
streamlit run main.py
```

Your app will now be running and accessible in your browser at http://localhost:8501

**License**
This project is licensed under the MIT License.

**Author**
Mitali
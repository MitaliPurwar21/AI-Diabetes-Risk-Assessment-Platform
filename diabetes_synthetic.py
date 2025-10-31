import numpy as np
import pandas as pd
import os
from scipy.special import expit as sigmoid  # sigmoid function for probabilities

# -------------------------------
# Configuration
# -------------------------------
num_records = 100000
np.random.seed(42)

# Means and standard deviations roughly based on original Pima dataset
means = {
    'Pregnancies': 3.84,
    'Glucose': 121.60,
    'BloodPressure': 72.20,
    'SkinThickness': 26.60,
    'Insulin': 118.68,
    'BMI': 32.40,
    'DiabetesPedigreeFunction': 0.47,
    'Age': 33.24
}

std_devs = {
    'Pregnancies': 3.36,
    'Glucose': 30.40,
    'BloodPressure': 12.10,
    'SkinThickness': 9.60,
    'Insulin': 93.08,
    'BMI': 6.80,
    'DiabetesPedigreeFunction': 0.33,
    'Age': 11.76
}

# -------------------------------
# Generate Features
# -------------------------------
data = {}
for feature in means:
    data[feature] = np.random.normal(means[feature], std_devs[feature], num_records)

df = pd.DataFrame(data)

# Clip features to realistic ranges
df['Pregnancies'] = df['Pregnancies'].clip(0, 15).round(0)
df['Glucose'] = df['Glucose'].clip(40, 250).round(1)
df['BloodPressure'] = df['BloodPressure'].clip(30, 140).round(1)
df['SkinThickness'] = df['SkinThickness'].clip(0, 99).round(1)
df['Insulin'] = df['Insulin'].clip(0, 900).round(1)
df['BMI'] = df['BMI'].clip(15, 60).round(1)
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].clip(0.1, 2.5).round(3)
df['Age'] = df['Age'].clip(18, 90).round(0)

# -------------------------------
# Generate HbA1c_level (correlated with Glucose & BMI)
# -------------------------------
# Typical HbA1c ~ 4% to 14%
df['HbA1c_level'] = 4 + 0.02*df['Glucose'] + 0.01*df['BMI'] + np.random.normal(0, 0.5, size=num_records)
df['HbA1c_level'] = df['HbA1c_level'].clip(4, 14).round(2)

# -------------------------------
# Generate Outcome probabilistically (logistic regression style)
# -------------------------------
# Coefficients roughly inspired by literature
coeffs = {
    'Pregnancies': 0.05,
    'Glucose': 0.04,
    'BloodPressure': -0.01,
    'SkinThickness': 0.005,
    'Insulin': 0.0005,
    'BMI': 0.03,
    'DiabetesPedigreeFunction': 0.8,
    'Age': 0.02,
    'HbA1c_level': 0.3
}

intercept = -8.0

linear_term = (
    coeffs['Pregnancies']*df['Pregnancies'] +
    coeffs['Glucose']*df['Glucose'] +
    coeffs['BloodPressure']*df['BloodPressure'] +
    coeffs['SkinThickness']*df['SkinThickness'] +
    coeffs['Insulin']*df['Insulin'] +
    coeffs['BMI']*df['BMI'] +
    coeffs['DiabetesPedigreeFunction']*df['DiabetesPedigreeFunction'] +
    coeffs['Age']*df['Age'] +
    coeffs['HbA1c_level']*df['HbA1c_level'] +
    intercept
)

# Convert linear term to probability
prob = sigmoid(linear_term)

# Sample Outcome from Bernoulli
df['Outcome'] = np.random.binomial(1, prob)

# -------------------------------
# Reorder columns
# -------------------------------
df = df[['HbA1c_level', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']]

# -------------------------------
# Save CSV to Downloads folder
# -------------------------------
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", "diabetes_synthetic_100k.csv")
df.to_csv(downloads_path, index=False)
print(f"✅ Synthetic dataset saved to: {downloads_path}")

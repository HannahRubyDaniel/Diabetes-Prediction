import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# ✅ Ensure correct dataset path
CSV_PATH = "C:/diabetes.csv"

# ✅ Load dataset with column names
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(CSV_PATH, names=columns, header=0)

# ✅ Extract features (X) and target (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ✅ Standardize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Split dataset into training and testing
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Train Logistic Regression model (using scaled training data)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ✅ Save trained model and scaler
MODEL_PATH = "D:/pythonProject/DiabetesPredictionNew/DiabetesPrediction/DiabetesPrediction/diabetes_model.pkl"
SCALER_PATH = "D:/pythonProject/DiabetesPredictionNew/DiabetesPrediction/DiabetesPrediction/scaler.pkl"
SHAP_PATH = "D:/pythonProject/DiabetesPredictionNew/DiabetesPrediction/DiabetesPrediction/shap_values.pkl"

with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)

with open(SCALER_PATH, "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"✅ Logistic Regression model saved at: {MODEL_PATH}")
print(f"✅ StandardScaler saved at: {SCALER_PATH}")

# --- SHAP EXPLAINABILITY INTEGRATION ---
# ✅ Compute SHAP values (Fix: Use predict_proba for probability of class 1)
explainer = shap.LinearExplainer(model, X_train_scaled)  # Directly pass the trained model
shap_values = explainer(X_train_scaled)  # Compute SHAP values for training data


# ✅ Save SHAP values for later use in Django app
with open(SHAP_PATH, "wb") as shap_file:
    pickle.dump(shap_values, shap_file)

print(f"✅ SHAP values saved at: {SHAP_PATH}")

# ✅ Generate and save SHAP summary plot (Fix: Use correct training data)
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_train_scaled, feature_names=df.columns[:-1], show=False)
SHAP_SUMMARY_PATH = os.path.join(os.path.dirname(SHAP_PATH), "shap_summary.png")
plt.savefig(SHAP_SUMMARY_PATH, bbox_inches='tight')
plt.close()

print(f"✅ SHAP summary plot saved at: {SHAP_SUMMARY_PATH}")


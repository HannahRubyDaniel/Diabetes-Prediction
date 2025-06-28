import joblib
import pickle

# ✅ Correct model & scaler paths
MODEL_PATH = "D:/pythonProject/DiabetesPredictionNew/DiabetesPrediction/DiabetesPrediction/diabetes_model.pkl"
SCALER_PATH = "D:/pythonProject/DiabetesPredictionNew/DiabetesPrediction/DiabetesPrediction/scaler.pkl"

# ✅ Load model & scaler correctly
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_diabetes(input_data):
    """Predict diabetes using the trained model."""
    input_scaled = scaler.transform([input_data])  # ✅ Ensure correct input shape
    prediction = model.predict(input_scaled)
    return prediction[0]  # Return single prediction


MODEL_PATH = "D:/pythonProject/DiabetesPredictionNew/DiabetesPrediction/DiabetesPrediction/diabetes_model.pkl"

# ✅ Load the model correctly
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

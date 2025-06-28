from django.shortcuts import render, redirect
import numpy as np
import joblib
import os
import shap
import pandas as pd  # Ensure pandas is imported
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .models import PredictionHistory

# ✅ Load the trained model and scaler
MODEL_PATH = os.path.join(settings.BASE_DIR, "DiabetesPrediction", "diabetes_model.pkl")
SCALER_PATH = os.path.join(settings.BASE_DIR, "DiabetesPrediction", "scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if model is None or scaler is None:
        raise ValueError("❌ Model or Scaler is not loaded properly.")

    explainer = shap.LinearExplainer(model, np.zeros((1, 8)))  # ✅ Fixed SHAP Explainer for Logistic Regression

except Exception as e:
    model, scaler, explainer = None, None, None
    print(f"❌ Error loading model, scaler, or SHAP explainer: {e}")

# --- Authentication Views ---

def home(request):
    return render(request, "home.html")

def learn_about_diabetes(request):
    return render(request, "learn_about_diabetes.html")

def guidelines(request):
    return render(request, 'guidelines.html')

def user_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("predict")
        else:
            messages.error(request, "Invalid credentials")
    return render(request, "home.html")

def user_register(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        if password == confirm_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already exists!")
            else:
                user = User.objects.create_user(username=username, password=password)
                login(request, user)
                return redirect("/")  # Redirect to home page

        else:
            messages.error(request, "Passwords do not match")

    return redirect("/")

def user_logout(request):
    logout(request)
    return redirect("home")

# --- Prediction View with SHAP Explainability ---

@login_required
def predict(request):
    result, shap_values, top_factors = None, None, []

    if request.method == "POST":
        try:
            if model is None or scaler is None or explainer is None:
                raise ValueError("❌ Model or Scaler not loaded!")

            # ✅ Extract user inputs
            pregnancies = int(request.POST.get("pregnancies"))
            glucose = float(request.POST.get("glucose"))
            blood_pressure = float(request.POST.get("blood_pressure"))
            skin_thickness = float(request.POST.get("skin_thickness"))
            insulin = float(request.POST.get("insulin"))
            bmi = float(request.POST.get("bmi"))
            diabetes_pedigree_function = float(request.POST.get("diabetes_pedigree_function"))
            age = int(request.POST.get("age"))

            # ✅ Define feature names correctly
            feature_names = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]

            # ✅ Prepare data as DataFrame
            input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                      insulin, bmi, diabetes_pedigree_function, age]], columns=feature_names)

            # ✅ Normalize input using StandardScaler (Fix Warning)
            input_scaled = scaler.transform(pd.DataFrame(input_df, columns=scaler.feature_names_in_))

            # ✅ Make prediction
            prediction = model.predict(input_scaled)[0]
            result = "Diabetic" if prediction == 1 else "Not Diabetic"

            # ✅ SHAP Explainability
            explainer = shap.LinearExplainer(model, masker=shap.maskers.Independent(
                scaler.transform(pd.DataFrame(
                    [[0] * 8], columns=feature_names
                ))
            ))

            shap_values = explainer.shap_values(input_scaled)[0]
            if isinstance(shap_values, list):  # Handle multi-class models
                shap_values = shap_values[0]

            # Extract importance values
            sorted_features = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
            top_factors = sorted_features[:3]  # Show top 3 contributing factors

            # ✅ Debugging Outputs
            print(f"📌 SHAP Raw Values: {shap_values}")
            print(f"📌 Top Risk Factors: {top_factors}")

            # ✅ Save prediction history if user is logged in
            if request.user.is_authenticated:
                PredictionHistory.objects.create(
                    user=request.user,
                    pregnancies=pregnancies,
                    glucose=glucose,
                    blood_pressure=blood_pressure,
                    skin_thickness=skin_thickness,
                    insulin=insulin,
                    bmi=bmi,
                    diabetes_pedigree_function=diabetes_pedigree_function,
                    age=age,
                    prediction=result
                )

        except Exception as e:
            result = f"Error: {str(e)}"
            print(f"❌ Error Occurred: {result}")

    # ✅ Fetch past predictions for logged-in user
    past_predictions = PredictionHistory.objects.filter(user=request.user).order_by("-created_at")

    return render(request, "predict.html", {
        "result": result,
        "history": past_predictions,
        "top_factors": top_factors  # Send top contributing factors to the template
    })


@login_required
def history(request):
    past_predictions = PredictionHistory.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "history.html", {"predictions": past_predictions})

# --- Result View (Fixing Issues) ---

@login_required
def result(request):
    if request.method == "POST":
        try:
            if model is None or scaler is None:
                raise ValueError("❌ Model, Scaler, or Explainer not loaded!")

            # Get user inputs with defaults to prevent crashes
            pregnancies = int(request.POST.get("n1", 0))
            glucose = float(request.POST.get("n2", 0))
            blood_pressure = float(request.POST.get("n3", 0))
            skin_thickness = float(request.POST.get("n4", 0))
            insulin = float(request.POST.get("n5", 0))
            bmi = float(request.POST.get("n6", 0))
            diabetes_pedigree_function = float(request.POST.get("n7", 0.0))
            age = int(request.POST.get("n8", 0))

            print(f"📌 Input Data: {pregnancies}, {glucose}, {blood_pressure}, {skin_thickness}, {insulin}, {bmi}, {diabetes_pedigree_function}, {age}")

            # Prepare data
            feature_names = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
            input_scaled = scaler.transform(pd.DataFrame(input_data, columns=scaler.feature_names_in_))

            # Predict
            prediction = model.predict(input_scaled)
            result_text = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

            # SHAP Explainability
            shap_values = explainer.shap_values(input_scaled)[0]
            if isinstance(shap_values, list):  # Handle multi-class models
                shap_values = shap_values[0]

            # Extract importance values
            sorted_features = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
            top_factors = sorted_features[:3]  # Show top 3 contributing factors

            # ✅ Debugging Outputs
            print(f"📌 SHAP Raw Values: {shap_values}")
            print(f"📌 Top Risk Factors: {top_factors}")

            # Save history
            PredictionHistory.objects.create(
                user=request.user,
                pregnancies=pregnancies,
                glucose=glucose,
                blood_pressure=blood_pressure,
                skin_thickness=skin_thickness,
                insulin=insulin,
                bmi=bmi,
                diabetes_pedigree_function=diabetes_pedigree_function,
                age=age,
                prediction=result_text
            )

            return render(request, "predict.html", {
                "result": result_text,
                "top_factors": top_factors  # Pass SHAP results to the template
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            return render(request, "predict.html", {"error": "An error occurred during prediction"})

    return redirect("predict")  # Redirect if accessed without POST

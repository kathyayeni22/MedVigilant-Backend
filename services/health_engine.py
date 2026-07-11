import joblib
import numpy as np
from pathlib import Path

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# ---------------- LOAD MODELS ----------------

# Symptoms → Disease
symptom_vectorizer, symptom_model, disease_encoder = joblib.load(
    MODEL_DIR / "symptom_model.pkl"
)

# Prescription
prescription_vectorizer, prescription_model = joblib.load(
    MODEL_DIR / "prescription_model.pkl"
)

# Mental Health
mental_vectorizer, mental_model = joblib.load(
    MODEL_DIR / "mental_health_model.pkl"
)

# Food Recommendation
food_model = joblib.load(MODEL_DIR / "food_model.pkl")
food_encoder = joblib.load(MODEL_DIR / "food_encoder.pkl")
food_disease_encoder = joblib.load(MODEL_DIR / "disease_encoder.pkl")

# BMI
bmi_model = joblib.load(MODEL_DIR / "bmi_model.pkl")
bmi_label_encoder = joblib.load(MODEL_DIR / "bmi_label_encoder.pkl")

# Lab Test
lab_model = joblib.load(MODEL_DIR / "lab_test_model.pkl")
lab_test_encoder = joblib.load(MODEL_DIR / "lab_test_test_encoder.pkl")
lab_label_encoder = joblib.load(MODEL_DIR / "lab_test_label_encoder.pkl")

# ---------------- UTIL FUNCTIONS ----------------

def calculate_bmi(height_cm, weight_kg):
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
    return bmi

def detect_emergency(symptoms):
    emergency_words = [
        "chest pain", "shortness of breath",
        "unconscious", "severe bleeding",
        "paralysis", "heart attack"
    ]
    return any(word in symptoms.lower() for word in emergency_words)

# ---------------- CORE ENGINE ----------------

def run_health_engine(user_input):
    """
    user_input = {
        "symptoms": "headache fatigue stress",
        "height": 165,
        "weight": 70,
        "age": 22,
        "glucose": 110,
        "mental_text": "I feel anxious and sad"
    }
    """

    # 🚨 Emergency first
    if detect_emergency(user_input["symptoms"]):
        return {
            "emergency": True,
            "message": "🚨 Emergency detected. Seek immediate medical attention."
        }

    # 1️⃣ Symptoms → Disease
    sym_vec = symptom_vectorizer.transform([user_input["symptoms"]])
    disease_probs = symptom_model.predict_proba(sym_vec)[0]
    disease_idx = np.argmax(disease_probs)
    disease = disease_encoder.inverse_transform([disease_idx])[0]

    # 2️⃣ BMI
    bmi_value = calculate_bmi(
        user_input["height"],
        user_input["weight"]
    )

    bmi_pred = bmi_model.predict(
        [[user_input["height"], user_input["weight"], bmi_value]]
    )[0]
    bmi_category = bmi_label_encoder.inverse_transform([bmi_pred])[0]

    # 3️⃣ Mental Health
    mental_vec = mental_vectorizer.transform(
        [user_input["mental_text"]]
    )
    mental_risk = mental_model.predict(mental_vec)[0]

    # 4️⃣ Food Recommendation
    disease_enc = food_disease_encoder.transform([disease])[0]

    food_pred = food_model.predict([[
        disease_enc,
        user_input["age"],
        bmi_value,
        user_input["glucose"]
    ]])[0]

    food = food_encoder.inverse_transform([food_pred])[0]

    # 5️⃣ Lab Test Analysis
    test_enc = lab_test_encoder.transform(["Blood Test"])[0]
    lab_pred = lab_model.predict([[test_enc, bmi_value]])[0]
    lab_status = lab_label_encoder.inverse_transform([lab_pred])[0]

    # 6️⃣ Prescription Type
    pres_vec = prescription_vectorizer.transform(
        [user_input["symptoms"]]
    )
    prescription_type = prescription_model.predict(pres_vec)[0]

    return {
        "emergency": False,
        "predicted_disease": disease,
        "bmi": {
            "value": bmi_value,
            "category": bmi_category
        },
        "mental_health_risk": mental_risk,
        "recommended_food": food,
        "lab_test_status": lab_status,
        "prescription_type": prescription_type,
        "advice": "Consult a certified doctor before taking medication"
    }

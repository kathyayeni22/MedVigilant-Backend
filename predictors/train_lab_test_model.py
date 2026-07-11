import joblib
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "models"

model = joblib.load(BASE / "lab_test_model.pkl")
test_encoder = joblib.load(BASE / "lab_test_test_encoder.pkl")
label_encoder = joblib.load(BASE / "lab_test_label_encoder.pkl")

def predict_lab(test_name, value):
    test_name = test_name.lower()

    # check if label exis
    if test_name not in test_encoder.classes_:
        return {
            "test": test_name,
            "status": "unknown test",
            "message": "This lab test is not supported by the model."
        }
    test_enc = test_encoder.transform([test_name])[0]
    X = np.array([[test_enc, value]])
    pred = model.predict(X)[0]

    return label_encoder.inverse_transform([pred])[0]
def recommend_lab_tests_from_symptoms(symptoms):

    symptoms = symptoms.lower()

    mapping = {
        "fever": ["CBC", "Dengue Test", "Malaria Test"],
        "cold": ["CBC"],
        "cough": ["Chest X-ray", "CBC"],
        "diabetes": ["Blood Sugar", "HbA1c"],
        "fatigue": ["Thyroid Test", "Vitamin D"],
        "chest pain": ["ECG", "Troponin Test"],
        "headache": ["MRI", "CT Scan"]
    }

    suggested_tests = []

    for key in mapping:
        if key in symptoms:
            suggested_tests.extend(mapping[key])

    if not suggested_tests:
        return ["General Blood Test", "Urine Test"]

    return list(set(suggested_tests))
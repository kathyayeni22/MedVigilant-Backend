import joblib
import numpy as np
from pathlib import Path

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[1] / "models"

MODEL_PATH = BASE_DIR / "bmi_model.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "bmi_label_encoder.pkl"

# ---------------- LOAD MODEL ----------------
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# ---------------- PREDICT FUNCTION ----------------
def predict_bmi(height_cm, weight_kg):
#    def predict_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 2)

    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25:
        category = "Normal"
    elif 25 <= bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return {
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "category": category
    }



# ---------------- TEST ----------------
if __name__ == "__main__":
    result = predict_bmi(170, 72)
    print(result)

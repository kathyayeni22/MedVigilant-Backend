import joblib
import numpy as np
from pathlib import Path

# Load models
BASE = Path(__file__).resolve().parents[1] / "models"

model = joblib.load(BASE / "food_model.pkl")
disease_encoder = joblib.load(BASE / "disease_encoder.pkl")
food_encoder = joblib.load(BASE / "food_encoder.pkl")

def recommend_food(disease, age, bmi, glucose):

    prompt = f"""
    Suggest a personalized diet plan for:
    Disease: {disease}
    Age: {age}
    BMI: {bmi}
    Glucose: {glucose}

    Give 5 simple food recommendations.
    """

    response = get_ai_response(prompt)

    return {
        "recommended_foods": response
    }
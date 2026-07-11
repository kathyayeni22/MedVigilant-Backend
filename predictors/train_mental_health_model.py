import joblib
from pathlib import Path
from services.ai_chatbot import get_ai_response

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mental_health_model.pkl"

vectorizer, model = joblib.load(MODEL_PATH)


def predict_mental_health(text):
    try:
        # 1. Prediction
        X = vectorizer.transform([text])
        risk = model.predict(X)[0]

        # 2. Interpretation
        if str(risk).lower() in ["high", "severe", "depressed"]:
            status = "High Risk"
        elif str(risk).lower() in ["moderate"]:
            status = "Moderate"
        else:
            status = "Low"

        # 3. Chatbot Suggestion
        if status == "High Risk":
            suggestion = get_ai_response(
                f"User says: '{text}'. They are mentally stressed. Give short calming advice."
            )
        elif status == "Moderate":
            suggestion = "Try meditation, take breaks, and maintain good sleep."
        else:
            suggestion = "You are doing well. Keep a healthy routine."

        return {
            "status": "success",
            "input": text,
            "risk_level": risk,
            "mental_status": status,
            "suggestion": suggestion
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
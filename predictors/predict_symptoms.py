import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "symptom_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "symptom_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "symptom_label_encoder.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

def predict_diseases(symptoms, top_n=3):

    # convert list → string if needed
    if isinstance(symptoms, list):
        symptoms_text = " ".join(symptoms)
    else:
        symptoms_text = symptoms

    X = vectorizer.transform([symptoms_text])
    probabilities = model.predict_proba(X)[0]

    disease_probs = list(zip(label_encoder.classes_, probabilities))
    disease_probs.sort(key=lambda x: x[1], reverse=True)

    results = []
    for disease, prob in disease_probs[:top_n]:
        results.append({
            "disease": disease.lower(),
            "confidence": round(float(prob) * 100, 2)
        })

    return results


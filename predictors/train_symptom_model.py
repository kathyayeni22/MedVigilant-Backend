import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "symptoms.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "symptom_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "symptom_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "symptom_label_encoder.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

def train_model():
    df = pd.read_csv(DATASET_PATH)

    X = df["symptoms"]
    y = df["disease"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_vec, y_enc)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print("✅ Symptom model trained successfully (balanced dataset)")

if __name__ == "__main__":
    train_model()

import pandas as pd
from pathlib import Path

# backend directory
BASE_DIR = Path(__file__).resolve().parent

# datasets directory inside backend
DATA_DIR = BASE_DIR / "datasets"

def load_csv(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"❌ CSV file not found: {path}")
    return pd.read_csv(path)

# Load datasets
food_df = load_csv("food_recommendations.csv")
lab_df = load_csv("lab_test_ranges.csv")
prescription_df = load_csv("prescription_keywords.csv")
bmi_df = load_csv("bmi_health.csv")
emergency_df = load_csv("emergency_symptoms.csv")
mental_df = load_csv("mental_health_questions.csv")

# ✅ NEW: Symptoms → Disease dataset
symptoms_disease_df = load_csv("symptoms_disease.csv")

print("✅ All CSV datasets loaded successfully")

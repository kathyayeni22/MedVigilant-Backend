from groq import Groq
# -----------------------------
# GROQ CLIENT
# -----------------------------

from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
# -----------------------------
# AI DOCTOR RECOMMENDER
# -----------------------------
def recommend_doctor(symptoms):

    try:

        prompt = f"""
        A patient has these symptoms:

        {symptoms}

        Suggest:

        1. Most suitable doctor specialization
        2. Severity level (Low / Medium / High)
        3. Whether hospital visit is needed
        4. Basic precautions

        Return ONLY JSON.

        Example:
        {{
            "specialist": "Cardiologist",
            "severity": "High",
            "hospital_visit": true,
            "precautions": [
                "Avoid stress",
                "Monitor blood pressure"
            ]
        }}
        """

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI medical triage assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=400
        )

        response = completion.choices[0].message.content

        import json
        return json.loads(response)

    except Exception as e:

        print("DOCTOR AI ERROR:", e)

        return {
            "specialist": "General Physician",
            "severity": "Unknown",
            "hospital_visit": False,
            "precautions": [
                "Consult doctor if symptoms worsen"
            ]
        }
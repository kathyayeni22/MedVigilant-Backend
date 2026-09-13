import os
import json
import re
from dotenv import load_dotenv
from groq import Groq

# Load .env locally
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "openai/gpt-oss-20b"


# ---------------------------------------------------------
# Safe default response
# ---------------------------------------------------------
def empty_prescription_response(message="Unable to analyze the prescription."):
    return {
        "doctor": "",
        "patient": "",
        "clinic": "",
        "summary": message,
        "medicines": []
    }


# ---------------------------------------------------------
# Create Groq client
# ---------------------------------------------------------
def get_groq_client():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it to the .env file locally or Render Environment Variables."
        )

    return Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------
# Clean JSON returned by the AI
# ---------------------------------------------------------
def clean_json_response(text):
    if not text:
        raise ValueError("Empty response received from Groq.")

    text = text.strip()

    # Remove markdown code fences if the model adds them
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


# ---------------------------------------------------------
# Validate and normalize AI result
# ---------------------------------------------------------
def normalize_prescription_result(data):

    if not isinstance(data, dict):
        return empty_prescription_response(
            "Invalid prescription analysis response."
        )

    doctor = str(data.get("doctor", "") or "")
    patient = str(data.get("patient", "") or "")
    clinic = str(data.get("clinic", "") or "")
    summary = str(data.get("summary", "") or "")

    medicines = data.get("medicines", [])

    if not isinstance(medicines, list):
        medicines = []

    normalized_medicines = []

    for medicine in medicines:

        if not isinstance(medicine, dict):
            continue

        normalized_medicines.append({
            "name": str(medicine.get("name", "") or ""),
            "dosage": str(medicine.get("dosage", "") or ""),
            "duration": str(medicine.get("duration", "") or ""),
            "instructions": str(
                medicine.get("instructions", "") or ""
            ),
            "warning": str(
                medicine.get("warning", "") or ""
            )
        })

    return {
        "doctor": doctor,
        "patient": patient,
        "clinic": clinic,
        "summary": summary,
        "medicines": normalized_medicines
    }


# ---------------------------------------------------------
# Analyze prescription OCR text
# ---------------------------------------------------------
def analyze_prescription(ocr_text):

    if not ocr_text or not ocr_text.strip():
        return empty_prescription_response(
            "No prescription text was detected."
        )

    try:

        client = get_groq_client()

        prompt = f"""
You are analyzing OCR text extracted from a medical prescription.

Your job is ONLY to organize information that is actually present
in the OCR text.

Do NOT invent:
- medicines
- dosage
- duration
- doctor name
- patient name
- clinic name
- medical conditions
- instructions

If a value is not clearly available, return an empty string.

Return ONLY valid JSON.

Required JSON structure:

{{
    "doctor": "",
    "patient": "",
    "clinic": "",
    "summary": "",
    "medicines": [
        {{
            "name": "",
            "dosage": "",
            "duration": "",
            "instructions": "",
            "warning": ""
        }}
    ]
}}

Important:
- "name" = medicine name
- "dosage" = strength/dose such as 500 mg
- "duration" = number of days/weeks if present
- "instructions" = directions such as after food, before food, once daily, etc.
- "warning" = only warnings explicitly present in the prescription
- Do not create medical advice that is not written in the prescription
- Keep the summary short
- If no medicine is confidently identifiable, return an empty medicines list

OCR TEXT:

{ocr_text}
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical prescription OCR "
                        "information extraction assistant. "
                        "Extract only information present in the OCR text."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_completion_tokens=1500,
            response_format={
                "type": "json_object"
            },
            include_reasoning=False
        )

        content = response.choices[0].message.content

        cleaned = clean_json_response(content)

        result = json.loads(cleaned)

        return normalize_prescription_result(result)

    except json.JSONDecodeError as e:

        print(
            "Prescription JSON parsing error:",
            str(e),
            flush=True
        )

        return empty_prescription_response(
            "The AI returned an invalid prescription analysis."
        )

    except Exception as e:

        print(
            "Prescription analysis error:",
            repr(e),
            flush=True
        )

        return empty_prescription_response(
            "Unable to analyze the prescription."
        )
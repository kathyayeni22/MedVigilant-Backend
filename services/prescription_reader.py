
from groq import Groq
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize EasyOCR once
from services.ocr_reader import get_reader


# -----------------------------
# OCR EXTRACTION
# -----------------------------
def extract_prescription_text(file_path):
    try:
        print("Reading prescription:", file_path)

        reader = get_reader()

        result = reader.readtext(
            file_path,
            detail=0
        )
        text = "\n".join(result)

        print("OCR TEXT:")
        print(text)

        if not text.strip():
            raise Exception("No text detected from prescription.")

        return text

    except Exception as e:
        print("OCR ERROR:", e)
        raise e


# -----------------------------
# CLEAN JSON
# -----------------------------
def clean_json(text):

    text = text.replace("```json", "")
    text = text.replace("```", "")

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text.strip()


# -----------------------------
# AI ANALYSIS
# -----------------------------
def analyze_prescription(extracted_text):

    try:

        prompt = f"""
Analyze this medical prescription carefully.

Return ONLY valid JSON.

{{
  "patient_name": "",
  "doctor_name": "",
  "medicines": [
    {{
      "name": "",
      "purpose": "",
      "dosage": "",
      "warning": ""
    }}
  ],
  "summary": ""
}}

Rules:
- Fix OCR spelling mistakes.
- Detect medicine names.
- Explain medicine purpose simply.
- Keep dosage short.
- Add warnings if available.

Prescription:
{extracted_text}
"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=700
        )

        response = completion.choices[0].message.content

        cleaned = clean_json(response)

        return json.loads(cleaned)

    except Exception as e:

        print("Prescription AI Error:", e)

        return {
            "patient_name": "",
            "doctor_name": "",
            "medicines": [],
            "summary": "Unable to analyze prescription."
        }
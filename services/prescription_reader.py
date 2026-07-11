import pytesseract
from PIL import Image
from groq import Groq
import json
import re

# -----------------------------
# TESSERACT PATH
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

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
# OCR EXTRACTION
# -----------------------------
from PIL import Image
import pytesseract

def extract_prescription_text(file_path):

    try:
        print("Opening:", file_path)

        image = Image.open(file_path)

        print("Image opened successfully")

        text = pytesseract.image_to_string(image)

        print("OCR Output:")
        print(text)

        return text

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


# -----------------------------
# CLEAN JSON RESPONSE
# -----------------------------
def clean_json(text):

    # Remove markdown
    text = text.replace("```json", "")
    text = text.replace("```", "")

    # Find first {
    start = text.find("{")

    # Find last }
    end = text.rfind("}")

    # Extract only JSON
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

Extract in JSON format:

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
- Detect real medicine names correctly.
- Explain medicine purpose in simple words.
- Keep dosage simple.
- Add medicine warnings if possible.
- Return ONLY valid JSON.

Prescription Text:
{extracted_text}
"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical prescription analyzer. "
                        "Always return valid JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=700
        )

        response_text = completion.choices[0].message.content

        cleaned = clean_json(response_text)

        parsed = json.loads(cleaned)

        return parsed

    except Exception as e:

        print("PRESCRIPTION AI ERROR:", e)

        return {
            "error": "Unable to analyze prescription"
        }
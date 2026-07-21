import os
import json
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

OCR_API_KEY = os.getenv("OCR_SPACE_API_KEY")

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# -----------------------------
# OCR USING OCR.SPACE
# -----------------------------
def extract_prescription_text(file_path):

    payload = {
        "apikey": OCR_API_KEY,
        "language": "eng",
        "isOverlayRequired": False,
    }

    with open(file_path, "rb") as f:
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": f},
            data=payload
        )

    result = response.json()

    if result.get("IsErroredOnProcessing"):
        raise Exception(result.get("ErrorMessage"))

    text = result["ParsedResults"][0]["ParsedText"]

    print(text)

    return text


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

    prompt = f"""
Analyze this medical prescription.

Return ONLY valid JSON.

{{
  "patient_name":"",
  "doctor_name":"",
  "medicines":[
    {{
      "name":"",
      "purpose":"",
      "dosage":"",
      "warning":""
    }}
  ],
  "summary":""
}}

Prescription:

{extracted_text}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role":"system",
                "content":"Return only JSON."
            },
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0.2,
        max_tokens=700
    )

    response = completion.choices[0].message.content

    cleaned = clean_json(response)

    return json.loads(cleaned)
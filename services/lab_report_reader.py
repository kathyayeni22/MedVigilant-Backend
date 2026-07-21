import os
import json
import re
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
def extract_lab_text(file_path):

    payload = {
        "apikey": OCR_API_KEY,
        "language": "eng",
        "isOverlayRequired": False,
    }

    with open(file_path, "rb") as f:
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": f},
            data=payload,
            timeout=60
        )

    result = response.json()

    if result.get("IsErroredOnProcessing"):
        raise Exception(result.get("ErrorMessage"))

    parsed_results = result.get("ParsedResults", [])

    if not parsed_results:
        raise Exception("No text detected.")

    text = parsed_results[0].get("ParsedText", "")

    print("LAB OCR:")
    print(text)

    return text


# -----------------------------
# CLEAN JSON
# -----------------------------
def clean_json(text):

    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        raise Exception("No valid JSON returned by AI.")

    text = text[start:end]

    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    return text


# -----------------------------
# AI ANALYSIS
# -----------------------------
def analyze_lab_report(extracted_text):

    try:

        prompt = f"""
Analyze this medical lab report.

Return ONLY valid JSON.

{{
  "patient_name":"",
  "tests":[
    {{
      "test_name":"",
      "value":"",
      "status":""
    }}
  ],
  "health_risks":[
    ""
  ],
  "summary":"",
  "advice":[
    ""
  ]
}}

Lab Report:

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
            temperature=0.1,
            max_tokens=900
        )

        response = completion.choices[0].message.content

        print("RAW AI RESPONSE:")
        print(response)

        cleaned = clean_json(response)

        return json.loads(cleaned)

    except Exception as e:

        print("LAB AI ERROR:", e)

        return {
            "patient_name": "",
            "tests": [],
            "health_risks": [],
            "summary": "Unable to analyze the report.",
            "advice": [
                "Please upload a clearer image."
            ]
        }
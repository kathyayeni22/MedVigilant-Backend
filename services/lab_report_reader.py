
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
# OCR
# -----------------------------
def extract_lab_text(file_path):

    print("STEP 1: Entered extract_lab_text")

    try:
        print("STEP 2: Loading EasyOCR Reader")

        reader = get_reader()

        print("STEP 3: Reader loaded successfully")

        result = reader.readtext(
            file_path,
            detail=0
        )

        print("STEP 4: OCR completed")

        text = "\n".join(result)

        print("STEP 5: Extracted Text:")
        print(text)

        if not text.strip():
            raise Exception("No text detected.")

        return text

    except Exception as e:
        print("OCR ERROR:", e)
        raise e

# -----------------------------
# CLEAN JSON
# -----------------------------
def clean_json(text):

    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        return ""

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

        print(response)

        cleaned = clean_json(response)

        parsed = json.loads(cleaned)

        return parsed

    except Exception as e:

        print("LAB AI ERROR:", e)

        return {
            "patient_name": "",
            "tests": [],
            "health_risks": [],
            "summary": "Unable to analyze the report.",
            "advice": [
                "Upload a clearer image."
            ]
        }
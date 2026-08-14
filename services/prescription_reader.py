import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def clean_json(text):
    text = text.replace("```json", "")
    text = text.replace("```", "")

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text.strip()


def analyze_prescription(extracted_text):

    prompt = f"""
You are an expert medical prescription OCR correction and analysis assistant.

The OCR text below may contain spelling mistakes, missing spaces,
incorrect characters, and incorrectly recognized medicine names.

Your job is to carefully interpret the OCR text.

IMPORTANT RULES:

1. Do NOT invent information.
2. Do NOT guess a patient's name.
3. Do NOT treat clinic information as a patient name.
4. Do NOT treat social media handles, websites, emails, or clinic services
   as patient names.
5. If patient name is not clearly present, return "".
6. If doctor name is not clearly present, return "".
7. Correct obvious OCR mistakes in medicine names ONLY when the medicine
   can be identified with high confidence.
8. Preserve dosage and duration when they are present.

9. IMPORTANT DOSAGE RULE:
   "Tab", "Tablet", "Cap", "Capsule", "Syr", "Syrup",
   "Inj", "Injection", "Gel", "Cream" and similar words
   are medicine FORM/TYPE, NOT dosage.

10. Dosage means strength such as:
   - 625mg
   - 40mg
   - 500mg
   - 10mg
   - 5ml
   - 100mg

11. Never return "Tab" as the dosage.

12. If OCR contains a medicine name followed by a number and unit,
   such as "Augmentin625mg", interpret it as:
   name = "Augmentin"
   dosage = "625mg"

13. If dosage is unclear, return "" instead of guessing.
9. If dosage is unclear, return "" instead of guessing.
10. If a medicine cannot be confidently identified, keep the OCR version
    rather than inventing a medicine.
11. Clinic information should go into clinic_name.

Examples of clinic/service information that must NOT be patient names:

- Smile Designing
- Teeth Whitening
- Dental Implants
- General Dentistry
- THE WHITETUSK
- @whitetuskdental
- websites
- email addresses

Return ONLY valid JSON.

JSON FORMAT:

{{
    "patient_name": "",
    "doctor_name": "",
    "clinic_name": "",
    "medicines": [
        {{
            "name": "",
            "purpose": "",
            "dosage": "",
            "duration": "",
            "warning": ""
        }}
    ],
    "summary": ""
}}

OCR TEXT:

{extracted_text}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. Never invent medical information."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=1000
    )

    response = completion.choices[0].message.content

    print("========== RAW AI RESPONSE ==========")
    print(response)
    print("=====================================")

    cleaned = clean_json(response)

    return json.loads(cleaned)
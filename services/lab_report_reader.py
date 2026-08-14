import os
import json
import re

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# =========================================================
# CLEAN JSON
# =========================================================

def clean_json(text):

    if not text:
        raise Exception("Empty response from AI.")

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise Exception("No valid JSON found.")

    text = text[start:end + 1]

    # Remove trailing commas
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    return text.strip()


# =========================================================
# LAB REPORT ANALYSIS
# =========================================================

def analyze_lab_report(extracted_text):

    try:

        prompt = f"""
You are an expert medical laboratory report analyzer.

The OCR text below comes from a laboratory report.

The OCR may contain:
- spelling mistakes
- missing spaces
- headers
- footers
- websites
- phone numbers
- units
- reference ranges
- formatting errors

Your task is to extract ONLY reliable information from the report.

IMPORTANT RULES:

1. NEVER invent information.

2. Patient identification:
   - Look for labels such as:
     "Name"
     "Patient Name"
     "PatientID"
     "Patient ID"
   - If a real patient name is clearly available, return it.
   - If only a Patient ID is available, use that ID as patient_name.
   - Example:
     PatientID: PN2
     -> patient_name = "PN2"

3. Ignore:
   - laboratory/company names
   - websites
   - emails
   - phone numbers
   - report IDs
   - collection dates
   - report dates
   - doctor/hospital information

4. Extract only actual laboratory tests.

5. Do NOT include:
   - section headings
   - units
   - reference ranges
   - "TEST DESCRIPTION"
   - "RESULT"
   - "REF RANGE"

6. Correct obvious OCR mistakes in test names.

7. Pair each test with its correct result.

8. Use the reference range to determine status.

9. Status must be exactly one of:
   "Low"
   "Normal"
   "High"
   ""

10. If a reference range is missing or unclear:
    - still extract the test if its value is clear
    - set status to ""

11. Do NOT calculate a status when the reference range is unavailable.

12. Do NOT confuse a reference-range number with the actual test result.

13. For example:

    Haemoglobin
    15
    13-17

    means:

    test_name = "Haemoglobin"
    value = "15"
    status = "Normal"

14. Another example:

    MCV
    80.00
    81-101

    means:

    test_name = "MCV"
    value = "80.00"
    status = "Low"

15. If a test has no clear value, DO NOT include it.

16. health_risks:
    Include only clearly abnormal results.
    Do not invent diseases.

17. summary:
    Give a short, factual summary based ONLY on clearly abnormal
    laboratory values.

    If there are no clearly abnormal values, say:
    "The reported laboratory values are within the provided reference ranges."

    If abnormal values exist, mention ONLY those abnormal values.

    Do NOT say vague phrases such as:
    "may indicate some health issues."

18. health_risks:
    Mention only clearly abnormal results.

    Example:
    "MCV is below the provided reference range."

    Do NOT convert a laboratory abnormality into a disease diagnosis.

19. advice:
    Give 1 or 2 short general recommendations.

    If abnormal values exist:
    "Discuss the abnormal results with a healthcare professional."

    Do NOT prescribe medicines.

20. IMPORTANT:
    Never assign Low or High status unless the OCR clearly provides
    both the test value and its corresponding reference range.

21. If the reference range is missing or unclear:
    status = ""

22. Do not guess the reference range from medical knowledge.
    Use ONLY the reference range visible in the OCR.

23. Return ONLY valid JSON.

24. Do NOT return markdown.

25. Do NOT return ```json.

26. Every JSON key must use double quotes.

EXPECTED JSON:

{{
    "patient_name": "",
    "tests": [
        {{
            "test_name": "",
            "value": "",
            "status": "Low/Normal/High"
        }}
    ],
    "health_risks": [],
    "summary": "",
    "advice": []
}}

OCR TEXT:

{extracted_text}
"""

        completion = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical laboratory report analyzer. "
                        "Return ONLY valid JSON. "
                        "Never invent medical information."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            response_format={
                "type": "json_object"
            },

            temperature=0,

            max_tokens=1200
        )

        response = completion.choices[0].message.content

        print("========== RAW LAB AI RESPONSE ==========")
        print(response)
        print("=========================================")

        cleaned = clean_json(response)

        try:

            result = json.loads(cleaned)

        except json.JSONDecodeError:

            print("JSON parsing failed. Attempting repair...")

            repair_prompt = f"""
Fix the following invalid JSON.

Rules:
- Return ONLY valid JSON.
- Do not add information.
- Do not remove information.
- Use double quotes.
- No markdown.

INVALID JSON:

{cleaned}
"""

            repaired = client.chat.completions.create(

                model="llama-3.3-70b-versatile",

                messages=[
                    {
                        "role": "system",
                        "content": "Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": repair_prompt
                    }
                ],

                response_format={
                    "type": "json_object"
                },

                temperature=0,

                max_tokens=1200
            )

            repaired_text = repaired.choices[0].message.content

            repaired_text = clean_json(repaired_text)

            result = json.loads(repaired_text)

        # =================================================
        # SAFETY / OUTPUT NORMALIZATION
        # =================================================

        if "patient_name" not in result:
            result["patient_name"] = ""

        if "tests" not in result:
            result["tests"] = []

        if "health_risks" not in result:
            result["health_risks"] = []

        if "summary" not in result:
            result["summary"] = ""

        if "advice" not in result:
            result["advice"] = []

        # -------------------------------------------------
        # Remove incomplete tests
        # -------------------------------------------------

        valid_tests = []

        for test in result["tests"]:

            if not isinstance(test, dict):
                continue

            test_name = str(
                test.get("test_name", "")
            ).strip()

            value = str(
                test.get("value", "")
            ).strip()

            status = str(
                test.get("status", "")
            ).strip()

            if not test_name or not value:
                continue

            if status not in ["Low", "Normal", "High", ""]:
                status = ""

            valid_tests.append({
                "test_name": test_name,
                "value": value,
                "status": status
            })

        result["tests"] = valid_tests

        return result

    except Exception as e:

        import traceback

        print("========== LAB AI ERROR ==========")

        traceback.print_exc()

        print("==================================")

        return {
            "patient_name": "",
            "tests": [],
            "health_risks": [],
            "summary": "Unable to analyze the laboratory report.",
            "advice": [
                "Please upload a clearer laboratory report."
            ]
        }


# =========================================================
# OPTIONAL ALIAS
# =========================================================
# Keep this only if main.py imports extract_lab_text.
# Your current main.py does import it.
# =========================================================

def extract_lab_text(image_path):
    """
    Uses the common OCR function from ocr_utils.py.
    """

    from ocr.ocr_utils import extract_text

    return extract_text(image_path)
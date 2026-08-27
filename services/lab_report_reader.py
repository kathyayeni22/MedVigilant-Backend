import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# GROQ CLIENT
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is not configured.")

client = Groq(api_key=api_key)


# ============================================================
# CLEAN JSON
# ============================================================

def clean_json(text):
    """
    Removes markdown code fences and extracts the JSON object.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove markdown fences
    text = text.replace("```json", "")
    text = text.replace("```JSON", "")
    text = text.replace("```", "")

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text.strip()


# ============================================================
# SAFE FALLBACK
# ============================================================

def fallback_result():
    return {
        "patient_name": "",
        "tests": [],
        "health_risks": [],
        "summary": "Unable to analyze the laboratory report.",
        "advice": [
            "Please upload a clearer laboratory report."
        ]
    }


# ============================================================
# LAB REPORT ANALYSIS
# ============================================================

def analyze_lab_report(extracted_text):

    # --------------------------------------------------------
    # Validate OCR text
    # --------------------------------------------------------

    if not extracted_text or len(extracted_text.strip()) < 20:
        return fallback_result()

    print("\n========== LAB OCR TEXT ==========")
    print(extracted_text)
    print("===================================\n")

    # --------------------------------------------------------
    # AI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a laboratory report analysis assistant.

You will receive OCR text extracted from a medical laboratory report.

The OCR text may contain:
- missing spaces
- incorrect characters
- broken words
- misplaced units
- table formatting problems
- repeated text
- headers mixed with test results

Your task is to carefully reconstruct the laboratory test information
from the OCR text.

IMPORTANT RULES:

1. Do NOT invent laboratory values.

2. Do NOT invent tests that are not present in the OCR.

3. Do NOT guess missing values.

4. Do NOT invent a patient name.

5. If a patient name is not clearly present, return "".

6. Ignore laboratory/company/clinic names when determining patient name.

7. "Name" followed by an actual person's name can be treated as patient_name.

8. Do NOT treat:
   - laboratory name
   - hospital name
   - clinic name
   - website
   - email
   - phone number
   - doctor name
   as patient_name.

9. Extract every laboratory test whose test name and result/value
   can be confidently identified.

10. Preserve the numerical value exactly when possible.

11. Preserve the unit when available.

12. Preserve the reference range when available.

13. If a unit is not clearly available, return "".

14. If a reference range is not clearly available, return "".

15. Determine status using the reference range when a reference
    range is explicitly available.

16. Status should normally be one of:
    - "Normal"
    - "High"
    - "Low"
    - "Unknown"

17. Do NOT diagnose diseases solely from one laboratory value.

18. health_risks should contain only reasonable observations based
    on clearly abnormal laboratory values.

19. Do not claim that an abnormal laboratory result definitely means
    the patient has a disease.

20. The summary should briefly explain the overall report.

21. Advice should contain simple general health advice and,
    when appropriate, suggest discussing abnormal findings with
    a qualified healthcare professional.

22. Do not provide medication prescriptions.

23. Do not invent medication names.

24. Ignore report headers, addresses, phone numbers, emails,
    websites, report IDs, patient IDs, collection dates, etc.
    unless needed for patient identification.

25. IMPORTANT:
    OCR may separate a test name and value across multiple lines.
    Reconstruct them when the relationship is clear.

26. IMPORTANT:
    OCR may contain sections such as:
       HAEMATOLOGY
       COMPLETE BLOOD COUNT
       RBC INDICES
       PLATELET INDICES

    These are section names, NOT individual tests.

27. Extract individual tests such as:
       Haemoglobin
       Total Leucocyte Count
       Neutrophils
       Lymphocytes
       Eosinophils
       Monocytes
       Basophils
       Absolute Neutrophils
       Absolute Lymphocytes
       RBC Count
       MCV
       MCH
       MCHC
       Hct
       RDW-CV
       RDW-SD
       Platelet Count
       MPV
    whenever their values are clearly present.

28. Do not confuse reference ranges with result values.

29. For example, if OCR contains:

       Haemoglobin
       15
       13-17

    return:

       test_name = "Haemoglobin"
       value = "15"
       reference_range = "13-17"
       status = "Normal"

30. If OCR contains:

       MCV
       80.00
       81-101
       fL

    return:

       test_name = "MCV"
       value = "80.00"
       reference_range = "81-101"
       unit = "fL"
       status = "Low"

31. If OCR contains:

       MCHC
       37.50
       31.5-34.5

    return:

       test_name = "MCHC"
       value = "37.50"
       reference_range = "31.5-34.5"
       status = "High"

32. If a result is exactly at the reference boundary, treat it as
    "Normal" unless the report itself indicates otherwise.

33. Return ONLY valid JSON.

JSON FORMAT:

{{
    "patient_name": "",
    "tests": [
        {{
            "test_name": "",
            "value": "",
            "unit": "",
            "reference_range": "",
            "status": ""
        }}
    ],
    "health_risks": [],
    "summary": "",
    "advice": []
}}

OCR TEXT:

{extracted_text}
"""

    # --------------------------------------------------------
    # GROQ REQUEST
    # --------------------------------------------------------

    try:

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful laboratory report "
                        "OCR correction and analysis assistant. "
                        "Return only valid JSON. "
                        "Never invent laboratory values."
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

            max_tokens=2500
        )

        response = completion.choices[0].message.content

        print("\n========== RAW GROQ LAB RESPONSE ==========")
        print(response)
        print("===========================================\n")

        # ----------------------------------------------------
        # CLEAN JSON
        # ----------------------------------------------------

        cleaned = clean_json(response)

        if not cleaned:
            print("ERROR: Empty AI response")
            return fallback_result()

        # ----------------------------------------------------
        # PARSE JSON
        # ----------------------------------------------------

        result = json.loads(cleaned)

        # ----------------------------------------------------
        # ENSURE REQUIRED FIELDS EXIST
        # ----------------------------------------------------

        if not isinstance(result, dict):
            return fallback_result()

        result.setdefault("patient_name", "")
        result.setdefault("tests", [])
        result.setdefault("health_risks", [])
        result.setdefault(
            "summary",
            "No summary generated."
        )
        result.setdefault("advice", [])

        # ----------------------------------------------------
        # SAFETY CHECK TYPES
        # ----------------------------------------------------

        if not isinstance(result["tests"], list):
            result["tests"] = []

        if not isinstance(result["health_risks"], list):
            result["health_risks"] = []

        if not isinstance(result["advice"], list):
            result["advice"] = []

        # ----------------------------------------------------
        # CLEAN TEST OBJECTS
        # ----------------------------------------------------

        cleaned_tests = []

        for test in result["tests"]:

            if not isinstance(test, dict):
                continue

            cleaned_test = {
                "test_name": str(
                    test.get("test_name", "")
                ).strip(),

                "value": str(
                    test.get("value", "")
                ).strip(),

                "unit": str(
                    test.get("unit", "")
                ).strip(),

                "reference_range": str(
                    test.get("reference_range", "")
                ).strip(),

                "status": str(
                    test.get("status", "Unknown")
                ).strip()
            }

            # Only keep tests with a name and value
            if (
                cleaned_test["test_name"]
                and cleaned_test["value"]
            ):
                cleaned_tests.append(cleaned_test)

        result["tests"] = cleaned_tests

        # ----------------------------------------------------
        # CLEAN RISKS
        # ----------------------------------------------------

        result["health_risks"] = [
            str(x).strip()
            for x in result["health_risks"]
            if str(x).strip()
        ]

        # ----------------------------------------------------
        # CLEAN ADVICE
        # ----------------------------------------------------

        result["advice"] = [
            str(x).strip()
            for x in result["advice"]
            if str(x).strip()
        ]

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        result["summary"] = str(
            result.get("summary", "")
        ).strip()

        if not result["summary"]:
            result["summary"] = (
                "Laboratory report analyzed successfully."
            )

        print("\n========== FINAL LAB RESULT ==========")
        print(json.dumps(result, indent=4))
        print("======================================\n")

        return result

    # --------------------------------------------------------
    # JSON ERROR
    # --------------------------------------------------------

    except json.JSONDecodeError as e:

        print("\nJSON PARSE ERROR:")
        print(str(e))
        print("AI RESPONSE:")
        print(response if "response" in locals() else "NO RESPONSE")

        return fallback_result()

    # --------------------------------------------------------
    # GROQ / API / OTHER ERROR
    # --------------------------------------------------------

    except Exception as e:

        print("\n========== LAB ANALYSIS ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        print("========================================\n")

        return fallback_result()
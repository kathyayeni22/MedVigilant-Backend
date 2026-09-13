# services/lab_report_reader.py

import re
import traceback


# ============================================================
# REFERENCE RANGES
# ============================================================

REFERENCE_RANGES = {
    "Haemoglobin": (13.0, 17.0),
    "Total Leucocyte Count": (4000.0, 10000.0),

    "Neutrophils": (40.0, 80.0),
    "Lymphocytes": (20.0, 40.0),
    "Eosinophils": (1.0, 6.0),
    "Monocytes": (2.0, 10.0),
    "Basophils": (0.0, 1.0),

    "Absolute Neutrophils": (2000.0, 7000.0),
    "Absolute Lymphocytes": (1000.0, 3000.0),
    "Absolute Eosinophils": (20.0, 500.0),
    "Absolute Monocytes": (200.0, 1000.0),

    "RBC Count": (4.5, 5.5),

    "MCV": (81.0, 101.0),
    "MCH": (27.0, 32.0),
    "MCHC": (31.5, 34.5),
    "Hct": (40.0, 50.0),

    "RDW-CV": (11.6, 14.0),
    "RDW-SD": (39.0, 46.0),

    "Platelet Count": (150000.0, 410000.0),
    "MPV": (7.5, 11.5),
}


# ============================================================
# OCR CLEANING
# ============================================================

def clean_ocr_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    replacements = {
        "COMPLETEBLOODCOUNT": "COMPLETE BLOOD COUNT",
        "TESTDESCRIPTION": "TEST DESCRIPTION",
        "REF.RANGE": "REF RANGE",

        "DifferentialLeucocyteCount":
            "Differential Leucocyte Count",

        "AbsoluteLeucocyteCount":
            "Absolute Leucocyte Count",

        "RBCIndices":
            "RBC Indices",

        "PlateletsIndices":
            "Platelets Indices",

        "TotalLeucocyteCount":
            "Total Leucocyte Count",

        "PatientID":
            "Patient ID",

        "Age/Gender":
            "Age / Gender",

        "ReportID":
            "Report ID",

        "ReferredBy":
            "Referred By",

        "ReportDate":
            "Report Date",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalize unicode dashes
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")

    # Normalize unicode colon
    text = text.replace("：", ":")

    return text


# ============================================================
# LINE NORMALIZATION
# ============================================================

def get_clean_lines(text: str):

    text = clean_ocr_text(text)

    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        # Normalize multiple spaces
        line = re.sub(r"\s+", " ", line)

        # Normalize spaces around hyphen
        line = re.sub(r"\s*-\s*", "-", line)

        lines.append(line)

    return lines


# ============================================================
# NUMBER PARSER
# ============================================================

def extract_number(value):

    if value is None:
        return None

    match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        str(value)
    )

    if not match:
        return None

    try:
        return float(match.group())

    except ValueError:
        return None


# ============================================================
# RANGE PARSER
# ============================================================

def extract_range(value):

    if not value:
        return None

    match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*-\s*"
        r"([-+]?\d+(?:\.\d+)?)",
        str(value)
    )

    if not match:
        return None

    try:
        return (
            float(match.group(1)),
            float(match.group(2))
        )

    except ValueError:
        return None


# ============================================================
# PATIENT NAME
# ============================================================

def extract_patient_name(text: str) -> str:

    lines = get_clean_lines(text)

    metadata = {
        "patientid",
        "patient id",
        "age/gender",
        "age / gender",
        "reportid",
        "report id",
        "referredby",
        "referred by",
        "collectiondate",
        "collection date:",
        "phoneno.",
        "phone no.",
        "reportdate",
        "report date",
        "haematology",
    }

    for i, line in enumerate(lines):

        normalized = (
            line.lower()
            .replace(":", "")
            .strip()
        )

        if normalized != "name":
            continue

        if i + 1 >= len(lines):
            return ""

        next_line = lines[i + 1].strip()

        next_normalized = (
            next_line
            .lower()
            .replace(":", "")
            .strip()
        )

        if next_normalized in metadata:
            return ""

        if next_normalized.startswith("patientid"):
            return ""

        if next_normalized.startswith("patient id"):
            return ""

        # Avoid obvious metadata
        if (
            "@" in next_line
            or "http" in next_line.lower()
            or re.search(r"\d{5,}", next_line)
        ):
            return ""

        return next_line

    return ""


# ============================================================
# STATUS
# ============================================================

def get_status(value, low, high):

    if value is None:
        return "Unknown"

    if value < low:
        return "Low"

    if value > high:
        return "High"

    return "Normal"


# ============================================================
# TEST CREATION
# ============================================================

def make_test(
    test_name,
    value,
    unit="",
    reference_range=""
):

    if value is None:
        return None

    if test_name in REFERENCE_RANGES:

        low, high = REFERENCE_RANGES[test_name]

        status = get_status(
            value,
            low,
            high
        )

    else:

        status = "Unknown"

    return {
        "test_name": test_name,
        "value": value,
        "unit": unit,
        "reference_range": reference_range,
        "status": status
    }


# ============================================================
# FIND TEST VALUE
# ============================================================

def find_test_value(
    lines,
    test_name,
    reference_range
):
    """
    Searches line-by-line for:

        Test Name
        Result
        Reference Range

    Example:

        Haemoglobin
        15
        13-17
    """

    target = test_name.lower().strip()

    for i, line in enumerate(lines):

        current = line.lower().strip()

        if current != target:
            continue

        # Search the next few lines
        for j in range(i + 1, min(i + 6, len(lines))):

            candidate = lines[j].strip()

            # Stop if another known test begins
            if candidate.lower() in {
                name.lower()
                for name in REFERENCE_RANGES
            }:
                if candidate.lower() != target:
                    break

            # If this is the expected reference range,
            # the result should be immediately before it.
            if candidate == reference_range:

                # Search backwards for the nearest number
                for k in range(j - 1, i, -1):

                    value = extract_number(lines[k])

                    if value is not None:

                        return value

                return None

    return None


# ============================================================
# RBC SPECIAL CASE
# ============================================================

def find_rbc_count(lines):

    for i, line in enumerate(lines):

        normalized = (
            line.lower()
            .replace(" ", "")
        )

        if normalized not in {
            "rbccount",
            "rbc count"
        }:
            continue

        # Normal format:
        #
        # RBC Count
        # 5
        #
        # or
        #
        # RBC Count
        # 4.5-5.5
        # Mil-
        # 5
        # lion/cumm

        for j in range(i + 1, min(i + 7, len(lines))):

            current = lines[j]

            # If this is the reference range,
            # continue searching for actual result.
            if extract_range(current):
                continue

            value = extract_number(current)

            if value is None:
                continue

            # Avoid accidentally taking the reference
            # range lower/upper number.
            if value in (4.5, 5.5):
                continue

            # RBC is normally around 4-6.
            if 3.0 <= value <= 8.0:

                return value

    return None


# ============================================================
# TEST EXTRACTION
# ============================================================

def extract_tests(text: str):

    lines = get_clean_lines(text)

    print("\n========== LAB OCR LINES ==========")

    for index, line in enumerate(lines):

        print(index, ":", line)

    print("====================================\n")

    tests = []

    # ========================================================
    # TEST DEFINITIONS
    # ========================================================

    test_definitions = [

        (
            "Haemoglobin",
            "13-17",
            "g/dL"
        ),

        (
            "Total Leucocyte Count",
            "4000-10000",
            "/cumm"
        ),

        (
            "Neutrophils",
            "40-80",
            "%"
        ),

        (
            "Lymphocytes",
            "20-40",
            "%"
        ),

        (
            "Eosinophils",
            "1-6",
            "%"
        ),

        (
            "Monocytes",
            "2-10",
            "%"
        ),

        (
            "Basophils",
            "0-1",
            "%"
        ),

        (
            "Absolute Neutrophils",
            "2000-7000",
            "/cumm"
        ),

        (
            "Absolute Lymphocytes",
            "1000-3000",
            "/cumm"
        ),

        (
            "Absolute Eosinophils",
            "20-500",
            "/cumm"
        ),

        (
            "Absolute Monocytes",
            "200-1000",
            "/cumm"
        ),

        (
            "MCV",
            "81-101",
            "fL"
        ),

        (
            "MCH",
            "27-32",
            "pg"
        ),

        (
            "MCHC",
            "31.5-34.5",
            "g/dL"
        ),

        (
            "Hct",
            "40-50",
            "%"
        ),

        (
            "RDW-CV",
            "11.6-14.0",
            "%"
        ),

        (
            "RDW-SD",
            "39-46",
            "fL"
        ),

        (
            "Platelet Count",
            "150000-410000",
            "/cumm"
        ),

        (
            "MPV",
            "7.5-11.5",
            "fL"
        ),
    ]

    # ========================================================
    # NORMAL TESTS
    # ========================================================

    for test_name, reference_range, unit in test_definitions:

        value = find_test_value(
            lines,
            test_name,
            reference_range
        )

        if value is None:
            continue

        test = make_test(
            test_name,
            value,
            unit,
            reference_range
        )

        if test:
            tests.append(test)

    # ========================================================
    # RBC COUNT
    # ========================================================

    rbc_value = find_rbc_count(lines)

    if rbc_value is not None:

        tests.append(
            make_test(
                "RBC Count",
                rbc_value,
                "million/cumm",
                "4.5-5.5"
            )
        )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_tests = []

    seen = set()

    for test in tests:

        name = test["test_name"]

        if name in seen:
            continue

        seen.add(name)

        unique_tests.append(test)

    # ========================================================
    # SORT IN REPORT ORDER
    # ========================================================

    order = [
        "Haemoglobin",
        "Total Leucocyte Count",
        "Neutrophils",
        "Lymphocytes",
        "Eosinophils",
        "Monocytes",
        "Basophils",
        "Absolute Neutrophils",
        "Absolute Lymphocytes",
        "Absolute Eosinophils",
        "Absolute Monocytes",
        "RBC Count",
        "MCV",
        "MCH",
        "MCHC",
        "Hct",
        "RDW-CV",
        "RDW-SD",
        "Platelet Count",
        "MPV",
    ]

    unique_tests.sort(
        key=lambda x: order.index(x["test_name"])
        if x["test_name"] in order
        else 999
    )

    # ========================================================
    # DEBUG
    # ========================================================

    print("\n========== TEST EXTRACTION ==========")

    print(
        "Tests detected:",
        len(unique_tests)
    )

    for test in unique_tests:

        print(
            f'{test["test_name"]} '
            f'=> {test["value"]} '
            f'=> {test["status"]}'
        )

    print("=====================================\n")

    return unique_tests


# ============================================================
# HEALTH RISKS
# ============================================================

def generate_health_risks(tests):

    risks = []

    for test in tests:

        name = test["test_name"]
        status = test["status"]

        if status == "Normal":
            continue

        if name == "Haemoglobin":

            if status == "Low":
                risks.append(
                    "Haemoglobin is below the provided reference range "
                    "and may warrant evaluation for anaemia."
                )

            elif status == "High":
                risks.append(
                    "Haemoglobin is above the provided reference range."
                )

        elif name == "MCV":

            if status == "Low":
                risks.append(
                    "MCV is below the provided reference range."
                )

            elif status == "High":
                risks.append(
                    "MCV is above the provided reference range."
                )

        elif name == "MCHC":

            if status == "Low":
                risks.append(
                    "MCHC is below the provided reference range."
                )

            elif status == "High":
                risks.append(
                    "MCHC is above the provided reference range."
                )

        elif name == "Platelet Count":

            if status == "Low":
                risks.append(
                    "Platelet count is below the provided reference range."
                )

            elif status == "High":
                risks.append(
                    "Platelet count is above the provided reference range."
                )

        elif name == "Total Leucocyte Count":

            if status == "Low":
                risks.append(
                    "White blood cell count is below the provided reference range."
                )

            elif status == "High":
                risks.append(
                    "White blood cell count is above the provided reference range."
                )

        elif name == "Neutrophils":

            if status == "Low":
                risks.append(
                    "Neutrophil percentage is below the provided reference range."
                )

            elif status == "High":
                risks.append(
                    "Neutrophil percentage is above the provided reference range."
                )

        elif name == "Lymphocytes":

            if status == "Low":
                risks.append(
                    "Lymphocyte percentage is below the provided reference range."
                )

            elif status == "High":
                risks.append(
                    "Lymphocyte percentage is above the provided reference range."
                )

        elif name == "Hct":

            if status == "Low":
                risks.append(
                    "Hematocrit is below the provided reference range."
                )

            elif status == "High":
                risks.append(
                    "Hematocrit is above the provided reference range."
                )

    return risks


# ============================================================
# SUMMARY
# ============================================================

def generate_summary(tests):

    if not tests:

        return (
            "No laboratory test results could be identified "
            "from the report."
        )

    normal_count = sum(
        1
        for test in tests
        if test["status"] == "Normal"
    )

    abnormal_count = sum(
        1
        for test in tests
        if test["status"] in ["High", "Low"]
    )

    total = len(tests)

    if abnormal_count == 0:

        return (
            f"The report contains {total} identified test results. "
            "All identified values are within the provided "
            "reference ranges."
        )

    return (
        f"The report contains {total} identified test results. "
        f"{normal_count} values are within the provided reference "
        f"ranges and {abnormal_count} values are outside the "
        "provided ranges. The abnormal values should be reviewed "
        "with a healthcare professional."
    )


# ============================================================
# ADVICE
# ============================================================

def generate_advice(tests, risks):

    abnormal_tests = [
        test
        for test in tests
        if test["status"] in ["High", "Low"]
    ]

    if not abnormal_tests:

        return [
            "Continue maintaining a balanced diet and healthy lifestyle.",
            "Stay adequately hydrated and maintain regular physical activity.",
            "Continue routine health check-ups as recommended by your doctor."
        ]

    advice = []

    for test in abnormal_tests:

        name = test["test_name"]
        status = test["status"]

        if name == "Haemoglobin" and status == "Low":

            advice.append(
                "Discuss the low haemoglobin result with a healthcare professional."
            )

        elif name == "MCV" and status == "Low":

            advice.append(
                "Discuss the low MCV result with a healthcare professional."
            )

        elif name == "MCHC" and status == "High":

            advice.append(
                "Discuss the elevated MCHC result with a healthcare professional."
            )

        elif name == "Platelet Count":

            advice.append(
                "Discuss the platelet count with a healthcare professional."
            )

        elif name == "Total Leucocyte Count":

            advice.append(
                "Discuss the white blood cell count with a healthcare professional."
            )

        elif name == "Neutrophils":

            advice.append(
                "Discuss the neutrophil result with a healthcare professional."
            )

        elif name == "Lymphocytes":

            advice.append(
                "Discuss the lymphocyte result with a healthcare professional."
            )

    advice.append(
        "Do not interpret an isolated laboratory value as a diagnosis."
    )

    advice.append(
        "Consider the reference range and clinical symptoms when interpreting results."
    )

    advice.append(
        "Consult a qualified healthcare professional for medical interpretation."
    )

    unique_advice = []

    for item in advice:

        if item not in unique_advice:
            unique_advice.append(item)

    return unique_advice[:6]


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_lab_report(extracted_text: str):

    try:

        print("\n========================================")
        print("       LAB REPORT ANALYSIS START")
        print("========================================")

        # ----------------------------------------------------
        # Validate OCR
        # ----------------------------------------------------

        if not extracted_text or not extracted_text.strip():

            return {
                "patient_name": "",
                "tests": [],
                "health_risks": [],
                "summary": (
                    "No text could be extracted from "
                    "the laboratory report."
                ),
                "advice": [
                    "Please upload a clear laboratory report."
                ]
            }

        # ----------------------------------------------------
        # Patient
        # ----------------------------------------------------

        patient_name = extract_patient_name(
            extracted_text
        )

        # ----------------------------------------------------
        # Extract tests
        # ----------------------------------------------------

        tests = extract_tests(
            extracted_text
        )

        print(
            "TOTAL TESTS:",
            len(tests)
        )

        # ----------------------------------------------------
        # Risks
        # ----------------------------------------------------

        health_risks = generate_health_risks(
            tests
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary = generate_summary(
            tests
        )

        # ----------------------------------------------------
        # Advice
        # ----------------------------------------------------

        advice = generate_advice(
            tests,
            health_risks
        )

        print(
            "PATIENT:",
            patient_name or "Unknown"
        )

        print(
            "RISKS:",
            len(health_risks)
        )

        print("========================================")
        print("       LAB REPORT ANALYSIS END")
        print("========================================\n")

        return {
            "patient_name": patient_name,
            "tests": tests,
            "health_risks": health_risks,
            "summary": summary,
            "advice": advice
        }

    except Exception as e:

        print("\n========================================")
        print("       LAB ANALYSIS ERROR")
        print("========================================")

        print(
            "ERROR:",
            str(e)
        )

        traceback.print_exc()

        print("========================================\n")

        return {
            "patient_name": "",
            "tests": [],
            "health_risks": [
                "The laboratory report could not be fully analyzed."
            ],
            "summary": (
                "The OCR text was received, but the laboratory "
                "results could not be structured correctly."
            ),
            "advice": [
                "Please upload a clearer laboratory report.",
                "If the problem continues, consult a healthcare professional."
            ]
        }
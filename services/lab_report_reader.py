import pytesseract
from PIL import Image
from groq import Groq
import json
# -----------------------------
# TESSERACT
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# -----------------------------
# GROQ
# -----------------------------
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
# -----------------------------
# OCR
# -----------------------------
def extract_lab_text(file_path):

    image = Image.open(file_path)

    text = pytesseract.image_to_string(image)

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
        return ""

    return text[start:end]


# -----------------------------
# AI ANALYSIS
# -----------------------------
def analyze_lab_report(extracted_text):

    try:

        prompt = f"""
        Analyze this lab report carefully.

        Return ONLY VALID JSON.

        FORMAT:

        {{
          "patient_name": "",
          "tests": [
            {{
              "test_name": "",
              "value": "",
              "status": "Normal"
            }}
          ],
          "health_risks": [
            ""
          ],
          "summary": "",
          "advice": [
            ""
          ]
        }}

        RULES:
        - No markdown
        - No explanation
        - No extra text
        - Always include summary
        - Always include advice
        - Keep JSON short and valid

        Lab Report:
        {extracted_text}
        """

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI medical lab report analyzer."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=900
        )

        response_text = completion.choices[0].message.content

        print("RAW AI RESPONSE:\n", response_text)

        # -----------------------------
        # CLEAN RESPONSE
        # -----------------------------
        cleaned = clean_json(response_text)

        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)

        parsed = json.loads(cleaned)

        return parsed

    except Exception as e:

        print("LAB REPORT AI ERROR:", e)

        return {
            "patient_name": "Unknown",
            "tests": [],
            "health_risks": [],
            "summary": "Unable to analyze report properly.",
            "advice": [
                "Please upload a clearer lab report image."
            ]
        }
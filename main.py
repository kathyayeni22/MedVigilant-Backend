from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import os
import shutil
import asyncio
import time
import traceback

from dotenv import load_dotenv
# Database
from database import init_db
from database1 import engine
from models import *

# Services
# -----------------------------
# SERVICES
# -----------------------------

from services.medicine_reminder import generate_reminders
from services.medicine_service import validate_medicine_data

from services.ai_chatbot import (
    get_ai_response,
    get_chat_history,
    clear_chat_history
)

from services.prescription import analyze_prescription
from services.lab_report_reader import analyze_lab_report
from services.doctor_recommender import recommend_doctor


# -----------------------------
# OCR
# -----------------------------

from ocr.ocr_utils import extract_text


# -----------------------------
# ML PREDICTORS
# -----------------------------

from predictors.predict_symptoms import predict_diseases

from predictors.train_food_model import recommend_food

from predictors.train_bmi_model import predict_bmi

from predictors.train_lab_test_model import (
    predict_lab,
    recommend_lab_tests_from_symptoms
)

from predictors.train_mental_health_model import (
    predict_mental_health
)


Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Healthcare Platform")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# -----------------------------
# REQUEST MODELS
# -----------------------------

class FullHealthRequest(BaseModel):
    symptoms: str
    height_cm: float
    weight_kg: float
    age: float
    glucose: float
    mood: str


class LabResultRequest(BaseModel):
    test_name: str
    value: float


class LabSuggestionRequest(BaseModel):
    symptoms: str


class ReminderMedicine(BaseModel):
    name: str
    times_per_day: int
    days: int


class ReminderRequest(BaseModel):
    medicines: List[ReminderMedicine]


class ChatRequest(BaseModel):
    message: str
    user_id: str = "user_123"


# -----------------------------
# ROOT
# -----------------------------

@app.get("/")
def home():
    return {"message": "AI Healthcare API running 🚀"}
@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "MedVigilant backend is running"
    }

# -----------------------------
# INDIVIDUAL LAB TEST ANALYSIS
# -----------------------------

@app.post("/lab-tests")
def lab_test_result(data: LabResultRequest):

    try:
        result = predict_lab(
            data.test_name,
            data.value
        )

        return {
            "status": "success",
            "test_name": data.test_name,
            "value": data.value,
            "result": result
        }

    except Exception as e:

        print("\n========== INDIVIDUAL LAB TEST ERROR ==========")
        print("ERROR:", str(e))
        traceback.print_exc()
        print("===============================================\n")

        return {
            "status": "error",
            "test_name": data.test_name,
            "value": data.value,
            "message": str(e)
        }
# -----------------------------
# LAB TEST SUGGESTION (FROM SYMPTOMS)
# -----------------------------

@app.post("/lab-tests-from-symptoms")
def lab_from_symptoms(data: LabSuggestionRequest):
    try:
        tests = recommend_lab_tests_from_symptoms(data.symptoms)

        return {
            "status": "success",
            "symptoms": data.symptoms,
            "recommended_tests": tests
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# -----------------------------
# FULL HEALTH ANALYSIS
# -----------------------------

@app.post("/full-health-analysis")
def full_health(data: FullHealthRequest):

    try:
        # -------------------------
        # 1. Disease Prediction
        # -------------------------
        disease_result = predict_diseases(data.symptoms)

        if isinstance(disease_result, list) and len(disease_result) > 0:
            if isinstance(disease_result[0], dict):
                disease = disease_result[0].get("disease", "Unknown")
                confidence = disease_result[0].get("confidence", 0)
            else:
                disease = disease_result[0]
                confidence = 0
        else:
            disease = "Unknown"
            confidence = 0

        # -------------------------
        # 2. BMI
        # -------------------------
        height_m = data.height_cm / 100
        bmi = round(
        data.weight_kg / (height_m * height_m),
        1
        )

        # BMI CATEGORY
        if bmi < 18.5:
            bmi_status = "Underweight"
        elif bmi < 25:
            bmi_status = "Normal"
        elif bmi < 30:
            bmi_status = "Overweight"

        else:
            bmi_status = "Obese"
        # -------------------------
        # 3. HEALTH SCORE
        # -------------------------

        health_score = 100

        # BMI EFFECT
        if bmi_status == "Underweight":
            health_score -= 10

        elif bmi_status == "Overweight":
            health_score -= 15

        elif bmi_status == "Obese":
            health_score -= 25

        # GLUCOSE EFFECT
        if data.glucose > 140:
            health_score -= 20

        elif data.glucose > 110:
            health_score -= 10

        # SYMPTOM RISK
        danger_keywords = [
            "chest pain",
            "breathing problem",
            "high fever",
            "vomiting",
            "dizziness",
            "diabetes",
            "heart pain"
        ]

        for word in danger_keywords:

            if word in data.symptoms.lower():

                health_score -= 15
                break

        # MENTAL HEALTH EFFECT
        if data.mood.lower() in [
            "stress",
            "anxiety",
            "depression",
            "panic"
        ]:
            health_score -= 10

        # SAFETY
        if health_score < 0:
            health_score = 0
        # -------------------------
        # 4. MENTAL HEALTH SCORE
        # -------------------------

        mental_score = 100

        if data.mood.lower() in [
            "stress",
            "tired"
        ]:
            mental_score -= 20

        elif data.mood.lower() in [
            "anxiety",
            "panic"
        ]:
            mental_score -= 35

        elif data.mood.lower() in [
            "depression",
            "hopeless"
        ]:
            mental_score -= 50

        if mental_score >= 80:
            mental_status = "Stable"

        elif mental_score >= 60:
            mental_status = "Mild Risk"

        elif mental_score >= 40:
            mental_status = "Moderate Risk"

        else:
            mental_status = "High Risk"
        # 3. Food Recommendation
        # -------------------------
        try:
            food_ai = get_ai_response(
                f"""
                Suggest exactly 5 simple food recommendations for:
                Disease: {disease}
                Age: {data.age}
                BMI: {bmi}
                Glucose: {data.glucose}
                Rules:
                - Give ONLY 5 points
                - Each point should be short (1 line)
                - No introduction
                - No explanation
                - No symbols like *, **
                Example:
                Eat bananas
                Drink coconut water
                Avoid oily food
                Eat light meals
                Stay hydrated
                """
            )
            # Convert AI text → list (optional clean UI)
            diet_plan = [line.strip() for line in food_ai.split("\n") if line.strip()]
        except:
            diet_plan = [
                "Drink plenty of fluids",
                "Eat light foods like rice and fruits",
                "Avoid oily and fried food"
                ]
        # -------------------------
        # 4. Lab Test Suggestions
        # -------------------------
        try:
            lab_tests = recommend_lab_tests_from_symptoms(data.symptoms)
        except:
            lab_tests = []

        # -------------------------
        # 5. Mental Health (FIXED)
        # -------------------------
        try:
            mental_raw = predict_mental_health(data.mood)
            if isinstance(mental_raw, dict):
                risk = mental_raw.get("risk_level", "Low")
            else:
                risk = str(mental_raw)
            # 🔥 AI suggestion
            try:
                ai_suggestion = get_ai_response(
                    f"""
                    User says: {data.mood}
                    Risk level: {risk}
                    Give 4 short bullet point suggestions for stress relief.
                    """
                )
            except:
                ai_suggestion = ""
            # ✅ FALLBACK (IMPORTANT)
            if not ai_suggestion or len(ai_suggestion.strip()) < 10:
                if risk.lower() == "high":
                    ai_suggestion = "Take deep breaths\nTalk to someone you trust\nPractice meditation\nGet proper sleep"
                elif risk.lower() == "moderate":
                    ai_suggestion = "Take breaks\nListen to music\nRelax your mind\nMaintain routine"
                else:
                    ai_suggestion = "You are doing well\nMaintain healthy lifestyle"

            mental = {
                "status": "success",
                "input": data.mood,
                "risk_level": risk,
                "suggestion": ai_suggestion
            }

        except Exception as e:
            mental = {
                "status": "error",
                "risk_level": "Unknown",
                "suggestion": "Unable to analyze mental health"
            }

        # -------------------------
        # ALERTS
        # -------------------------

        alerts = []

        if bmi_status == "Overweight":
            alerts.append("⚠ Overweight risk detected")

        elif bmi_status == "Obese":
            alerts.append("🚨 Obesity risk detected")

        if data.glucose > 140:
            alerts.append("⚠ High glucose level")

        if mental_status == "High Risk":
            alerts.append("⚠ High stress detected")

        if health_score < 50:
            alerts.append("🚨 Overall health score is low")

        return {
            "status": "success",

            "disease": disease,

            "confidence_score": confidence,

            "health_score": health_score,

            "mental_score": mental_score,

            "bmi": bmi,

            "bmi_status": bmi_status,

            "diet_plan": diet_plan,

            "lab_tests": lab_tests,

            "mental_health": {
                "status": mental_status,
                "score": mental_score,
                "suggestion": mental.get("suggestion", "")
            },

            "alerts": alerts
        }
    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }
# -----------------------------
# MEDICINE REMINDER
# -----------------------------

@app.post("/generate-reminders")
def create_reminders(data: ReminderRequest):

    response = []

    for med in data.medicines:

        validation = validate_medicine_data(
            med.name,
            med.times_per_day,
            med.days
        )

        if validation != "Valid":
            return {"error": f"{med.name}: {validation}"}

        result = generate_reminders(
            med.name,
            med.times_per_day,
            med.days
        )

        response.append(result)

    return response


# -----------------------------
# AI CHAT
# -----------------------------


@app.post("/ai-chat")
def ai_chat(data: ChatRequest):

    response = get_ai_response(
        data.message,
        data.user_id
    )

    return {
        #"user_message": data.message,
        "ai_response": response
    }


# -----------------------------
# CHAT HISTORY
# -----------------------------

@app.get("/chat-history")
def history(user_id: str):

    return get_chat_history(user_id)


# -----------------------------
# CLEAR CHAT
# -----------------------------

@app.delete("/clear-chat/{user_id}")
def clear_chat(user_id: str):

    return clear_chat_history(user_id)
# -----------------------------------
# PRESCRIPTION READER
# -----------------------------------
@app.post("/read-prescription")
async def read_prescription(file: UploadFile = File(...)):

    file_path = None

    try:
        print("\n========== PRESCRIPTION START ==========")

        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)

        safe_filename = os.path.basename(
            file.filename or "prescription.png"
        )

        file_path = os.path.join(
            upload_folder,
            safe_filename
        )

        print("FILE NAME:", safe_filename)

        contents = await file.read()

        print("FILE SIZE:", len(contents), "bytes")

        if not contents:
            return {
                "status": "error",
                "message": "Uploaded file is empty."
            }

        with open(file_path, "wb") as f:
            f.write(contents)

        print("FILE SAVED:", file_path)

        # -----------------------------
        # OCR
        # -----------------------------

        print("STARTING RAPIDOCR...")

        extracted_text = await asyncio.to_thread(
            extract_text,
            file_path
        )

        print("OCR FINISHED")
        print("OCR LENGTH:", len(extracted_text or ""))

        print("\n========== PRESCRIPTION OCR ==========")
        print(extracted_text)
        print("======================================\n")

        # -----------------------------
        # AI ANALYSIS
        # -----------------------------

        print("STARTING PRESCRIPTION AI ANALYSIS...")

        ai_summary = await asyncio.to_thread(
            analyze_prescription,
            extracted_text
        )

        print("PRESCRIPTION AI ANALYSIS FINISHED")

        return {
            "status": "success",
            "extracted_text": extracted_text,
            "ai_summary": ai_summary
        }

    except Exception as e:

        import traceback

        print("\n========== PRESCRIPTION ERROR ==========")
        print("ERROR:", str(e))
        traceback.print_exc()
        print("========================================\n")

        return {
            "status": "error",
            "message": str(e)
        }

    finally:

        if file_path and os.path.exists(file_path):

            try:
                os.remove(file_path)
                print("TEMP FILE DELETED:", file_path)

            except Exception as e:

                print(
                    "TEMP FILE DELETE ERROR:",
                    str(e)
                )

        print("========== PRESCRIPTION END ==========\n")
# -----------------------------
# LAB REPORT READER
# -----------------------------
# -----------------------------# LAB REPORT READER
# -----------------------------

@app.post("/read-lab-report")
async def read_lab_report(file: UploadFile = File(...)):

    file_path = None
    start_time = time.time()

    try:

        print("\n========================================")
        print("       LAB REPORT PROCESSING START")
        print("========================================")

        # ---------------------------------
        # 1. Validate file
        # ---------------------------------

        if not file.filename:
            return {
                "status": "error",
                "message": "No file was uploaded."
            }

        # ---------------------------------
        # 2. Create uploads directory
        # ---------------------------------

        upload_folder = "uploads"

        os.makedirs(
            upload_folder,
            exist_ok=True
        )

        # ---------------------------------
        # 3. Create safe filename
        # ---------------------------------

        safe_filename = os.path.basename(
            file.filename
        )

        file_path = os.path.join(
            upload_folder,
            safe_filename
        )

        print("FILE NAME:", safe_filename)

        # ---------------------------------
        # 4. Read uploaded file
        # ---------------------------------

        contents = await file.read()

        print(
            "FILE SIZE:",
            len(contents),
            "bytes"
        )

        if not contents:
            return {
                "status": "error",
                "message": "Uploaded file is empty."
            }

        # ---------------------------------
        # 5. Save file
        # ---------------------------------

        with open(file_path, "wb") as f:
            f.write(contents)

        print("FILE SAVED:", file_path)

        # ---------------------------------
        # 6. OCR
        # ---------------------------------

        print("\nSTARTING RAPIDOCR...")

        ocr_start = time.time()

        extracted_text = await asyncio.to_thread(
            extract_text,
            file_path
        )

        ocr_time = round(
            time.time() - ocr_start,
            2
        )

        extracted_text = extracted_text or ""

        print("OCR FINISHED")
        print("OCR TIME:", ocr_time, "seconds")
        print(
            "OCR TEXT LENGTH:",
            len(extracted_text)
        )

        print("\n========== OCR OUTPUT ==========")
        print(extracted_text)
        print("================================\n")

        # ---------------------------------
        # 7. Check OCR result
        # ---------------------------------

        if len(extracted_text.strip()) < 10:

            return {
                "status": "error",
                "extracted_text": extracted_text,
                "ai_summary": {
                    "patient_name": "",
                    "tests": [],
                    "health_risks": [
                        "Unable to extract sufficient text from the report."
                    ],
                    "summary": (
                        "The laboratory report could not be "
                        "read properly by OCR."
                    ),
                    "advice": [
                        "Please upload a clearer laboratory report."
                    ]
                }
            }

        # ---------------------------------
        # 8. Analyze OCR text
        # ---------------------------------

        print("\nSTARTING LAB ANALYSIS...")

        analysis_start = time.time()

        # IMPORTANT:
        # analyze_lab_report is synchronous,
        # so run it in a separate thread.

        lab_analysis = await asyncio.to_thread(
            analyze_lab_report,
            extracted_text
        )

        analysis_time = round(
            time.time() - analysis_start,
            2
        )

        print("LAB ANALYSIS FINISHED")
        print(
            "LAB ANALYSIS TIME:",
            analysis_time,
            "seconds"
        )

        # ---------------------------------
        # 9. Make sure analysis is valid
        # ---------------------------------

        if not isinstance(lab_analysis, dict):

            lab_analysis = {
                "patient_name": "",
                "tests": [],
                "health_risks": [],
                "summary": str(lab_analysis),
                "advice": []
            }

        # ---------------------------------
        # 10. Total processing time
        # ---------------------------------

        total_time = round(
            time.time() - start_time,
            2
        )

        print(
            "\nTOTAL PROCESSING TIME:",
            total_time,
            "seconds"
        )

        print("========================================")
        print("       LAB REPORT PROCESSING END")
        print("========================================\n")

        # ---------------------------------
        # 11. Final response
        # ---------------------------------

        return {
            "status": "success",
            "extracted_text": extracted_text,
            "ai_summary": lab_analysis,
            "processing_time": total_time
        }

    except Exception as e:

        print("\n========================================")
        print("        LAB REPORT PROCESSING ERROR")
        print("========================================")

        print("ERROR:", str(e))

        traceback.print_exc()

        print("========================================\n")

        return {
            "status": "error",
            "message": str(e)
        }

    finally:

        # ---------------------------------
        # Delete temporary file
        # ---------------------------------

        if file_path and os.path.exists(file_path):

            try:

                os.remove(file_path)

                print(
                    "TEMP FILE DELETED:",
                    file_path
                )

            except Exception as e:

                print(
                    "TEMP FILE DELETE ERROR:",
                    str(e)
                )

# -----------------------------
# AI DOCTOR RECOMMENDATION
# -----------------------------
@app.post("/doctor-recommendation")
async def doctor_recommendation(data: dict):

    symptoms = data.get("symptoms", "")

    result = recommend_doctor(symptoms)

    return {
        "status": "success",
        "recommendation": result
    }
from predictors.predict_symptoms import predict_diseases
from predictors.train_bmi_model import predict_bmi
from predictors.train_food_model import recommend_food
from predictors.train_mental_health_model import predict_mental_health
from predictors.train_lab_test_model import predict_lab
from services.medicine_reminder import generate_reminders
from services.ai_chatbot import get_ai_response


def orchestrate_health(user):

    diseases = predict_diseases(user["symptoms"])
    primary_disease = diseases[0]["disease"]

    bmi_data = predict_bmi(user["height_cm"], user["weight_kg"])

    food = recommend_food(
        primary_disease,
        user["age"],
        bmi_data["bmi"],
        user["glucose"]
    )

    mental = predict_mental_health(user["mental_text"])

    lab = predict_lab(
        user["lab_test"]["name"],
        user["lab_test"]["value"]
    )

    reminders = []
    for med in user["medicines"]:
        reminder = generate_reminders(
            med["name"],
            med["times_per_day"],
            med["days"]
        )
        reminders.append(reminder)

    # AI explanation
    summary = f"""
    Disease Risk: {primary_disease}
    BMI: {bmi_data['bmi']} ({bmi_data['category']})
    Mental Health: {mental}
    Lab Test: {lab}
    Food Advice: {food}
    """

    ai_advice = get_ai_response(summary)

    return {
        "disease_prediction": diseases,
        "bmi": bmi_data,
        "food": food,
        "mental_health": mental,
        "lab_result": lab,
        "medicine_reminders": reminders,
        "ai_health_advice": ai_advice
    }

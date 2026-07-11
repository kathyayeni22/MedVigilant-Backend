from predictors.predict_symptoms import predict_diseases
from predictors.train_bmi_model import predict_bmi
from predictors.train_food_model import recommend_food
from predictors.train_mental_health_model import predict_mental_health
from predictors.train_lab_test_model import predict_lab


def test_symptom_model():
    print("\n🔍 Testing Symptom Model")
    result = predict_diseases("fever cough headache")
    print("Output:", result)

    assert isinstance(result, list), "❌ Should return list"
    assert "disease" in result[0], "❌ Missing disease key"
    assert "confidence" in result[0], "❌ Missing confidence key"
    print("✅ Symptom model working")


def test_bmi_model():
    print("\n🔍 Testing BMI Model")
    result = predict_bmi(170, 70)
    print("Output:", result)

    assert "bmi" in result, "❌ Missing BMI value"
    print("✅ BMI model working")


def test_food_model():
    print("\n🔍 Testing Food Recommendation Model")
    result = recommend_food("Flu", 25, 22.5, 110)
    print("Output:", result)

    assert result is not None, "❌ Food model failed"
    print("✅ Food model working")


def test_mental_health_model():
    print("\n🔍 Testing Mental Health Model")
    result = predict_mental_health("I feel anxious and stressed")
    print("Output:", result)

    assert result is not None, "❌ Mental health model failed"
    print("✅ Mental health model working")


def test_lab_model():
    print("\n🔍 Testing Lab Test Model")
    result = predict_lab("hemoglobin", 11)
    print("Output:", result)

    assert result is not None, "❌ Lab model failed"
    print("✅ Lab test model working")


if __name__ == "__main__":
    print("🚀 Running ML System Check\n")

    test_symptom_model()
    test_bmi_model()
    test_food_model()
    test_mental_health_model()
    test_lab_model()

    print("\n🎉 All ML models executed successfully")

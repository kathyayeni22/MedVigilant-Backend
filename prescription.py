from services.medicine_service import validate_medicine_data
from services.medicine_reminder import generate_reminders

print("🚀 Testing Medicine System\n")

test_input = {
    "medicine_name": "Paracetamol",
    "dosage": "500mg",
    "times_per_day": 2,
    "meal": "after",
    "duration_days": 3
}

# Step 1: Validate
valid, message = validate_medicine_data(test_input)
print("Validation:", message)

if valid:
    result = generate_reminders(test_input)
    print("Reminder Output:")
    print(result)
else:
    print("Validation Failed")

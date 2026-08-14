from ocr.ocr_utils import extract_text
from prescription import analyze_prescription

print("🚀 Testing Medicine System")

image_path = "sample_prescription.jpeg"

# Step 1: OCR
extracted_text = extract_text(image_path)

print("\n========== OCR OUTPUT ==========\n")
print(extracted_text)

# Step 2: AI analysis
print("\n🚀 Sending OCR text to AI...\n")

result = analyze_prescription(extracted_text)

print("\n========== AI RESULT ==========\n")

print(result)
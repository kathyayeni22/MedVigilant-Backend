import pytesseract
from PIL import Image
from ocr.ocr_utils import preprocess_image, clean_ocr_text

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def extract_text_from_prescription(image_path: str) -> str:
    image = Image.open(image_path)
    image = preprocess_image(image)

    raw_text = pytesseract.image_to_string(image, config="--psm 6")
    cleaned_text = clean_ocr_text(raw_text)

    return cleaned_text

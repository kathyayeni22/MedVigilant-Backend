import re
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_image(image: Image.Image) -> Image.Image:
    image = image.convert("L")  # grayscale
    image = image.filter(ImageFilter.MedianFilter())
    image = image.filter(ImageFilter.SHARPEN)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.5)

    return image


def clean_ocr_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

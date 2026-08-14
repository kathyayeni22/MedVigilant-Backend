from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()


def extract_text(image_path):

    result, _ = engine(image_path)

    if result is None:
        return ""

    text = []

    for line in result:
        text.append(line[1])

    return "\n".join(text)
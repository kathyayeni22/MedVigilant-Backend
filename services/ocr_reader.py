import easyocr

reader = None

def get_reader():
    global reader

    if reader is None:
        print("Loading EasyOCR model...")
        reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

    return reader
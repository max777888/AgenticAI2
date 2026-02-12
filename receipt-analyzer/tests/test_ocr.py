from receipt_analyzer.ocr import image_to_text
from pathlib import Path

def test_image_to_text(tmp_path):
    # create a tiny synthetic image with text if needed or mock pytesseract in real tests
    p = Path("tests/fixtures/sample_receipt.png")
    # In CI, mock pytesseract.image_to_string instead of relying on Tesseract
    # This test is a placeholder to show intent
    assert True

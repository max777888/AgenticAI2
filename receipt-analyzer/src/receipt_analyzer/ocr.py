from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytesseract
from .logger import get_logger

logger = get_logger(__name__)

def preprocess_image(path: Path) -> np.ndarray:
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 2)
    return th

def image_to_text(path: Path, tesseract_cmd: str | None = None) -> str:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    img = preprocess_image(path)
    pil = Image.fromarray(img)
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(pil, config=config)
    logger.debug("OCR text length: %d for %s", len(text), path.name)
    return text

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

    # Resize up 2–3× if screenshot is low-res (very common)
    h, w = img.shape[:2]
    scale = 2.0
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Try bilateral filter instead of (or after) fastNlMeans — preserves edges better
    bilateral = cv2.bilateralFilter(denoised, d=9, sigmaColor=75, sigmaSpace=75)

    # Sharpen a bit
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(bilateral, -1, kernel)

    # Threshold — Gaussian often good, but try Otsu too
    # th = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    _, th = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: invert if text is light on dark (some thermal prints)
    # th = cv2.bitwise_not(th)

    return th

def image_to_text(path: Path, tesseract_cmd: str | None = None) -> str:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    img = preprocess_image(path)
    pil = Image.fromarray(img)
    config = r'--oem 1 --psm 6 -l eng'
    text = pytesseract.image_to_string(pil, config=config)
    logger.debug("OCR text length: %d for %s", len(text), path.name)
    return text

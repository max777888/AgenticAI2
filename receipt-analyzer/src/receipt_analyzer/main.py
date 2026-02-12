from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from .config import load_config
from .logger import get_logger
from .ocr import image_to_text
from .parser import parse_receipt_text
from .exporter import export_to_excel

logger = get_logger(__name__)

def process_file(path: Path, cfg) -> Dict[str, Any]:
    logger.info("Processing %s", path.name)
    try:
        text = image_to_text(path, tesseract_cmd=cfg.tesseract_cmd)
        parsed = parse_receipt_text(text, model=cfg.ollama_model, ollama_cmd=cfg.ollama_cmd, timeout=cfg.ollama_timeout)
        return {
            "file_name": path.name,
            "vendor": parsed.vendor,
            "date": parsed.date,
            "total_amount": parsed.total_amount,
            "tax": parsed.tax,
            "confidence": parsed.confidence,
            "raw_text": text[:1000]
        }
    except Exception as exc:
        logger.exception("Failed to process %s: %s", path.name, exc)
        return {
            "file_name": path.name,
            "vendor": None,
            "date": None,
            "total_amount": None,
            "tax": None,
            "confidence": None,
            "raw_text": ""
        }

def run() -> None:
    cfg = load_config()
    # reconfigure logger with configured level
    global logger
    logger = get_logger("receipt_analyzer", level=cfg.log_level)
    receipts_dir = Path(cfg.receipts_dir)
    if not receipts_dir.exists():
        logger.error("Receipts directory does not exist: %s", receipts_dir)
        raise SystemExit(1)
    image_files = [p for p in receipts_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff"}]
    logger.info("Found %d images in %s", len(image_files), receipts_dir)
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = {ex.submit(process_file, p, cfg): p for p in image_files}
        for fut in as_completed(futures):
            results.append(fut.result())
    export_to_excel(results, Path(cfg.output_excel))

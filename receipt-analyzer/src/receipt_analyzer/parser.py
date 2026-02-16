import subprocess
import json
import ollama
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .logger import get_logger
from .models import ParsedReceipt
from .utils import normalize_amount, normalize_date

logger = get_logger(__name__)

PROMPT_TEMPLATE = """
You are a strict JSON extractor. Input is raw OCR text from a receipt. Extract exactly these keys:
vendor, date, total_amount, tax, confidence.
Return only valid JSON. If a field is not present, set it to null.
Date should be ISO 8601 (YYYY-MM-DD) if possible. Amounts should be numbers (no currency symbol).
Here is the OCR text:
\"\"\"{ocr_text}\"\"\"
"""

class OllamaError(RuntimeError):
    pass


from ollama import ResponseError  # optional, for nicer errors

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(OllamaError)
)
def call_ollama(prompt: str, model: str, ollama_cmd: str, timeout: int) -> str:
    # Note: ollama_cmd and timeout are unused now → you can remove them from args later
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.0,      # deterministic → good for strict JSON
                "num_ctx": 8192,         # adjust higher if receipts are very long
                "num_predict": 512,      # cap output to prevent runaway generation
            }
        )
        return response['response'].strip()
    except ResponseError as e:
        logger.warning(f"Ollama API error {e.status_code}: {e.message}")
        raise OllamaError(f"Ollama error: {e.message}") from e
    except Exception as e:
        logger.warning(f"Ollama failed: {str(e)}")
        raise OllamaError(f"Ollama failed: {str(e)}") from e
    
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
#        retry=retry_if_exception_type(OllamaError))
# def call_ollama(prompt: str, model: str, ollama_cmd: str, timeout: int) -> str:
#     cmd = [ollama_cmd, "run", model, prompt]
#     logger.debug("Calling Ollama: %s", " ".join(cmd))
#     try:
#         proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
#     except subprocess.TimeoutExpired as exc:
#         logger.warning("Ollama timeout")
#         raise OllamaError("Ollama timeout") from exc
#     if proc.returncode != 0:
#         logger.warning("Ollama returned non-zero exit: %s", proc.stderr.strip())
#         raise OllamaError(proc.stderr.strip() or "Ollama error")
#     return proc.stdout.strip()

def extract_json_from_output(raw: str) -> dict:
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        json_text = raw[start:end]
        return json.loads(json_text)
    except Exception as exc:
        logger.error("Failed to extract JSON: %s", exc)
        raise ValueError("Invalid JSON from Ollama") from exc

def parse_receipt_text(ocr_text: str, model: str, ollama_cmd: str, timeout: int) -> ParsedReceipt:
    prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)
    raw = call_ollama(prompt, model=model, ollama_cmd=ollama_cmd, timeout=timeout)
    data = extract_json_from_output(raw)
    total = normalize_amount(data.get("total_amount"))
    tax = normalize_amount(data.get("tax"))
    date = normalize_date(data.get("date"))
    vendor = data.get("vendor")
    confidence = data.get("confidence")
    parsed = ParsedReceipt(
        vendor=vendor,
        date=date,
        total_amount=total,
        tax=tax,
        confidence=float(confidence) if confidence is not None else None
    )
    return parsed

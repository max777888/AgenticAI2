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
You are an expert at extracting data from Canadian receipts. Output ONLY JSON.
Look carefully at numbers near words like "TOTAL", "Sales Tax", "HST", "Subtotal", "Grand Total", "Amount Due".
Tax is usually ~13% in Ontario (HST).
Do NOT guess or hallucinate numbers — if unclear, put null.
Keys: total_amount (grand total after tax), tax_amount, subtotal (before tax), vendor, date.

Return ONLY valid JSON using this structure:
{{
  "vendor": "null",
  "date": "YYYY-MM-DD",
  "total_amount": 0.0,
  "tax": 0.0,
  "confidence": 0.0
}}


OCR Text:
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
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            format="json",
            options={
                "temperature": 0.0,      # deterministic → good for strict JSON
                "num_ctx": 4096,         # adjust higher if receipts are very long
                "num_predict": 512,      # cap output to prevent runaway generation
            }
        )
        return response['response'].strip()
    except ResponseError as e:
        logger.warning(f"Ollama API error {e.status_code}: {e.error}")
        raise OllamaError(f"Ollama error: {e.error}") from e
    except Exception as e:
        logger.warning(f"Ollama failed: {str(e)}")
        raise OllamaError(f"Ollama failed: {str(e)}") from e
    


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

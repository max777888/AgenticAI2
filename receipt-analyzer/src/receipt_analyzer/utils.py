import re
from dateutil import parser as dateparser
from typing import Optional
from .logger import get_logger

logger = get_logger(__name__)

def normalize_amount(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        s = str(value).strip()
        s = re.sub(r"[^\d\.\-\,]", "", s)
        s = s.replace(",", "")
        return float(s) if s != "" else None
    except Exception:
        logger.debug("normalize_amount failed for %s", value)
        return None

def normalize_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        dt = dateparser.parse(str(value), dayfirst=False, fuzzy=True)
        return dt.date().isoformat()
    except Exception:
        logger.debug("normalize_date failed for %s", value)
        return None

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from .logger import get_logger

logger = get_logger(__name__)

def export_to_excel(rows: List[Dict[str, Any]], output: Path) -> None:
    df = pd.DataFrame(rows)
    cols = ["file_name", "vendor", "date", "total_amount", "tax", "confidence", "raw_text"]
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    df.to_excel(output, index=False, engine="openpyxl")
    logger.info("Exported %d rows to %s", len(df), output)

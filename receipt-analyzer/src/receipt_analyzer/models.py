from pydantic import BaseModel
from typing import Optional

class ParsedReceipt(BaseModel):
    vendor: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[float] = None
    tax: Optional[float] = None
    confidence: Optional[float] = None

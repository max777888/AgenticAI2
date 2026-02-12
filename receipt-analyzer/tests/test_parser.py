from unittest.mock import patch, MagicMock
from receipt_analyzer.parser import parse_receipt_text

@patch("receipt_analyzer.parser.call_ollama")
def test_parse_receipt_text(mock_call):
    mock_call.return_value = '{"vendor":"Test Store","date":"2024-12-01","total_amount":"12.34","tax":"0.99","confidence":0.95}'
    parsed = parse_receipt_text("dummy ocr", model="m", ollama_cmd="ollama", timeout=10)
    assert parsed.vendor == "Test Store"
    assert parsed.date == "2024-12-01"
    assert parsed.total_amount == 12.34
    assert parsed.tax == 0.99

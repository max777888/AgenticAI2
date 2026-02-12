import tempfile
from receipt_analyzer.exporter import export_to_excel
from pathlib import Path

def test_export_to_excel(tmp_path):
    rows = [{"file_name":"a.png","vendor":"X","date":"2024-01-01","total_amount":10.0,"tax":0.5,"raw_text":"..."}]
    out = tmp_path / "out.xlsx"
    export_to_excel(rows, out)
    assert out.exists()

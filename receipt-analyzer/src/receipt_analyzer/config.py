from pydantic import BaseSettings, Field
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    receipts_dir: Path = Field(default=Path("sample_receipts"))
    output_excel: Path = Field(default=Path("receipts_output.xlsx"))
    ollama_model: str = Field(default="llama2")
    ollama_cmd: str = Field(default="ollama")
    ollama_timeout: int = Field(default=30)
    max_workers: int = Field(default=4)
    tesseract_cmd: Optional[str] = Field(default=None)
    log_level: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def load_config(path: Path = Path("config.yaml")) -> Settings:
    import yaml
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)
    return Settings()

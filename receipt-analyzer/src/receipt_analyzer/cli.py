import argparse
from .main import run
from .logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(prog="receipt-analyzer", description="Analyze receipts and export to Excel")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    try:
        run()
    except Exception as exc:
        logger.exception("Application failed: %s", exc)
        raise SystemExit(1)

if __name__ == "__main__":
    main()

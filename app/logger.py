"""
Simple logging setup for the application.

- Creates a 'logs/' directory if it doesn't exist.
- Provides two loggers:
    * "app"       -> general application logs (logs/app.log)
    * "predict"   -> prediction-related logs (logs/predictions.log)
- Each logger uses RotatingFileHandler (max 1 MB per file, keep 3 backups).
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Directory where all log files are stored
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)  # Create logs/ if missing


def _mk_handler(path: Path) -> RotatingFileHandler:
    """Create a rotating file handler with a standard log text format."""
    handler = RotatingFileHandler(
        path,
        maxBytes=1_000_000,  # Rotate when file reaches ~1 MB
        backupCount=3,       # Keep up to 3 old log files
    )
    # Example line: "2025-11-18 10:23:45,123 INFO app: Server started"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    return handler


def get_app_logger() -> logging.Logger:
    """
    Get the main application logger.

    - Logger name: "app"
    - Output file: logs/app.log
    - Level: INFO and above
    - Adds the file handler only once, even if called multiple times.
    """
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent double logging via root logger
    if not logger.handlers:   # Only add handler if none exist
        logger.addHandler(_mk_handler(LOG_DIR / "app.log"))
    return logger


def get_pred_logger() -> logging.Logger:
    """
    Get the prediction logger.

    - Logger name: "predict"
    - Output file: logs/predictions.log
    - Level: INFO and above
    - Use for logging model inputs/outputs, latency, etc.
    """
    logger = logging.getLogger("predict")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(_mk_handler(LOG_DIR / "predictions.log"))
    return logger

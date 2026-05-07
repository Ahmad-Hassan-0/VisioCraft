"""
app/logging_config.py
Centralized logging configuration for VisioCraft.
"""
import logging
import sys
from pathlib import Path

import config


def setup_logging():
    """Configure application-wide logging."""
    log_format = (
        "%(asctime)s │ %(levelname)-7s │ %(name)-28s │ %(message)s"
    )
    date_format = "%H:%M:%S"

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root.addHandler(console)

    # File handler — everything
    log_file = config.PROJECT_ROOT / "visiocraft.log"
    file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

    logging.getLogger('visiocraft').info("=" * 60)
    logging.getLogger('visiocraft').info("Logging initialized — file: %s", log_file)
    logging.getLogger('visiocraft').info("=" * 60)

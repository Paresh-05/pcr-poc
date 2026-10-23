"""Console + rotating file logging for every "dtcbot.*" logger, so each pipeline step
(query analysis, retrieval, tree build, LLM calls, outcomes) is visible live in the
terminal and kept in logs/dtcbot.log for later review.

Deliberately attached to the "dtcbot" logger only, with propagate=False, so it never
duplicates uvicorn's own access/error log lines on stdout.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import Settings

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(settings: Settings) -> None:
    global _configured
    if _configured:
        return
    _configured = True

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(log_dir / "dtcbot.log", maxBytes=10_000_000, backupCount=5)
    file_handler.setFormatter(formatter)

    root = logging.getLogger("dtcbot")
    root.setLevel(settings.log_level.upper())
    root.propagate = False
    root.handlers = [console_handler, file_handler]

    root.info("Logging configured: level=%s file=%s", settings.log_level.upper(), log_dir / "dtcbot.log")

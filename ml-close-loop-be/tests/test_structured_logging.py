import json
import logging
import re
from io import StringIO

import structlog

from app.logging import configure_logging

# Avoid polluting test output with structured logs during import
configure_logging(log_level="WARNING", debug=False)


def _capture_logs(log_level: str = "INFO", debug: bool = False):
    """Reconfigure logging and capture structured output to a StringIO."""
    configure_logging(log_level=log_level, debug=debug)

    root = logging.getLogger()
    formatter = root.handlers[0].formatter

    stream = StringIO()
    root.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    return stream


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_configure_logging_json_in_production():
    stream = _capture_logs(log_level="INFO", debug=False)
    logger = structlog.get_logger("test_json_prod")
    logger.info("hello", key="value")

    output = stream.getvalue().strip()
    data = json.loads(output)
    assert data["level"] == "info"
    assert data["event"] == "hello"
    assert "timestamp" in data


def test_configure_logging_human_in_development():
    stream = _capture_logs(log_level="INFO", debug=True)
    logger = structlog.get_logger("test_console_dev")
    logger.info("hello", key="value")

    output = _strip_ansi(stream.getvalue())
    assert "hello" in output
    assert "key=value" in output


def test_log_entry_has_required_fields():
    stream = _capture_logs(log_level="INFO", debug=False)
    logger = structlog.get_logger("test_required_fields")
    logger.warning("something happened", extra="data")

    output = stream.getvalue().strip()
    data = json.loads(output)
    assert "timestamp" in data
    assert data["level"] == "warning"
    assert "logger" in data
    assert data["event"] == "something happened"


def test_log_level_filtering():
    stream = _capture_logs(log_level="WARNING", debug=False)
    logger = structlog.get_logger("test_filtering")
    logger.info("this should not appear")
    logger.warning("this should appear")

    output = stream.getvalue().strip()
    lines = [line for line in output.splitlines() if line.strip()]
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["event"] == "this should appear"


def test_exception_logged_with_exc_info():
    stream = _capture_logs(log_level="INFO", debug=False)
    logger = structlog.get_logger("test_exc_info")
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("error occurred", run_id="123")

    output = stream.getvalue().strip()
    data = json.loads(output)
    assert data["event"] == "error occurred"
    assert data["level"] == "error"
    assert "exc_info" in data

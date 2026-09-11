import logging
import re
import sys

import structlog

_SECRET_PATTERNS = re.compile(
    r"(password|token|jwt_secret|api_key|credential|secret_key|secret|access_key)",
    re.IGNORECASE,
)


def _scrub_secrets(logger, method_name, event_dict):
    """Redact secret fields from structured log events."""
    for key in list(event_dict.keys()):
        if _SECRET_PATTERNS.search(str(key)):
            event_dict[key] = "***REDACTED***"
    return event_dict


def configure_logging(log_level: str = "INFO", debug: bool = False) -> None:
    """Configure structlog with JSON rendering in production, console in development."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _scrub_secrets,
    ]

    if debug:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler._configure_logging_owned = True  # noqa: SLF001 -- our own marker, not stdlib API

    root_logger = logging.getLogger()
    # Only remove handlers this function added on a previous call (e.g. lifespan re-running
    # per TestClient in tests) -- clearing unconditionally also destroys externally-attached
    # handlers such as pytest's caplog handler, which broke every caplog-based test whose
    # `client` fixture re-triggers the FastAPI lifespan (see tests/test_error_logging.py).
    for existing in list(root_logger.handlers):
        if getattr(existing, "_configure_logging_owned", False):
            root_logger.removeHandler(existing)
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

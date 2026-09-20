"""OpenTelemetry setup for DEFNEX MLOps (PRD §28)."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Metric instruments — populated by setup_telemetry()
_training_runs_counter: Any = None
_deployments_counter: Any = None
_training_duration: Any = None
_inference_latency: Any = None
_active_training_runs: Any = None
_gpu_lock_wait: Any = None

# RED (Rate/Errors/Duration) HTTP metrics (issue #169): unlike the OTel-only instruments
# above, these are created here at import time - independent of `setup_telemetry()`/
# `OTEL_ENABLED` - so `/metrics` isn't a no-op just because OTel tracing is off. Still
# genuinely optional: prometheus-client is only declared under the `otel` extra
# (pyproject.toml), so this stays `None` (silent no-op, matching get_prometheus_metrics'
# existing ImportError handling below) wherever that extra isn't installed.
try:
    from prometheus_client import Counter, Histogram

    _http_requests_total: Any = Counter(
        "http_requests_total",
        "Total HTTP requests, labeled by method/route template/status code",
        ["method", "path", "status_code"],
    )
    _http_request_duration_seconds: Any = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds, labeled by method/route template",
        ["method", "path"],
    )
except ImportError:
    _http_requests_total = None
    _http_request_duration_seconds = None


def record_http_request(
    method: str, path: str, status_code: int, duration_seconds: float
) -> None:
    """Record one request's RED signals (issue #169). Called from AccessLogMiddleware for
    every request. No-op when prometheus-client isn't installed."""
    if _http_requests_total is not None:
        _http_requests_total.labels(
            method=method, path=path, status_code=str(status_code)
        ).inc()
    if _http_request_duration_seconds is not None:
        _http_request_duration_seconds.labels(method=method, path=path).observe(
            duration_seconds
        )


def _add_trace_to_structlog(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add trace_id and span_id to structlog context when OTel is active."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.trace_id:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
    except Exception:
        pass
    return event_dict


def setup_telemetry(app: Any, engine: Any = None) -> None:
    """Initialize OTel tracing and metrics.

    Called once during app startup. When ``OTEL_ENABLED`` is ``False`` (the
    default) this is a no-op — no OTel packages need to be installed.
    """
    from app.config import settings

    if not settings.otel_enabled:
        logger.info("otel_disabled")
        return

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.trace import TracerProvider

        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        # Tracer
        provider = TracerProvider()
        trace.set_tracer_provider(provider)

        # Metrics — Prometheus exporter
        from opentelemetry.exporter.prometheus import PrometheusMetricReader

        reader = PrometheusMetricReader()
        meter_provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)

        # Create instruments
        meter = metrics.get_meter("defnex-mlops")
        global _training_runs_counter, _deployments_counter
        global _training_duration, _inference_latency
        global _active_training_runs, _gpu_lock_wait

        _training_runs_counter = meter.create_counter(
            "training_runs_total", description="Total training runs"
        )
        _deployments_counter = meter.create_counter(
            "deployments_total", description="Total deployments"
        )
        _training_duration = meter.create_histogram(
            "training_duration_seconds", description="Training duration"
        )
        _inference_latency = meter.create_histogram(
            "inference_latency_seconds", description="Inference latency"
        )
        _active_training_runs = meter.create_up_down_counter(
            "active_training_runs", description="Active training runs"
        )
        _gpu_lock_wait = meter.create_histogram(
            "gpu_lock_wait_seconds", description="GPU lock wait time"
        )

        # Auto-instrument
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()

        if engine is not None:
            try:
                from opentelemetry.instrumentation.sqlalchemy import (
                    SQLAlchemyInstrumentor,
                )

                SQLAlchemyInstrumentor().instrument(engine=engine)
            except Exception:
                logger.warning("otel_sqlalchemy_instrumentation_failed")

        logger.info("otel_initialized")
    except ImportError:
        logger.warning(
            "otel_import_failed",
            hint="pip install -e '.[otel]'",
        )


def get_prometheus_metrics() -> bytes:
    """Return Prometheus-format metrics from the global reader."""
    try:
        from prometheus_client import generate_latest

        return generate_latest()
    except ImportError:
        return b""

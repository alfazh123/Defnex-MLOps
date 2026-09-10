"""Alerting service for critical events (PRD §28.1, issue #84).

Sends HTTP POST alerts to a configurable webhook endpoint when important
events occur: training failures, stale run detection, deploy failures.
Alerts carry context (job_id, version, error, trace_id) but never secrets.
"""

from __future__ import annotations

import asyncio

import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class AlertService:
    """Sends alerts to a configurable webhook endpoint (PRD §28.1)."""

    @staticmethod
    async def send(
        event: str,
        *,
        job_id: str = "",
        version: str = "",
        error: str = "",
        trace_id: str = "",
        **kwargs: object,
    ) -> None:
        if not settings.alert_webhook_url:
            return
        import httpx

        payload: dict[str, object] = {
            "event": event,
            "job_id": job_id,
            "version": version,
            "error": error,
            "trace_id": trace_id,
            **kwargs,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(settings.alert_webhook_url, json=payload)
        except Exception:
            logger.warning("alert_send_failed", event_type=event)


def _schedule_alert(coro):
    """Fire-and-forget an async coroutine from a sync caller.

    Tries the running loop first (works inside FastAPI request handlers and
    async workers). Falls back to creating a new loop + thread so tests and
    sync CLI callers don't crash — best-effort alerting must never break the
    caller.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(coro)
            else:
                loop.run_until_complete(coro)
        except RuntimeError:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(coro)
                loop.close()
            except Exception:
                logger.warning("alert_schedule_failed")


def alert_training_failed(job_id: str, error: str, trace_id: str = "") -> None:
    """Fire-and-forget alert for a training run failure."""
    _schedule_alert(
        AlertService.send(
            "TRAINING_FAILED", job_id=job_id, error=error, trace_id=trace_id
        )
    )


def alert_training_stale(job_id: str, trace_id: str = "") -> None:
    """Fire-and-forget alert for a stale (crashed worker) training run."""
    _schedule_alert(
        AlertService.send("TRAINING_STALE", job_id=job_id, trace_id=trace_id)
    )


def alert_deploy_failed(version: str, error: str, trace_id: str = "") -> None:
    """Fire-and-forget alert for a deployment failure."""
    _schedule_alert(
        AlertService.send(
            "DEPLOY_FAILED",
            version=version,
            error=error,
            trace_id=trace_id,
        )
    )

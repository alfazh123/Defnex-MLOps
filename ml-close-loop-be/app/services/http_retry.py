"""Shared HTTP retry policy for outbound API calls (issue #40).

Single source of truth for the "retry on 5xx / connection errors, never on 4xx"
policy that was previously duplicated in `unsloth_client`. Exposes an async
wrapper (`request_with_retry`) for Unsloth Studio and a sync wrapper
(`request_sync_with_retry`) for the vLLM serving backend, so both boundaries
behave identically and the policy is fixed in exactly one place.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF = [1.0, 2.0, 4.0]


def _log_failure(
    url: str,
    *,
    context: str,
    last_exc: Exception | None,
    retries_exhausted: bool,
    destination: Any,
    event: str,
) -> None:
    status_code = (
        last_exc.response.status_code
        if isinstance(last_exc, httpx.HTTPStatusError)
        else None
    )
    response_body = (
        last_exc.response.text[:500]
        if isinstance(last_exc, httpx.HTTPStatusError) and last_exc.response
        else ""
    )
    destination.error(
        event,
        endpoint=url,
        status_code=status_code,
        response_body=response_body,
        context=context,
        retries_exhausted=retries_exhausted,
    )


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    context: str = "",
    dest: Any = logger,
    error_event: str = "http_api_error",
    retry_event: str = "http_api_retry",
    parse_json: bool = True,
) -> dict[str, Any] | httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.request(
                method, url, json=json, params=params, headers=headers
            )
            resp.raise_for_status()
            if parse_json:
                return resp.json()
            return resp
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:
                _log_failure(
                    url,
                    context=context,
                    last_exc=exc,
                    retries_exhausted=False,
                    destination=dest,
                    event=error_event,
                )
                raise
            last_exc = exc
        except httpx.RequestError as exc:
            last_exc = exc

        if attempt < MAX_RETRIES - 1:
            delay = RETRY_BACKOFF[attempt]
            dest.warning(
                retry_event,
                endpoint=url,
                attempt=attempt + 1,
                delay=delay,
                context=context,
            )
            await asyncio.sleep(delay)

    _log_failure(
        url,
        context=context,
        last_exc=last_exc,
        retries_exhausted=True,
        destination=dest,
        event=error_event,
    )
    raise last_exc  # type: ignore[misc]


def request_sync_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    context: str = "",
    dest: Any = logger,
    error_event: str = "http_api_error",
    retry_event: str = "http_api_retry",
    parse_json: bool = True,
) -> dict[str, Any] | httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.request(
                method, url, json=json, params=params, headers=headers
            )
            resp.raise_for_status()
            if parse_json:
                return resp.json()
            return resp
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:
                _log_failure(
                    url,
                    context=context,
                    last_exc=exc,
                    retries_exhausted=False,
                    destination=dest,
                    event=error_event,
                )
                raise
            last_exc = exc
        except httpx.RequestError as exc:
            last_exc = exc

        if attempt < MAX_RETRIES - 1:
            delay = RETRY_BACKOFF[attempt]
            dest.warning(
                retry_event,
                endpoint=url,
                attempt=attempt + 1,
                delay=delay,
                context=context,
            )
            time.sleep(delay)

    _log_failure(
        url,
        context=context,
        last_exc=last_exc,
        retries_exhausted=True,
        destination=dest,
        event=error_event,
    )
    raise last_exc  # type: ignore[misc]

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services import unsloth_client
from app.services.http_retry import request_sync_with_retry, request_with_retry


def _mock_response(status_code: int, json_data: dict | None = None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = '{"error": "server"}' if json_data is None else ""
    resp.is_success = 200 <= status_code < 300
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            message=f"{status_code}",
            request=MagicMock(),
            response=resp,
        )
        if status_code >= 400
        else None
    )
    return resp


def _plain_text_response(status_code: int, text: str):
    """Build a mock response that raises JSONDecodeError on .json() — matching
    the vLLM 0.30.0 /v1/load_lora_adapter behavior."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    resp.is_success = 200 <= status_code < 300
    resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            message=f"{status_code}",
            request=MagicMock(),
            response=resp,
        )
        if status_code >= 400
        else None
    )
    return resp


def _always_return(response):
    """Return a side_effect function that always returns the same response."""

    def _fn(*args, **kwargs):
        return response

    return _fn


def _make_mock_client(request_side_effect):
    mock = AsyncMock()
    if callable(request_side_effect) and not isinstance(request_side_effect, MagicMock):
        mock.request = AsyncMock(side_effect=request_side_effect)
    else:
        mock.request = AsyncMock(side_effect=request_side_effect)
    return mock


def test_retries_then_succeeds_on_3rd_attempt():
    responses = [
        _mock_response(500),
        _mock_response(502),
        _mock_response(200, {"job_id": "j1"}),
    ]
    mock_client = _make_mock_client(responses)

    with patch.object(unsloth_client, "_get_client", return_value=mock_client):
        with patch("app.services.http_retry.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(unsloth_client.start_training("run-1", "model", {}))

    assert result == {"job_id": "j1"}
    assert mock_client.request.call_count == 3


def test_raises_after_max_retries_exhausted():
    mock_client = _make_mock_client(_always_return(_mock_response(503)))

    with patch.object(unsloth_client, "_get_client", return_value=mock_client):
        with patch("app.services.http_retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(unsloth_client.get_training_status("run-2"))

    assert mock_client.request.call_count == unsloth_client._MAX_RETRIES


def test_exponential_backoff_delays():
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    mock_client = _make_mock_client(_always_return(_mock_response(500)))

    with patch.object(unsloth_client, "_get_client", return_value=mock_client):
        with patch("app.services.http_retry.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(unsloth_client.stop_training("run-3"))

    assert (
        sleep_calls == unsloth_client._RETRY_BACKOFF[: unsloth_client._MAX_RETRIES - 1]
    )


def test_no_retry_on_400():
    mock_client = _make_mock_client(_always_return(_mock_response(400)))

    with patch.object(unsloth_client, "_get_client", return_value=mock_client):
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(unsloth_client.start_training("run-4", "model", {}))

    assert mock_client.request.call_count == 1


def test_no_retry_on_404():
    mock_client = _make_mock_client(_always_return(_mock_response(404)))

    with patch.object(unsloth_client, "_get_client", return_value=mock_client):
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(unsloth_client.get_training_status("run-5"))

    assert mock_client.request.call_count == 1


def test_retries_on_timeout_then_succeeds():
    mock_client = _make_mock_client(
        [
            httpx.ConnectTimeout("connection timed out"),
            _mock_response(200, {"status": "ok"}),
        ]
    )

    with patch.object(unsloth_client, "_get_client", return_value=mock_client):
        with patch("app.services.http_retry.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(unsloth_client.start_training("run-6", "model", {}))

    assert result == {"status": "ok"}
    assert mock_client.request.call_count == 2


def test_retries_on_read_error_then_succeeds():
    mock_client = _make_mock_client(
        [
            httpx.ReadError("connection reset"),
            httpx.ReadError("connection reset"),
            _mock_response(200, {"status": "done"}),
        ]
    )

    with patch.object(unsloth_client, "_get_client", return_value=mock_client):
        with patch("app.services.http_retry.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(unsloth_client.get_training_status("run-7"))

    assert result == {"status": "done"}
    assert mock_client.request.call_count == 3


# ---------------------------------------------------------------------------
# parse_json parameter (issue #167 — Phase 2C plain-text vLLM responses)
# ---------------------------------------------------------------------------


def test_sync_request_parse_json_true_returns_dict():
    """Default behaviour: parse_json=True returns parsed JSON dict."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"ok": True}
    resp.raise_for_status = MagicMock(return_value=None)

    mock_client = MagicMock()
    mock_client.request = MagicMock(return_value=resp)

    result = request_sync_with_retry(
        mock_client, "POST", "http://vllm:8000/v1/load_lora_adapter", parse_json=True
    )
    assert result == {"ok": True}


def test_sync_request_parse_json_false_returns_response_object():
    """parse_json=False returns the raw httpx.Response without calling .json()."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.text = "Success: LoRA adapter 'x' added successfully."
    resp.raise_for_status = MagicMock(return_value=None)

    mock_client = MagicMock()
    mock_client.request = MagicMock(return_value=resp)

    result = request_sync_with_retry(
        mock_client, "POST", "http://vllm:8000/v1/load_lora_adapter", parse_json=False
    )
    assert result is resp
    resp.json.assert_not_called()


def test_sync_request_parse_json_false_plain_text_no_error():
    """Phase 2C regression: plain-text HTTP 200 does not raise JSONDecodeError.

    This is the exact scenario discovered during the Phase 2C live rehearsal:
    vLLM 0.30.0 /v1/load_lora_adapter returns HTTP 200 with Content-Type
    text/plain and body 'Success: LoRA adapter ... added successfully.'
    The old code unconditionally called resp.json(), which raised
    JSONDecodeError, which propagated as STAGING_DEPLOY_NOT_ALLOWED.
    """
    resp = _plain_text_response(
        200, "Success: LoRA adapter 'smoke-llm-v3-v1' added successfully."
    )

    mock_client = MagicMock()
    mock_client.request = MagicMock(return_value=resp)

    with patch("app.services.http_retry.time.sleep"):
        result = request_sync_with_retry(
            mock_client,
            "POST",
            "http://vllm:8000/v1/load_lora_adapter",
            parse_json=False,
        )
    assert result is resp
    assert result.text == "Success: LoRA adapter 'smoke-llm-v3-v1' added successfully."


def test_sync_request_parse_json_true_still_raises_on_non_json():
    """parse_json=True still raises JSONDecodeError for non-JSON 200 responses
    (preserves existing error propagation behaviour)."""
    resp = _plain_text_response(200, "not json at all")

    mock_client = MagicMock()
    mock_client.request = MagicMock(return_value=resp)

    with pytest.raises(json.JSONDecodeError):
        with patch("app.services.http_retry.time.sleep"):
            request_sync_with_retry(
                mock_client,
                "POST",
                "http://vllm:8000/v1/load_lora_adapter",
                parse_json=True,
            )


# ---------------------------------------------------------------------------
# Async parse_json tests (mirror the sync tests above)
# ---------------------------------------------------------------------------


def test_async_request_parse_json_true_returns_dict():
    """Default behaviour: parse_json=True returns parsed JSON dict."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"ok": True}
    resp.raise_for_status = MagicMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.request = AsyncMock(return_value=resp)

    with patch("app.services.http_retry.asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(
            request_with_retry(
                mock_client,
                "POST",
                "http://vllm:8000/v1/load_lora_adapter",
                parse_json=True,
            )
        )
    assert result == {"ok": True}


def test_async_request_parse_json_false_returns_response_object():
    """parse_json=False returns the raw httpx.Response without calling .json()."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.text = "Success: LoRA adapter 'x' added successfully."
    resp.raise_for_status = MagicMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.request = AsyncMock(return_value=resp)

    with patch("app.services.http_retry.asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(
            request_with_retry(
                mock_client,
                "POST",
                "http://vllm:8000/v1/load_lora_adapter",
                parse_json=False,
            )
        )
    assert result is resp
    resp.json.assert_not_called()


def test_async_request_parse_json_false_plain_text_no_error():
    """Phase 2C regression: plain-text HTTP 200 does not raise JSONDecodeError."""
    resp = _plain_text_response(
        200, "Success: LoRA adapter 'smoke-llm-v3-v1' added successfully."
    )

    mock_client = AsyncMock()
    mock_client.request = AsyncMock(return_value=resp)

    with patch("app.services.http_retry.asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(
            request_with_retry(
                mock_client,
                "POST",
                "http://vllm:8000/v1/load_lora_adapter",
                parse_json=False,
            )
        )
    assert result is resp

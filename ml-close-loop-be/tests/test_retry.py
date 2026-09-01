import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services import unsloth_client


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
        with patch("app.services.unsloth_client.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(unsloth_client.start_training("run-1", "model", {}))

    assert result == {"job_id": "j1"}
    assert mock_client.request.call_count == 3


def test_raises_after_max_retries_exhausted():
    mock_client = _make_mock_client(_always_return(_mock_response(503)))

    with patch.object(unsloth_client, "_get_client", return_value=mock_client):
        with patch("app.services.unsloth_client.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(unsloth_client.get_training_status("run-2"))

    assert mock_client.request.call_count == unsloth_client._MAX_RETRIES


def test_exponential_backoff_delays():
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    mock_client = _make_mock_client(_always_return(_mock_response(500)))

    with patch.object(unsloth_client, "_get_client", return_value=mock_client):
        with patch("app.services.unsloth_client.asyncio.sleep", side_effect=fake_sleep):
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
        with patch("app.services.unsloth_client.asyncio.sleep", new_callable=AsyncMock):
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
        with patch("app.services.unsloth_client.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(unsloth_client.get_training_status("run-7"))

    assert result == {"status": "done"}
    assert mock_client.request.call_count == 3

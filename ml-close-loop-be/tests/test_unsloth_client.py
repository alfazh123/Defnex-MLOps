import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.config import settings
from app.services import unsloth_client
from app.services.unsloth_client import (
    _map_training_config,
    _request_with_retry,
    close,
    get_available_models,
    get_default_model,
)


def test_get_available_models_splits_comma_separated(monkeypatch):
    monkeypatch.setattr(settings, "unsloth_models", "a,b, c ", raising=False)
    assert get_available_models() == ["a", "b", "c"]


def test_get_available_models_returns_default_when_empty(monkeypatch):
    monkeypatch.setattr(settings, "unsloth_models", "  ", raising=False)
    assert get_available_models() == [settings.unsloth_default_model]


def test_get_default_model_returns_default():
    assert get_default_model() == settings.unsloth_default_model


def test_map_training_config_lora_base_case():
    config = {"peft_method": "lora", "lora_r": 8, "lora_alpha": 32, "lora_dropout": 0.1}
    result = _map_training_config(config, "unsloth/Qwen3-0.6B")
    assert result["model_name"] == "unsloth/Qwen3-0.6B"
    assert result["training_type"] == "LoRA/QLoRA"
    assert result["use_lora"] is True
    assert result["use_rslora"] is False
    assert result["lora_r"] == 8


def test_map_training_config_none_is_full_finetune():
    result = _map_training_config({"peft_method": "none"}, "m")
    assert result["training_type"] == "Full Finetuning"
    assert result["use_lora"] is False


def test_map_training_config_dora_disables_rslora():
    result = _map_training_config({"peft_method": "dora"}, "m")
    assert result["training_type"] == "LoRA/QLoRA"
    assert result["use_rslora"] is False


def test_map_training_config_rslora_enables_rslora():
    result = _map_training_config({"peft_method": "rslora"}, "m")
    assert result["use_rslora"] is True


def test_map_training_config_learning_rate_stringified():
    result = _map_training_config({"learning_rate": 0.0002}, "m")
    assert result["learning_rate"] == "0.0002"


def test_map_training_config_defaults_populated_when_empty():
    result = _map_training_config({}, "m")
    assert result["num_epochs"] == 1
    assert result["max_seq_length"] == 2048
    assert result["random_seed"] == 42
    assert result["load_in_4bit"] is False
    assert result["hf_dataset"] == ""


def test_map_training_config_custom_lora_propagated():
    result = _map_training_config(
        {"lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.2}, "m"
    )
    assert result["lora_r"] == 64
    assert result["lora_alpha"] == 128
    assert result["lora_dropout"] == 0.2


def test_map_training_config_hf_dataset_fallback_from_dataset_path():
    result = _map_training_config(
        {"hf_dataset": "", "dataset_path": "my/local/path"}, "m"
    )
    assert result["hf_dataset"] == "my/local/path"


def test_close_sets_client_to_none():
    unsloth_client._client = httpx.AsyncClient()
    asyncio.run(close())
    assert unsloth_client._client is None


def _error_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code,
        text="error",
        request=httpx.Request("GET", "http://x/api"),
    )


def test_request_with_retry_retries_5xx_raises_on_last_failure():
    responses = [
        _error_response(500),
        _error_response(500),
        _error_response(500),
    ]
    mock_request = AsyncMock(side_effect=responses)
    with (
        patch.object(httpx.AsyncClient, "request", mock_request),
        patch("app.services.unsloth_client.asyncio.sleep", new=AsyncMock()),
    ):
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(_request_with_retry("GET", "http://x/api", context="c"))
    assert mock_request.call_count == 3


def test_request_with_retry_retries_5xx_then_succeeds():
    success = httpx.Response(
        200, json={"ok": True}, request=httpx.Request("GET", "http://x/api")
    )
    responses = [
        _error_response(500),
        _error_response(500),
        success,
    ]
    mock_request = AsyncMock(side_effect=responses)
    with (
        patch.object(httpx.AsyncClient, "request", mock_request),
        patch("app.services.unsloth_client.asyncio.sleep", new=AsyncMock()),
    ):
        result = asyncio.run(_request_with_retry("GET", "http://x/api", context="c"))
    assert result == {"ok": True}
    assert mock_request.call_count == 3


def test_request_with_retry_raises_immediately_on_4xx():
    mock_request = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "400 BAD REQUEST",
            request=httpx.Request("GET", "http://x"),
            response=httpx.Response(400, text="bad"),
        )
    )
    with (
        patch.object(httpx.AsyncClient, "request", mock_request),
        patch("app.services.unsloth_client.asyncio.sleep", new=AsyncMock()),
    ):
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(_request_with_retry("GET", "http://x/api", context="c"))
    assert mock_request.call_count == 1


def test_request_with_retry_retries_on_connection_error_then_raises():
    mock_request = AsyncMock(
        side_effect=httpx.ConnectError("boom", request=httpx.Request("GET", "http://x"))
    )
    with (
        patch.object(httpx.AsyncClient, "request", mock_request),
        patch("app.services.unsloth_client.asyncio.sleep", new=AsyncMock()),
    ):
        with pytest.raises(httpx.RequestError):
            asyncio.run(_request_with_retry("GET", "http://x/api", context="c"))
    assert mock_request.call_count == 3

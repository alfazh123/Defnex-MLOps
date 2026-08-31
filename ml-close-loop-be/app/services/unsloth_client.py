from __future__ import annotations

import os
from typing import Any

import httpx

_UNSLOTH_URL = os.getenv("UNSLOTH_STUDIO_URL", "http://unsloth-studio:8888")
_UNSLOTH_KEY = os.getenv("UNSLOTH_API_KEY", "")
_DEFAULT_MODEL = os.getenv("UNSLOTH_DEFAULT_MODEL", "unsloth/Qwen3-0.6B")

_client: httpx.AsyncClient | None = None


def _headers() -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json"}
    if _UNSLOTH_KEY:
        h["Authorization"] = f"Bearer {_UNSLOTH_KEY}"
    return h


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    return _client


def get_available_models() -> list[str]:
    raw = os.getenv("UNSLOTH_MODELS", "")
    if raw.strip():
        return [m.strip() for m in raw.split(",") if m.strip()]
    return [_DEFAULT_MODEL]


def get_default_model() -> str:
    return _DEFAULT_MODEL


def _map_training_config(config: dict[str, Any], base_model: str) -> dict[str, Any]:
    """Map our internal training config dict to the Unsloth Studio API format."""
    return {
        "model": base_model,
        "dataset_path": config.get("dataset_path", ""),
        "epochs": config.get("epochs", 1),
        "batch_size": config.get("batch_size", 4),
        "learning_rate": config.get("learning_rate", 2e-5),
        "max_seq_length": config.get("max_seq_length", 2048),
        "lora_r": config.get("lora_r", 16),
        "lora_alpha": config.get("lora_alpha", 32),
        "lora_dropout": config.get("lora_dropout", 0.05),
        "output_dir": config.get("output_dir", "./outputs"),
    }


async def start_training(
    training_run_id: str,
    base_model: str,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    client = await _get_client()
    payload = {
        "training_run_id": training_run_id,
        **_map_training_config(training_config, base_model),
    }
    resp = await client.post(
        f"{_UNSLOTH_URL}/api/train/start",
        json=payload,
        headers=_headers(),
    )
    resp.raise_for_status()
    return resp.json()


async def get_training_status(training_run_id: str) -> dict[str, Any]:
    client = await _get_client()
    resp = await client.get(
        f"{_UNSLOTH_URL}/api/train/status",
        params={"training_run_id": training_run_id},
        headers=_headers(),
    )
    resp.raise_for_status()
    return resp.json()


async def stop_training(training_run_id: str) -> dict[str, Any]:
    client = await _get_client()
    resp = await client.post(
        f"{_UNSLOTH_URL}/api/train/stop",
        json={"training_run_id": training_run_id},
        headers=_headers(),
    )
    resp.raise_for_status()
    return resp.json()


async def stream_progress(training_run_id: str):
    """Yield SSE events from Unsloth Studio progress endpoint."""
    client = await _get_client()
    url = f"{_UNSLOTH_URL}/api/train/progress"
    params = {"training_run_id": training_run_id}
    async with client.stream("GET", url, params=params, headers=_headers()) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                yield line[5:].strip()


async def close() -> None:
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None

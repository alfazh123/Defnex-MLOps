from __future__ import annotations

from typing import Any

import httpx
import structlog

from app.config import settings
from app.schemas.training import SUPPORTED_PEFT_METHODS

# Retry policy lives in app/services/http_retry.py (issue #40); the _MAX_RETRIES/_RETRY_BACKOFF
# aliases exist only because tests reference unsloth_client._MAX_RETRIES / _RETRY_BACKOFF.
from app.services.http_retry import (  # noqa: E402
    MAX_RETRIES as _MAX_RETRIES,  # noqa: F401 - re-exported for tests
    RETRY_BACKOFF as _RETRY_BACKOFF,  # noqa: F401 - re-exported for tests
    request_with_retry,
)

logger = structlog.get_logger(__name__)

_UNSLOTH_URL = settings.unsloth_studio_url
_UNSLOTH_KEY = settings.unsloth_api_key
_DEFAULT_MODEL = settings.unsloth_default_model

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
    raw = settings.unsloth_models
    if raw.strip():
        return [m.strip() for m in raw.split(",") if m.strip()]
    return [_DEFAULT_MODEL]


def get_default_model() -> str:
    return _DEFAULT_MODEL


def _map_training_config(config: dict[str, Any], base_model: str) -> dict[str, Any]:
    """Map our internal training config dict to the Unsloth Studio TrainingStartRequest.

    Accepts only the values in `SUPPORTED_PEFT_METHODS` (single source of truth in
    app/schemas/training.py). `qlora` forces 4-bit quantization via `load_in_4bit`
    and `rslora` forces `use_rslora` so they are not executed identically to `lora`
    (issue #34).

    Field mapping from our schema → Unsloth API:
    - model → model_name
    - dataset_path → hf_dataset
    - epochs → num_epochs
    - learning_rate → learning_rate (string)
    - peft_method → training_type ("LoRA/QLoRA")
    - load_in_4bit → load_in_4bit
    - lora_r, lora_alpha, lora_dropout → same names

    See: https://github.com/unslothai/unsloth/blob/main/studio/backend/models/training.py
    """
    peft_method = config.get("peft_method", "lora")
    if peft_method not in SUPPORTED_PEFT_METHODS:
        raise ValueError(
            f"peft_method {peft_method!r} is not supported; "
            f"supported values: {', '.join(SUPPORTED_PEFT_METHODS)}"
        )
    use_lora = True

    # learning_rate must be string for Unsloth
    lr = config.get("learning_rate", 2e-5)
    lr_str = str(lr) if lr is not None else "2e-4"

    # Dataset: hf_dataset or local path
    hf_dataset = config.get("hf_dataset") or config.get("dataset_path") or ""

    # Training type mapping — always LoRA/QLoRA; the "none" Full Finetuning branch was
    # removed because "none" is rejected at the request schema (issue #34).
    training_type = "LoRA/QLoRA"

    return {
        # Model
        "model_name": base_model,
        "load_in_4bit": config.get("load_in_4bit", False) or peft_method == "qlora",
        "max_seq_length": config.get("max_seq_length", 2048),
        "trust_remote_code": config.get("trust_remote_code", False),
        # Dataset
        "hf_dataset": hf_dataset,
        "format_type": config.get("format_type", "chatml"),
        "train_split": config.get("train_split", "train"),
        "eval_split": config.get("eval_split"),
        "eval_steps": config.get("eval_steps", 0.0),
        # Training type
        "training_type": training_type,
        # Hyperparameters
        "num_epochs": config.get("epochs", 1),
        "learning_rate": lr_str,
        "batch_size": config.get("batch_size", 1),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
        "warmup_steps": config.get("warmup_steps"),
        "warmup_ratio": config.get("warmup_ratio"),
        "max_steps": config.get("max_steps"),
        "save_steps": config.get("save_steps", 100),
        "weight_decay": config.get("weight_decay", 0.001),
        "max_grad_norm": config.get("max_grad_norm", 0.0),
        "random_seed": config.get("random_seed", 42),
        "packing": config.get("packing", False),
        "optim": config.get("optim", "adamw_8bit"),
        "lr_scheduler_type": config.get("lr_scheduler_type", "linear"),
        # LoRA
        "use_lora": use_lora,
        "lora_r": config.get("lora_r", 16),
        "lora_alpha": config.get("lora_alpha", 16),
        "lora_dropout": config.get("lora_dropout", 0.0),
        "target_modules": config.get("target_modules", []),
        # rslora only changes the scale factor alpha/sqrt(r), so it is honored via
        # use_rslora (issue #34) instead of being silently downgraded to plain LoRA.
        "use_rslora": config.get("use_rslora", False) or peft_method == "rslora",
        "use_loftq": config.get("use_loftq", False),
        # Gradient checkpointing
        "gradient_checkpointing": config.get("gradient_checkpointing", "unsloth"),
    }


async def _request_with_retry(
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
    context: str = "",
) -> dict[str, Any]:
    client = await _get_client()
    return await request_with_retry(
        client,
        method,
        url,
        json=json,
        params=params,
        headers=_headers(),
        context=context,
        dest=logger,
        error_event="unsloth_api_error",
        retry_event="unsloth_api_retry",
    )


async def start_training(
    training_run_id: str,
    base_model: str,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    payload = _map_training_config(training_config, base_model)
    return await _request_with_retry(
        "POST",
        f"{_UNSLOTH_URL}/api/train/start",
        json=payload,
        context=training_run_id,
    )


async def get_training_status(training_run_id: str) -> dict[str, Any]:
    return await _request_with_retry(
        "GET",
        f"{_UNSLOTH_URL}/api/train/status",
        params={"training_run_id": training_run_id},
        context=training_run_id,
    )


async def stop_training(training_run_id: str) -> dict[str, Any]:
    return await _request_with_retry(
        "POST",
        f"{_UNSLOTH_URL}/api/train/stop",
        json={"training_run_id": training_run_id},
        context=training_run_id,
    )


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

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

TrainingRunStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]


class TrainingConfig(BaseModel):
    """Training configuration knobs (openapi.yaml TrainingConfig).

    Loosely typed (`extra="allow"`) per openapi.yaml's own
    `additionalProperties: true` — known properties are documented from
    model-artifact-versioning-lineage.md §6's sample record. Defaults below
    are only the PRD §11 confirmed baseline (peft_method, load_in_4bit,
    max_seq_length); the rest have no confirmed default and stay
    configurable/unset rather than hardcoding an example run's values.
    """

    model_config = ConfigDict(extra="allow")

    peft_method: str = "dora"
    load_in_4bit: bool = False
    lora_r: int | None = None
    lora_alpha: int | None = None
    learning_rate: float | str | None = None
    epochs: int | None = None
    max_seq_length: int = 4096


class TrainingRunCreateRequest(BaseModel):
    """Request to create a training run (openapi.yaml TrainingRunCreateRequest)."""

    dataset_id: str
    dataset_version: int
    model_id: str
    base_model: str
    training_config: TrainingConfig
    triggered_by: str | None = None


class TrainingRun(BaseModel):
    """A training run's stored record plus live status/progress (openapi.yaml TrainingRun)."""

    training_run_id: str
    status: TrainingRunStatus
    dataset_id: str
    dataset_version: int
    model_id: str
    base_model: str
    training_config: TrainingConfig
    triggered_by: str | None = None
    created_at: datetime
    current_epoch: int | None = None
    current_step: int | None = None
    train_loss: float | None = None
    eval_loss: float | None = None
    model_version: int | None = None

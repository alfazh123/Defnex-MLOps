from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

TrainingRunStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]
PeftMethod = Literal["lora", "qlora", "dora", "qdora", "rslora"]


class TrainingConfig(BaseModel):
    """Training configuration knobs mapped to Unsloth Studio TrainingStartRequest.

    Loosely typed (`extra="allow"`) per openapi.yaml's own
    `additionalProperties: true` — known properties are documented from
    model-artifact-versioning-lineage.md §6's sample record. Defaults match
    Unsloth Studio defaults where confirmed, otherwise use sensible fallbacks.

    See: https://github.com/unslothai/unsloth/blob/main/studio/backend/models/training.py
    """

    model_config = ConfigDict(extra="allow")

    # PEFT method
    peft_method: PeftMethod = "lora"
    load_in_4bit: bool = False

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = []
    use_loftq: bool = False

    # Dataset
    hf_dataset: str = ""
    format_type: str = "chatml"
    train_split: str = "train"
    eval_split: str | None = None
    eval_steps: float = 0.0

    # Training hyperparameters
    learning_rate: float | str | None = None
    epochs: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_steps: int | None = None
    warmup_ratio: float | None = None
    max_steps: int | None = None
    save_steps: int = 100
    weight_decay: float = 0.001
    max_grad_norm: float = 0.0
    random_seed: int = 42
    packing: bool = False
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    max_seq_length: int = 4096

    # Gradient checkpointing
    gradient_checkpointing: str = "unsloth"

    # Model options
    trust_remote_code: bool = False


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

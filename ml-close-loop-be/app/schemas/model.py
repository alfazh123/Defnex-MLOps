from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.schemas.training import TrainingConfig

ModelLifecycleStatus = Literal[
    "REGISTERED", "EVALUATED", "PROMOTED", "REJECTED", "DEPLOYED", "RETIRED"
]
ArtifactType = Literal["adapter", "merged", "gguf"]


class Artifact(BaseModel):
    """A stored model artifact (openapi.yaml Artifact)."""

    type: ArtifactType
    uri: str
    size_bytes: int | None = None
    checksum: str | None = None


class EvalLossTrend(BaseModel):
    previous_version_eval_loss: float | None = None
    this_version_eval_loss: float


class QualitativeComparison(BaseModel):
    question_table_version: int
    wins: int
    losses: int
    ties: int
    total: int


class GeneralDomainRegressionCheck(BaseModel):
    checked: bool
    regressions_found: list[dict] = []


class EvaluationObject(BaseModel):
    """The three-signal evaluation payload (openapi.yaml EvaluationObject)."""

    eval_loss_trend: EvalLossTrend | None = None
    qualitative_comparison: QualitativeComparison | None = None
    general_domain_regression_check: GeneralDomainRegressionCheck | None = None


class EvaluationUpdateRequest(BaseModel):
    """Partial evaluation payload (openapi.yaml EvaluationUpdateRequest) - any subset of the
    three signal fields may be submitted; the backend merges onto the existing record.

    `eval_set_id`/`eval_set_version` name the stored golden/eval set the signals were
    measured against (issue #43); they are stored once, on the model version, and may be
    (re)submitted alongside any signal."""

    eval_set_id: str | None = None
    eval_set_version: int | None = None
    eval_loss_trend: EvalLossTrend | None = None
    qualitative_comparison: QualitativeComparison | None = None
    general_domain_regression_check: GeneralDomainRegressionCheck | None = None


class EvaluationSubmitResponse(BaseModel):
    """Inline response shape for POST .../evaluation (openapi.yaml, not a named component)."""

    evaluation: EvaluationObject
    status: ModelLifecycleStatus


class ModelSummary(BaseModel):
    """Lightweight list-view entry for GET /models (openapi.yaml ModelSummary)."""

    model_id: str
    latest_version: int
    status: ModelLifecycleStatus


class ModelRegistryRecord(BaseModel):
    """Full model registry record (openapi.yaml ModelRegistryRecord,
    model-artifact-versioning-lineage.md §8)."""

    model_id: str
    version: int
    # Issue #38: stable version name `{model_id}-{base_model_slug}-v{N}`.
    name: str
    status: ModelLifecycleStatus
    training_run_id: str
    base_model: str
    dataset_id: str
    dataset_version: int
    dataset_validation_report_ref: str | None = None
    training_config: TrainingConfig
    created_at: datetime
    created_by: str | None = None
    evaluation: EvaluationObject | None = None
    eval_set_id: str | None = None
    eval_set_version: int | None = None
    artifacts: list[Artifact]
    promotion_decision_ref: str | None = None
    previous_model_id: str | None = None

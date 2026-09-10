from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.schemas.training import TrainingConfig

ModelLifecycleStatus = Literal[
    "REGISTERED",
    "EVALUATED",
    "PROMOTED",
    "REJECTED",
    "DEPLOYED",
    "RETIRED",
    "ARCHIVED",
    "STAGING",
    "VALIDATED",
    "PRODUCTION",
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
    """Partial evaluation payload - any subset of the three signal fields may be submitted;
    the backend merges onto the existing record (`model_service.submit_evaluation`).

    `eval_set_id`/`eval_set_version` name the stored golden/eval set the signals were
    measured against (issue #43); they are stored once, on the model version, and may be
    (re)submitted alongside any signal.

    Issue #128: no longer reachable from the public API with caller-supplied signal values -
    `POST .../evaluation` now takes `EvaluationTriggerRequest` instead. This schema is kept as
    the internal shape `model_service.submit_evaluation` accepts, used by the evaluation worker
    (app/workers/evaluation_worker.py, once it has computed real signals via ServingBackend) and
    by tests that build fixtures directly through the service layer."""

    eval_set_id: str | None = None
    eval_set_version: int | None = None
    eval_loss_trend: EvalLossTrend | None = None
    qualitative_comparison: QualitativeComparison | None = None
    general_domain_regression_check: GeneralDomainRegressionCheck | None = None


class EvaluationTriggerRequest(BaseModel):
    """Request body for `POST .../evaluation` (issue #128, openapi.yaml EvaluationUpdateRequest).

    Trigger-only: the endpoint no longer accepts caller-supplied signal numbers (`eval_loss_trend`,
    `qualitative_comparison`, `general_domain_regression_check`) - those are now computed
    server-side by the evaluation worker against the stored golden/eval set (issue #43) via
    `ServingBackend`. `extra="forbid"` makes a request still shaped like the old manual payload
    fail with 422 rather than silently accepting/ignoring caller-controlled evaluation results.
    `eval_set_id`/`eval_set_version` (optional) name/override the golden set to evaluate against;
    when omitted, the model version's already-stored eval set reference is used."""

    model_config = ConfigDict(extra="forbid")

    eval_set_id: str | None = None
    eval_set_version: int | None = None


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
    training_config_hash: str
    created_at: datetime
    created_by: str | None = None
    evaluation: EvaluationObject | None = None
    eval_set_id: str | None = None
    eval_set_version: int | None = None
    artifacts: list[Artifact]
    promotion_decision_ref: str | None = None
    previous_model_id: str | None = None

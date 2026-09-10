"""Server-side evaluation tests (issue #128): `evaluation_engine.compute_evaluation_update`
and `evaluation_worker.process_next_evaluation` - real inference through `ServingBackend`
against a stored golden/eval set (#43), no caller-supplied numbers, no real GPU (mock backend
only, per CLAUDE.md's GPU constraint)."""

from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.eval_set import EvalSetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import (
    dataset_service,
    eval_set_service,
    model_service,
    training_service,
)
from app.services.evaluation_engine import compute_evaluation_update
from app.services.serving import InferenceError, MockServingBackend, ServingError
from app.workers.evaluation_worker import (
    claim_pending_evaluation,
    process_next_evaluation,
)


def _registered_model_version(
    db_session, *, model_id="qwen-sft-domain-x", eval_loss=0.84
):
    dataset_version = dataset_service.create_dataset_version(
        db_session,
        "no_robots",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="HuggingFaceH4/no_robots",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )
    training_run = training_service.create_training_run(
        db_session,
        dataset_version,
        TrainingRunCreateRequest(
            dataset_id="no_robots",
            dataset_version=1,
            model_id=model_id,
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
        ),
    )
    training_service.claim_training_run(db_session, training_run)
    if eval_loss is not None:
        training_run.eval_loss = eval_loss
    training_service.complete_training_run(
        db_session, training_run, artifact_uri="file:///tmp/adapter"
    )
    return model_service.register_model_version(db_session, training_run)


def _golden_set(db_session, eval_set_id="domain-benchmark", n=2):
    records = [
        {"messages": [{"role": "user", "content": f"golden-question-{i}"}]}
        for i in range(n)
    ]
    version = eval_set_service.create_eval_set_version(
        db_session, eval_set_id, EvalSetVersionCreateRequest(records=records)
    )
    return version.version


class _AlwaysFailBackend(MockServingBackend):
    """A backend whose adapter loads fine but every generation fails - exercises the
    loss/regression path without a real GPU."""

    def generate(self, prompt: str, model_id: str, version: int) -> str:
        raise InferenceError("simulated generation failure")


class _DeployFailsBackend(MockServingBackend):
    def deploy(self, model_version) -> None:
        raise ServingError("simulated adapter load failure")


def test_compute_evaluation_update_runs_real_inference_via_serving_backend(db_session):
    model_version = _registered_model_version(db_session)
    eval_version = _golden_set(db_session, n=3)
    model_service.trigger_evaluation(
        db_session,
        model_version,
        eval_set_id="domain-benchmark",
        eval_set_version=eval_version,
    )
    backend = MockServingBackend()

    update = compute_evaluation_update(db_session, model_version, backend)

    # Real (mock) inference happened: deploy/generate/unload actually reached the backend.
    assert backend.deployed == [(model_version.model_id, model_version.version)]
    assert backend.unloaded == [(model_version.model_id, model_version.version)]
    assert update.qualitative_comparison.wins == 3
    assert update.qualitative_comparison.losses == 0
    assert update.qualitative_comparison.total == 3
    assert update.general_domain_regression_check.checked is True
    assert update.general_domain_regression_check.regressions_found == []
    assert update.eval_loss_trend.this_version_eval_loss == 0.84
    assert update.eval_loss_trend.previous_version_eval_loss is None


def test_compute_evaluation_update_previous_version_eval_loss_chains(db_session):
    v1 = _registered_model_version(db_session, eval_loss=0.9)
    eval_version = _golden_set(db_session, n=1)
    model_service.trigger_evaluation(
        db_session, v1, eval_set_id="domain-benchmark", eval_set_version=eval_version
    )
    model_service.submit_evaluation(
        db_session,
        v1,
        compute_evaluation_update(db_session, v1, MockServingBackend()),
    )
    assert v1.status == "EVALUATED"

    v2 = _registered_model_version(db_session, eval_loss=0.5)
    model_service.trigger_evaluation(
        db_session, v2, eval_set_id="domain-benchmark", eval_set_version=eval_version
    )

    update = compute_evaluation_update(db_session, v2, MockServingBackend())

    assert update.eval_loss_trend.this_version_eval_loss == 0.5
    assert update.eval_loss_trend.previous_version_eval_loss == 0.9


def test_compute_evaluation_update_records_losses_on_inference_error(db_session):
    model_version = _registered_model_version(db_session)
    eval_version = _golden_set(db_session, n=2)
    model_service.trigger_evaluation(
        db_session,
        model_version,
        eval_set_id="domain-benchmark",
        eval_set_version=eval_version,
    )

    update = compute_evaluation_update(db_session, model_version, _AlwaysFailBackend())

    assert update.qualitative_comparison.wins == 0
    assert update.qualitative_comparison.losses == 2
    assert len(update.general_domain_regression_check.regressions_found) == 2


def test_compute_evaluation_update_raises_without_eval_set_reference(db_session):
    model_version = _registered_model_version(db_session)

    try:
        compute_evaluation_update(db_session, model_version, MockServingBackend())
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "eval_set_id" in str(exc)


def test_compute_evaluation_update_raises_when_eval_set_version_not_found(db_session):
    model_version = _registered_model_version(db_session)
    model_service.trigger_evaluation(
        db_session, model_version, eval_set_id="does-not-exist", eval_set_version=1
    )

    try:
        compute_evaluation_update(db_session, model_version, MockServingBackend())
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "not found" in str(exc)


def test_compute_evaluation_update_raises_when_adapter_load_fails(db_session):
    model_version = _registered_model_version(db_session)
    eval_version = _golden_set(db_session, n=1)
    model_service.trigger_evaluation(
        db_session,
        model_version,
        eval_set_id="domain-benchmark",
        eval_set_version=eval_version,
    )

    try:
        compute_evaluation_update(db_session, model_version, _DeployFailsBackend())
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "could not load adapter" in str(exc)


def test_claim_pending_evaluation_is_compare_and_set(db_session):
    model_version = _registered_model_version(db_session)
    model_service.trigger_evaluation(
        db_session, model_version, eval_set_id="x", eval_set_version=1
    )
    db_session.flush()

    first = claim_pending_evaluation(db_session, model_version)
    second = claim_pending_evaluation(db_session, model_version)

    assert first is True
    assert second is False
    assert model_version.evaluation_requested is False


def test_process_next_evaluation_returns_none_when_queue_empty(db_session):
    assert process_next_evaluation(db_session) is None


def test_process_next_evaluation_end_to_end_transitions_to_evaluated(db_session):
    model_version = _registered_model_version(db_session)
    eval_version = _golden_set(db_session, n=2)
    model_service.trigger_evaluation(
        db_session,
        model_version,
        eval_set_id="domain-benchmark",
        eval_set_version=eval_version,
    )

    processed = process_next_evaluation(db_session, backend=MockServingBackend())

    assert processed is model_version
    assert model_version.status == "EVALUATED"
    assert model_version.evaluation_requested is False
    assert model_version.qualitative_comparison["wins"] == 2


def test_process_next_evaluation_skips_gracefully_without_eval_set(db_session):
    """No eval_set reference: the worker must not crash the poll loop (mirrors
    training_worker's "skip this cycle" posture for a lock timeout)."""
    model_version = _registered_model_version(db_session)
    model_version.evaluation_requested = True
    db_session.flush()

    processed = process_next_evaluation(db_session, backend=MockServingBackend())

    assert processed is model_version
    assert model_version.status == "REGISTERED"
    assert model_version.evaluation_requested is False
    assert model_version.qualitative_comparison is None

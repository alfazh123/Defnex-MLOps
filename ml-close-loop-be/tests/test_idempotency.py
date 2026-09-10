"""Issue #81 — deployment idempotency tests."""

import time

from app.services import deployment_service


def _promoted_model_version(db_session):
    """Drive a model version to PROMOTED."""
    from app.schemas.dataset import DatasetVersionCreateRequest
    from app.schemas.model import (
        EvalLossTrend,
        EvaluationUpdateRequest,
        GeneralDomainRegressionCheck,
        QualitativeComparison,
    )
    from app.schemas.promotion import DecisionCreateRequest
    from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
    from app.services import (
        dataset_service,
        model_service,
        promotion_service,
        training_service,
    )

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
            model_id="idempotent-model",
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
        ),
    )
    training_service.start_training_run(db_session, training_run)
    training_service.complete_training_run(
        db_session, training_run, artifact_uri="file:///tmp/adapter"
    )
    model_version = model_service.register_model_version(db_session, training_run)
    model_service.submit_evaluation(
        db_session,
        model_version,
        EvaluationUpdateRequest(
            eval_loss_trend=EvalLossTrend(this_version_eval_loss=0.84),
            qualitative_comparison=QualitativeComparison(
                question_table_version=1, wins=13, losses=5, ties=2, total=20
            ),
            general_domain_regression_check=GeneralDomainRegressionCheck(
                checked=True, regressions_found=[]
            ),
        ),
    )
    promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="PROMOTED", decided_by="reviewer-1", rationale="Signals aligned."
        ),
    )
    return model_version


class TestIdempotencyCache:
    def test_check_idempotency_returns_none_when_no_key(self):
        assert deployment_service.check_idempotency(None, 1) is None

    def test_check_idempotency_returns_none_when_not_cached(self):
        assert deployment_service.check_idempotency("key-abc", 999) is None

    def test_store_and_check_idempotency_returns_cached_result(self):
        result = {"model_id": "m1", "current_deployed_version": 1}
        deployment_service.store_idempotency("key-1", 10, result)
        cached = deployment_service.check_idempotency("key-1", 10)
        assert cached == result
        # cleanup
        del deployment_service._DEPLOY_IDEMPOTENCY_CACHE["key-1:10"]

    def test_different_model_versions_get_independent_results(self):
        r1 = {"model_id": "m1", "current_deployed_version": 1}
        r2 = {"model_id": "m1", "current_deployed_version": 2}
        deployment_service.store_idempotency("key-x", 10, r1)
        deployment_service.store_idempotency("key-x", 20, r2)
        assert deployment_service.check_idempotency("key-x", 10) == r1
        assert deployment_service.check_idempotency("key-x", 20) == r2
        # cleanup
        del deployment_service._DEPLOY_IDEMPOTENCY_CACHE["key-x:10"]
        del deployment_service._DEPLOY_IDEMPOTENCY_CACHE["key-x:20"]

    def test_expired_key_returns_none(self):
        result = {"model_id": "m1", "current_deployed_version": 3}
        deployment_service._DEPLOY_IDEMPOTENCY_CACHE["expired-key:5"] = (
            result,
            time.time() - 7200,  # 2 hours ago, past TTL
        )
        assert deployment_service.check_idempotency("expired-key", 5) is None
        # cleanup
        deployment_service._DEPLOY_IDEMPOTENCY_CACHE.pop("expired-key:5", None)

    def test_store_idempotency_none_key_is_noop(self):
        deployment_service.store_idempotency(None, 1, {})
        assert len(deployment_service._DEPLOY_IDEMPOTENCY_CACHE) == 0

"""Eval-set service (issue #43): version handling and the overlap guard's normalization."""

import pytest

from app.schemas.eval_set import EvalSetVersionCreateRequest
from app.services import eval_set_service


def test_create_eval_set_version_increments_per_eval_set(db_session):
    first = eval_set_service.create_eval_set_version(
        db_session,
        "domain-benchmark",
        EvalSetVersionCreateRequest(records=[{"id": "a"}]),
    )
    second = eval_set_service.create_eval_set_version(
        db_session,
        "domain-benchmark",
        EvalSetVersionCreateRequest(records=[{"id": "b"}]),
    )
    separate = eval_set_service.create_eval_set_version(
        db_session,
        "other-benchmark",
        EvalSetVersionCreateRequest(records=[{"id": "c"}]),
    )

    assert (first.version, second.version, separate.version) == (1, 2, 1)


def test_overlap_check_uses_normalized_user_content(db_session):
    from app.schemas.dataset import DatasetVersionCreateRequest
    from app.services import dataset_service, validation_service

    dataset_version = dataset_service.create_dataset_version(
        db_session,
        "no_robots",
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="HuggingFaceH4/no_robots",
            source_format="chatml",
        ),
    )
    training = {
        "id": "t1",
        "messages": [
            {"role": "user", "content": "  What is the capital of France?  "},
            {"role": "assistant", "content": " ".join(f"word{i}" for i in range(25))},
        ],
        "metadata": {"source_dataset": "no_robots", "source_id": "t1"},
    }
    validation_service.validate_dataset_version(
        db_session, dataset_version, [training], rule_set_version="2.2.0"
    )

    with pytest.raises(ValueError):
        eval_set_service.create_eval_set_version(
            db_session,
            "domain-benchmark",
            EvalSetVersionCreateRequest(
                records=[
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "What is the capital of France?",
                            }
                        ]
                    }
                ]
            ),
        )

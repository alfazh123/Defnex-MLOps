import hashlib
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.model import (
    EvalLossTrend,
    EvaluationUpdateRequest,
    GeneralDomainRegressionCheck,
    QualitativeComparison,
)
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, model_service, training_service


def _training_config_hash(config: dict) -> str:
    """Mirror model_service._training_config_hash (issue #64), so metadata.json and the
    row-level checksum can be asserted exactly."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _dataset_version(db_session, dataset_id: str = "no_robots"):
    return dataset_service.create_dataset_version(
        db_session,
        dataset_id,
        DatasetVersionCreateRequest(
            source_type="huggingface",
            source_dataset="HuggingFaceH4/no_robots",
            source_commit_or_snapshot_date="2026-08-01",
            source_format="chatml",
        ),
    )


def _completed_training_run(db_session, **overrides):
    dataset_version = overrides.pop("dataset_version", None) or _dataset_version(
        db_session
    )
    defaults = {
        "dataset_id": dataset_version.dataset_id,
        "dataset_version": dataset_version.version,
        "model_id": "qwen-sft-domain-x",
        "base_model": "Qwen/Qwen3.8-27B",
        "training_config": TrainingConfig(),
    }
    defaults.update(overrides)
    training_run = training_service.create_training_run(
        db_session, dataset_version, TrainingRunCreateRequest(**defaults)
    )
    training_service.claim_training_run(db_session, training_run)
    training_service.complete_training_run(
        db_session, training_run, artifact_uri="file:///tmp/adapter"
    )
    return training_run


def test_register_model_version_from_completed_run(db_session):
    training_run = _completed_training_run(db_session)

    model_version = model_service.register_model_version(db_session, training_run)

    assert model_version.model_id == "qwen-sft-domain-x"
    assert model_version.version == 1
    assert model_version.status == "REGISTERED"
    assert model_version.training_run_id == training_run.training_run_id
    assert model_version.base_model == "Qwen/Qwen3.8-27B"
    assert model_version.training_config["peft_method"] == "lora"
    assert model_version.artifacts == [
        {"type": "adapter", "uri": "file:///tmp/adapter"}
    ]


def test_register_model_version_increments_version_per_model_id(db_session):
    first_run = _completed_training_run(db_session)
    model_service.register_model_version(db_session, first_run)

    second_run = _completed_training_run(db_session)
    second_version = model_service.register_model_version(db_session, second_run)

    assert second_version.version == 2


def test_register_model_version_requires_completed_status(db_session):
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
            model_id="qwen-sft-domain-x",
            base_model="Qwen/Qwen3.8-27B",
            training_config=TrainingConfig(),
        ),
    )

    with pytest.raises(ValueError):
        model_service.register_model_version(db_session, training_run)


def test_submit_evaluation_partial_stays_registered(db_session):
    training_run = _completed_training_run(db_session)
    model_version = model_service.register_model_version(db_session, training_run)

    model_service.submit_evaluation(
        db_session,
        model_version,
        EvaluationUpdateRequest(
            eval_loss_trend=EvalLossTrend(this_version_eval_loss=0.84)
        ),
    )

    assert model_version.status == "REGISTERED"
    evaluation = model_service.get_evaluation(model_version)
    assert evaluation.eval_loss_trend.this_version_eval_loss == 0.84
    assert evaluation.qualitative_comparison is None
    assert evaluation.general_domain_regression_check is None


def test_submit_evaluation_persists_eval_set_reference(db_session):
    training_run = _completed_training_run(db_session)
    model_version = model_service.register_model_version(db_session, training_run)

    model_service.submit_evaluation(
        db_session,
        model_version,
        EvaluationUpdateRequest(eval_set_id="domain-benchmark", eval_set_version=2),
    )

    assert model_version.eval_set_id == "domain-benchmark"
    assert model_version.eval_set_version == 2
    record = model_service.to_schema(model_version)
    assert record.eval_set_id == "domain-benchmark"
    assert record.eval_set_version == 2


def test_submit_evaluation_all_three_transitions_to_evaluated(db_session):
    training_run = _completed_training_run(db_session)
    model_version = model_service.register_model_version(db_session, training_run)

    model_service.submit_evaluation(
        db_session,
        model_version,
        EvaluationUpdateRequest(
            eval_set_id="domain-benchmark",
            eval_set_version=1,
            eval_loss_trend=EvalLossTrend(this_version_eval_loss=0.84),
        ),
    )
    model_service.submit_evaluation(
        db_session,
        model_version,
        EvaluationUpdateRequest(
            qualitative_comparison=QualitativeComparison(
                question_table_version=1, wins=13, losses=5, ties=2, total=20
            )
        ),
    )
    assert model_version.status == "REGISTERED"

    model_service.submit_evaluation(
        db_session,
        model_version,
        EvaluationUpdateRequest(
            general_domain_regression_check=GeneralDomainRegressionCheck(
                checked=True, regressions_found=[]
            )
        ),
    )

    assert model_version.status == "EVALUATED"
    evaluation = model_service.get_evaluation(model_version)
    assert evaluation.eval_loss_trend.this_version_eval_loss == 0.84
    assert evaluation.qualitative_comparison.wins == 13
    assert evaluation.general_domain_regression_check.checked is True


def test_list_models_has_no_n_plus_1(count_queries):
    with count_queries() as (session, counters):
        for idx in range(10):
            run = _completed_training_run(session, model_id=f"model-{idx}")
            model_service.register_model_version(session, run)
        session.commit()
        counters["n"] = 0
        model_service.list_models(session)
        count_10 = counters["n"]

    with count_queries() as (session, counters):
        run = _completed_training_run(session, model_id="model-0")
        model_service.register_model_version(session, run)
        session.commit()
        counters["n"] = 0
        model_service.list_models(session)
        count_1 = counters["n"]

    assert count_10 - count_1 < 9


# --- issue #38: version names, atomic versioning, immutable artifacts + metadata ---


def test_build_version_name_follows_project_base_model_v_contract():
    assert (
        model_service.build_version_name("qwen-sft-domain-x", "Qwen/Qwen3.8-27B", 1)
        == "qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1"
    )
    # base_model is slugged so the name is a safe filesystem directory name
    assert (
        model_service.build_version_name("m", "Some/Model:name (x)", 2)
        == "m-Some-Model-name-x-v2"
    )


def test_register_stores_and_exposes_version_name(db_session):
    training_run = _completed_training_run(db_session)

    model_version = model_service.register_model_version(db_session, training_run)

    assert model_version.name == "qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1"
    assert model_service.to_schema(model_version).name == model_version.name


def test_register_copies_git_commit_and_timestamps(db_session, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "deadbeef0123456789")
    training_run = _completed_training_run(db_session)

    model_version = model_service.register_model_version(db_session, training_run)

    assert model_version.git_commit == "deadbeef0123456789"
    assert model_version.training_started_at == training_run.started_at
    assert model_version.training_completed_at == training_run.finished_at


def test_register_stores_deterministic_training_config_hash(db_session):
    """Issue #64: the training config hash is stored on the row, exposed via to_schema,
    and is deterministic — identical configs hash identically, so the §37 lineage chain
    can point at one config hash.

    "Identical" now includes the dataset pin (issue #209 writes it into `training_config`),
    so both runs here are built from the *same* dataset version. Building a fresh version
    per run would make the configs genuinely differ, which is the point below, not a
    counterexample to it.
    """
    dataset_version = _dataset_version(db_session)
    first = model_service.register_model_version(
        db_session, _completed_training_run(db_session, dataset_version=dataset_version)
    )
    second = model_service.register_model_version(
        db_session, _completed_training_run(db_session, dataset_version=dataset_version)
    )

    expected = _training_config_hash(first.training_config)
    expected_16 = hashlib.sha256(
        json.dumps(second.training_config, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    assert isinstance(first.training_config_hash, str)
    assert len(first.training_config_hash) == 16
    assert (
        first.training_config_hash
        == second.training_config_hash
        == expected
        == expected_16
    )
    assert model_service.to_schema(first).training_config_hash == expected


def test_config_hash_discriminates_the_dataset_pin(db_session):
    """Issue #209: the pin is part of `training_config`, so two runs on different dataset
    versions hash differently even with identical knobs. That is what makes the config hash
    usable as a lineage discriminator rather than just a fingerprint of the hyperparameters.
    """
    v1 = _completed_training_run(db_session)
    v2 = _completed_training_run(db_session)

    assert (
        v1.training_config["dataset_pin"]["dataset_version"]
        != (v2.training_config["dataset_pin"]["dataset_version"])
    )
    first = model_service.register_model_version(db_session, v1)
    second = model_service.register_model_version(db_session, v2)
    assert first.training_config_hash != second.training_config_hash


def test_register_training_config_hash_is_sensitive_to_config(db_session):
    """Issue #64: a different training config must hash to a different id (the hash is a
    real discriminator, not a constant)."""
    changed = _completed_training_run(
        db_session, training_config=TrainingConfig(lora_r=32)
    )
    changed_version = model_service.register_model_version(db_session, changed)

    baseline = _completed_training_run(db_session)
    baseline_version = model_service.register_model_version(db_session, baseline)

    assert changed_version.training_config_hash != baseline_version.training_config_hash
    assert changed_version.training_config_hash == _training_config_hash(
        changed_version.training_config
    )


def test_register_with_staging_finalizes_immutable_artifact_and_metadata(
    db_session, tmp_path, monkeypatch
):
    """Issue #38 required test (metadata per-field): the run's staged output is moved into
    `{base}/{model_id}/{name}/`, metadata.json carries the full lineage, the stored artifact
    URI and the run's artifact_uri both point at the immutable path, and the staging dir is
    gone (moved, not copied)."""
    from app.services.artifact_storage import LocalFilesystemArtifactStorage

    monkeypatch.setenv("GIT_COMMIT", "deadbeef")
    training_run = _completed_training_run(db_session)

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "adapter_model.safetensors").write_bytes(b"merged-weights")

    model_version = model_service.register_model_version(
        db_session,
        training_run,
        staging_dir=str(staging),
        storage=LocalFilesystemArtifactStorage(base_dir=tmp_path / "store"),
    )

    name = "qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1"
    target = tmp_path / "store" / "qwen-sft-domain-x" / name
    assert model_version.artifacts[0]["uri"] == f"file://{target}"
    assert training_run.artifact_uri == f"file://{target}"
    assert not staging.exists()

    metadata = json.loads((target / "metadata.json").read_text())
    # datetimes are stringified by json.dumps(default=str) in finalize_version; the encoded
    # "merged-weights" payload-size encoding below must mirror _compute_checksum (issue #62).
    expected_checksum = hashlib.sha256(
        b"adapter_model.safetensors:14:merged-weights"
    ).hexdigest()
    assert metadata == {
        "model_id": "qwen-sft-domain-x",
        "name": name,
        "version": 1,
        "training_run_id": training_run.training_run_id,
        "dataset_id": "no_robots",
        "dataset_version": 1,
        "base_model": "Qwen/Qwen3.8-27B",
        "training_config": training_run.training_config,
        "training_config_hash": _training_config_hash(training_run.training_config),
        "git_commit": "deadbeef",
        "started_at": str(training_run.started_at),
        "finished_at": str(training_run.finished_at),
        "checksum": expected_checksum,
    }
    assert (target / "adapter_model.safetensors").read_bytes() == b"merged-weights"
    assert model_version.artifacts[0]["checksum"] == expected_checksum
    schema = model_service.to_schema(model_version)
    assert schema.artifacts[0].checksum == expected_checksum


def test_allocate_version_retries_when_first_pick_already_taken(
    db_session, monkeypatch
):
    """Deterministic trigger for the exact bug issue #38 fixes: the old code did a plain
    `MAX+1` select then a blind INSERT, so a stale MAX (a concurrent registration that just
    committed) 500'd as an IntegrityError. Here the first allocation attempt is forced to see
    a stale empty table (MAX=None -> v1, already taken by an earlier registration); the
    SAVEPOINT rollback + retry must still register a distinct version with no exception."""
    import sqlalchemy as sa

    first_run = _completed_training_run(db_session)
    model_service.register_model_version(db_session, first_run)  # takes v1
    second_run = _completed_training_run(db_session)

    real_scalar = db_session.scalar
    stale_attempts = {"n": 0}

    def stale_max_on_first_select(stmt, *args, **kwargs):
        # only the version-allocating MAX query is stale; other session queries behave normally
        if isinstance(stmt, sa.sql.expression.Select):
            cols = (
                stmt.selected_columns
                if hasattr(stmt, "selected_columns")
                else stmt.columns
            )
            if stmt.whereclause is not None and any(
                str(col) == "model_versions.version" for col in cols
            ):
                if stale_attempts["n"] == 0:
                    stale_attempts["n"] += 1
                    return None
        return real_scalar(stmt, *args, **kwargs)

    monkeypatch.setattr(db_session, "scalar", stale_max_on_first_select)

    model_version = model_service.register_model_version(db_session, second_run)

    assert stale_attempts["n"] == 1
    assert model_version.version == 2


def test_concurrent_registration_allocates_distinct_versions(tmp_path):
    """Issue #38 required test (concurrency): two threads registering the SAME model_id
    must produce versions N and N+1 with no unhandled exception. Follows the proven pattern
    from test_concurrent_workers_run_single_pending_job_once (issue #33): file-backed SQLite
    shared by two connections, each with its own engine/session. SQLite allows one writer at
    a time; the SAVEPOINT retry (catching IntegrityError + OperationalError) handles the
    lock contention. A small stagger ensures the first thread acquires the write lock before
    the second starts, avoiding a livelock where both keep colliding on every retry."""
    import threading
    import time

    db_path = str(tmp_path / "registry.db")
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    Base.metadata.create_all(engine)

    with Session(engine) as setup:
        first = _completed_training_run(setup, model_id="race-model")
        second = _completed_training_run(setup, model_id="race-model")
        setup.commit()
        first_id, second_id = first.training_run_id, second.training_run_id
    engine.dispose()

    versions: list[int] = []
    errors: list[Exception] = []
    barrier = threading.Event()

    def register(run_id, go_second=False):
        eng = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
        try:
            if go_second:
                barrier.wait(timeout=5)
                time.sleep(0.05)
            else:
                barrier.set()
            with Session(eng) as session:
                run = training_service.get_training_run(session, run_id)
                mv = model_service.register_model_version(session, run)
                versions.append(mv.version)
                session.commit()
        except Exception as exc:
            errors.append(exc)
        finally:
            eng.dispose()

    threads = [
        threading.Thread(target=register, args=(first_id, False)),
        threading.Thread(target=register, args=(second_id, True)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == [], f"concurrent registration raised: {errors}"
    assert sorted(versions) == [1, 2]


# --- issue #66: ARCHIVED is distinct from REJECTED and reversible ---


def _evaluated_version(db_session):
    training_run = _completed_training_run(db_session)
    model_version = model_service.register_model_version(db_session, training_run)
    model_service.submit_evaluation(
        db_session,
        model_version,
        EvaluationUpdateRequest(
            eval_set_id="domain-benchmark",
            eval_set_version=1,
            eval_loss_trend=EvalLossTrend(this_version_eval_loss=0.9),
            qualitative_comparison=QualitativeComparison(
                question_table_version=1, wins=10, losses=5, ties=5, total=20
            ),
            general_domain_regression_check=GeneralDomainRegressionCheck(
                checked=True, regressions_found=[]
            ),
        ),
    )
    assert model_version.status == "EVALUATED"
    return model_version


def _promoted_version(db_session):
    from app.schemas.promotion import DecisionCreateRequest
    from app.services import promotion_service

    model_version = _evaluated_version(db_session)
    promotion_service.create_decision(
        db_session,
        model_version,
        DecisionCreateRequest(
            decision="PROMOTED", decided_by="reviewer-1", rationale="ok"
        ),
    )
    assert model_version.status == "PROMOTED"
    return model_version


@pytest.mark.parametrize("status", ["PROMOTED", "RETIRED"])
def test_archive_from_promoted_or_retired(db_session, status):
    model_version = _promoted_version(db_session)
    model_version.status = status
    artifacts_before = list(model_version.artifacts)

    result = model_service.archive_model_version(db_session, model_version)

    assert result.status == "ARCHIVED"
    # archive is a soft-hide: the artifact is untouched and stays available for rollback
    assert model_version.artifacts == artifacts_before


@pytest.mark.parametrize(
    "status", ["REGISTERED", "EVALUATED", "REJECTED", "DEPLOYED", "ARCHIVED"]
)
def test_archive_rejects_invalid_source_status(db_session, status):
    model_version = _evaluated_version(db_session)
    model_version.status = status

    with pytest.raises(ValueError):
        model_service.archive_model_version(db_session, model_version)


@pytest.mark.parametrize("restore", ["PROMOTED", "REJECTED", "RETIRED"])
def test_unarchive_restores_to_previous_active_state(db_session, restore):
    model_version = _promoted_version(db_session)
    model_service.archive_model_version(db_session, model_version)
    assert model_version.status == "ARCHIVED"

    result = model_service.unarchive_model_version(
        db_session, model_version, restore_status=restore
    )

    assert result.status == restore


def test_unarchive_rejects_non_archived_source(db_session):
    model_version = _evaluated_version(db_session)
    model_version.status = "PROMOTED"

    with pytest.raises(ValueError):
        model_service.unarchive_model_version(
            db_session, model_version, restore_status="PROMOTED"
        )


def test_unarchive_rejects_invalid_restore_target(db_session):
    model_version = _promoted_version(db_session)
    model_service.archive_model_version(db_session, model_version)

    with pytest.raises(ValueError):
        model_service.unarchive_model_version(
            db_session, model_version, restore_status="DEPLOYED"
        )

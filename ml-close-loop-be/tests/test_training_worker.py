from app.schemas.dataset import DatasetVersionCreateRequest
from app.schemas.training import TrainingConfig, TrainingRunCreateRequest
from app.services import dataset_service, training_service
from app.workers.training_worker import process_next_job


class _StubRunner:
    def __init__(self, artifact_uri=None, error=None):
        self.artifact_uri = artifact_uri
        self.error = error
        self.calls = []

    def run(self, training_run):
        self.calls.append(training_run.training_run_id)
        if self.error:
            raise self.error
        return self.artifact_uri


def _queued_training_run(db_session):
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
    return training_service.create_training_run(
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


def test_process_next_job_returns_none_when_queue_empty(db_session):
    assert process_next_job(db_session, _StubRunner()) is None


def test_process_next_job_completes_on_success(db_session):
    training_run = _queued_training_run(db_session)
    runner = _StubRunner(artifact_uri="file:///tmp/adapter")

    processed = process_next_job(db_session, runner)

    assert processed.training_run_id == training_run.training_run_id
    assert processed.status == "COMPLETED"
    assert processed.artifact_uri == "file:///tmp/adapter"
    assert runner.calls == [training_run.training_run_id]


def test_process_next_job_fails_on_runner_exception(db_session):
    _queued_training_run(db_session)
    runner = _StubRunner(error=RuntimeError("out of memory"))

    processed = process_next_job(db_session, runner)

    assert processed.status == "FAILED"
    assert processed.error_message == "out of memory"


def test_process_next_job_picks_oldest_pending_first(db_session):
    first = _queued_training_run(db_session)
    second = _queued_training_run(db_session)
    second.created_at = first.created_at.replace(year=first.created_at.year + 1)
    db_session.flush()

    processed = process_next_job(db_session, _StubRunner(artifact_uri="uri"))

    assert processed.training_run_id == first.training_run_id


def test_process_next_job_with_runner_error_sets_failed_status(db_session):
    _queued_training_run(db_session)

    processed = process_next_job(
        db_session, _StubRunner(error=RuntimeError("cuda oom"))
    )

    assert processed.status == "FAILED"


def test_process_next_job_with_runner_error_records_message(db_session):
    _queued_training_run(db_session)

    processed = process_next_job(
        db_session, _StubRunner(error=RuntimeError("cuda oom"))
    )

    assert processed.status == "FAILED"
    assert processed.error_message == "cuda oom"


def test_process_next_job_registers_model_version_on_success(db_session):
    training_run = _queued_training_run(db_session)

    process_next_job(db_session, _StubRunner(artifact_uri="file:///tmp/adapter"))

    from app.services import model_service

    model_version = model_service.get_model_version(db_session, "qwen-sft-domain-x", 1)
    assert model_version is not None
    assert model_version.status == "REGISTERED"
    assert model_version.training_run_id == training_run.training_run_id

from app.schemas.dataset import DatasetVersionCreateRequest
from app.services import dataset_service


def _create_request(**overrides):
    defaults = {
        "source_type": "huggingface",
        "source_dataset": "HuggingFaceH4/no_robots",
        "source_commit_or_snapshot_date": "2026-08-01",
        "source_format": "chatml",
    }
    defaults.update(overrides)
    return DatasetVersionCreateRequest(**defaults)


def test_register_dataset_is_idempotent(db_session):
    first = dataset_service.register_dataset(db_session, "no_robots")
    second = dataset_service.register_dataset(db_session, "no_robots")
    assert first.dataset_id == second.dataset_id == "no_robots"


def test_create_dataset_version_registers_dataset_and_starts_at_1(db_session):
    version = dataset_service.create_dataset_version(db_session, "no_robots", _create_request())

    assert version.dataset_id == "no_robots"
    assert version.version == 1
    assert version.status == "PROCESSING"
    assert version.source_url_or_hf_id == "HuggingFaceH4/no_robots"
    assert version.row_count is None
    assert version.cleaning_steps_applied == []


def test_create_dataset_version_increments_per_dataset(db_session):
    dataset_service.create_dataset_version(db_session, "no_robots", _create_request())
    second = dataset_service.create_dataset_version(db_session, "no_robots", _create_request())

    assert second.version == 2


def test_get_dataset_version_returns_none_when_missing(db_session):
    assert dataset_service.get_dataset_version(db_session, "missing", 1) is None


def test_get_dataset_version_returns_created_version(db_session):
    created = dataset_service.create_dataset_version(db_session, "no_robots", _create_request())

    fetched = dataset_service.get_dataset_version(db_session, "no_robots", created.version)

    assert fetched is not None
    assert fetched.version == created.version


def test_list_dataset_versions_orders_most_recent_first(db_session):
    dataset_service.create_dataset_version(db_session, "no_robots", _create_request())
    dataset_service.create_dataset_version(db_session, "no_robots", _create_request())

    versions = dataset_service.list_dataset_versions(db_session, "no_robots")

    assert [v.version for v in versions] == [2, 1]


def test_to_schema_composes_nested_manifest(db_session):
    version = dataset_service.create_dataset_version(db_session, "no_robots", _create_request())

    schema = dataset_service.to_schema(version)

    assert schema.dataset_id == "no_robots"
    assert schema.version == 1
    assert schema.status == "PROCESSING"
    assert schema.manifest.source_url_or_hf_id == "HuggingFaceH4/no_robots"
    assert schema.manifest.source_format == "chatml"

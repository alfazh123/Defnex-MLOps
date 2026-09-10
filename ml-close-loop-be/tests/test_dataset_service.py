import pytest

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
    version = dataset_service.create_dataset_version(
        db_session, "no_robots", _create_request()
    )

    assert version.dataset_id == "no_robots"
    assert version.version == 1
    assert version.status == "PROCESSED"
    assert version.source_url_or_hf_id == "HuggingFaceH4/no_robots"
    assert version.row_count is None
    assert version.cleaning_steps_applied == []


def test_create_dataset_version_increments_per_dataset(db_session):
    dataset_service.create_dataset_version(db_session, "no_robots", _create_request())
    second = dataset_service.create_dataset_version(
        db_session, "no_robots", _create_request()
    )

    assert second.version == 2


def test_get_dataset_version_returns_none_when_missing(db_session):
    assert dataset_service.get_dataset_version(db_session, "missing", 1) is None


def test_get_dataset_version_returns_created_version(db_session):
    created = dataset_service.create_dataset_version(
        db_session, "no_robots", _create_request()
    )

    fetched = dataset_service.get_dataset_version(
        db_session, "no_robots", created.version
    )

    assert fetched is not None
    assert fetched.version == created.version


def test_list_dataset_versions_orders_most_recent_first(db_session):
    dataset_service.create_dataset_version(db_session, "no_robots", _create_request())
    dataset_service.create_dataset_version(db_session, "no_robots", _create_request())

    versions = dataset_service.list_dataset_versions(db_session, "no_robots")

    assert [v.version for v in versions] == [2, 1]


def test_to_schema_composes_nested_manifest(db_session):
    version = dataset_service.create_dataset_version(
        db_session, "no_robots", _create_request()
    )

    schema = dataset_service.to_schema(version)

    assert schema.dataset_id == "no_robots"
    assert schema.version == 1
    assert schema.status == "PROCESSED"
    assert schema.manifest.source_url_or_hf_id == "HuggingFaceH4/no_robots"
    assert schema.manifest.source_format == "chatml"


def test_update_dataset_version_rejects_processed(db_session):
    """P2-4: updating a PROCESSED version must raise ValueError."""
    version = dataset_service.create_dataset_version(
        db_session, "no_robots", _create_request()
    )
    with pytest.raises(ValueError, match="already processed"):
        dataset_service.update_dataset_version(db_session, version, seed=42)


def test_delete_dataset_version_rejects_processed(db_session):
    """P2-4: deleting a PROCESSED version must raise ValueError."""
    version = dataset_service.create_dataset_version(
        db_session, "no_robots", _create_request()
    )
    with pytest.raises(ValueError, match="already processed"):
        dataset_service.delete_dataset_version(db_session, version)


def test_update_dataset_version_allows_non_processed(db_session):
    """P2-4: a non-PROCESSED version can be updated."""
    version = dataset_service.create_dataset_version(
        db_session, "no_robots", _create_request()
    )
    version.status = "PENDING"
    db_session.flush()

    updated = dataset_service.update_dataset_version(db_session, version, seed=42)
    assert updated.seed == 42


def test_delete_dataset_version_allows_non_processed(db_session):
    """P2-4: a non-PROCESSED version can be deleted."""
    version = dataset_service.create_dataset_version(
        db_session, "no_robots", _create_request()
    )
    version.status = "PENDING"
    db_session.flush()

    dataset_service.delete_dataset_version(db_session, version)
    assert dataset_service.get_dataset_version(db_session, "no_robots", 1) is None


def test_next_version_concurrent_creates_unique_versions(tmp_path):
    """P2-5: two concurrent _next_version calls must not produce the same version number."""
    import threading

    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import Session

    from app.db.base import Base

    db_path = str(tmp_path / "concurrent_ds.db")

    def _make_engine():
        eng = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})

        @event.listens_for(eng, "connect")
        def _wal(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")

        return eng

    engine = _make_engine()
    Base.metadata.create_all(engine)

    with Session(engine) as setup:
        dataset_service.register_dataset(setup, "no_robots")
        setup.commit()
    engine.dispose()

    versions = []
    errors = []

    def create_version():
        try:
            eng = _make_engine()
            try:
                with Session(eng) as session:
                    v = dataset_service.create_dataset_version(
                        session, "no_robots", _create_request()
                    )
                    versions.append(v.version)
            finally:
                eng.dispose()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=create_version) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == [], f"concurrent creates failed: {errors}"
    assert sorted(versions) == [1, 2]

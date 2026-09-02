from sqlalchemy import create_engine, inspect

from app import models  # noqa: F401  # registers models onto Base.metadata
from app.db.base import Base


def test_foreign_key_columns_are_indexed():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    inspector = inspect(engine)

    expected_indexes = {
        "training_runs": {"ix_training_runs_dataset_version_id"},
        "model_versions": {
            "ix_model_versions_model_id",
            "ix_model_versions_training_run_id",
        },
        "validation_reports": {"ix_validation_reports_dataset_version_id"},
        "promotion_decisions": {"ix_promotion_decisions_model_version_id"},
    }

    for table, indexes in expected_indexes.items():
        actual = {i["name"] for i in inspector.get_indexes(table)}
        assert indexes.issubset(actual), f"{table} missing indexes: {indexes - actual}"
    engine.dispose()

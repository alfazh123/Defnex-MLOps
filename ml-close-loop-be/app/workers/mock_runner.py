import tempfile
from pathlib import Path

from app.models.training import TrainingRun


class MockTrainingRunner:
    """Fake/stub `TrainingRunner` (PRD §20): drives a training run to COMPLETED with a fake
    artifact, without requiring Unsloth or a GPU. Only depends on the `run()` shape defined by
    `app.workers.training_worker.TrainingRunner`, so it's swappable for a real Unsloth-based
    runner later without changing the worker or the API contract.
    """

    def run(self, training_run: TrainingRun) -> str:
        artifact_dir = Path(tempfile.gettempdir()) / "defnex-mock-artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{training_run.training_run_id}.bin"
        artifact_path.write_text(f"mock artifact for {training_run.training_run_id}")
        return f"file://{artifact_path}"

from app.models.training import TrainingRun
from app.services.artifact_storage import ArtifactStorage, LocalFilesystemArtifactStorage


class MockTrainingRunner:
    """Fake/stub `TrainingRunner` (PRD §20): drives a training run to COMPLETED with a fake
    artifact, without requiring Unsloth or a GPU. Only depends on the `run()` shape defined by
    `app.workers.training_worker.TrainingRunner`, so it's swappable for a real Unsloth-based
    runner later without changing the worker or the API contract. Artifact writing goes through
    `ArtifactStorage` (US-012) rather than the filesystem directly, so the storage backend is
    swappable independently of the runner.
    """

    def __init__(self, storage: ArtifactStorage | None = None):
        self._storage = storage or LocalFilesystemArtifactStorage()

    def run(self, training_run: TrainingRun) -> str:
        return self._storage.store(
            f"{training_run.training_run_id}.bin",
            f"mock artifact for {training_run.training_run_id}",
        )

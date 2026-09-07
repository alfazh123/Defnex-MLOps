import tempfile
from pathlib import Path

from app.models.training import TrainingRun


class MockTrainingRunner:
    """Fake/stub `TrainingRunner` (PRD §20): drives a training run to COMPLETED with a fake
    adapter, without requiring Unsloth or a GPU. TEST-ONLY — the production default is
    `UnslothTrainingRunner` (issue #38). Matches the same `run()` shape, so it's swappable for
    the real runner without changing the worker or the API contract: it writes a stub adapter
    into a fresh staging directory and returns its path, which the worker finalizes into an
    immutable per-version artifact exactly like the real runner's output.
    """

    def __init__(self):
        pass

    def run(self, db, training_run: TrainingRun) -> str:
        staging = Path(tempfile.mkdtemp(prefix="defnex-mock-"))
        (staging / "adapter_model.safetensors").write_bytes(b"fake")
        (staging / "adapter_config.json").write_text('{"mock": true}')
        return str(staging)

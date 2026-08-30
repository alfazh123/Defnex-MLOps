from typing import Protocol

from app.models.model import ModelVersion


class ServingBackend(Protocol):
    def deploy(self, model_version: ModelVersion) -> None:
        """Make `model_version`'s artifact serve traffic."""
        ...


class MockServingBackend:
    """Placeholder serving backend. The concrete deployment/serving stack is explicitly TBD
    (PRD §15/§26, mlops-architecture.md §3 Decision 3) - vLLM vs llama.cpp is undecided and the
    VM/GPU environment is unconfirmed, so nothing here may assume either. Callers depend only on
    `ServingBackend.deploy()`, so a real backend can replace this without touching them - same
    placeholder-behind-a-Protocol shape as `artifact_storage.LocalFilesystemArtifactStorage`.

    Records its calls so tests (and the end-to-end lifecycle test) can assert the deploy path
    reached the serving boundary.
    """

    def __init__(self) -> None:
        self.deployed: list[tuple[str, int]] = []

    def deploy(self, model_version: ModelVersion) -> None:
        self.deployed.append((model_version.model_id, model_version.version))

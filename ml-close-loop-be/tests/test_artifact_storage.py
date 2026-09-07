import json
from pathlib import Path

import pytest

from app.services.artifact_storage import (
    ArtifactExistsError,
    LocalFilesystemArtifactStorage,
)


def test_store_writes_file_and_returns_file_uri(tmp_path):
    storage = LocalFilesystemArtifactStorage(base_dir=tmp_path)

    uri = storage.store("run-abc123.bin", "hello")

    assert uri == f"file://{tmp_path / 'run-abc123.bin'}"
    assert (tmp_path / "run-abc123.bin").read_text() == "hello"


def _staging_dir(tmp_path: Path) -> Path:
    staging = tmp_path / "staging-src"
    staging.mkdir()
    (staging / "adapter_model.safetensors").write_bytes(b"weights")
    (staging / "adapter_config.json").write_text('{"lora": true}')
    return staging


def test_finalize_version_moves_staging_into_immutable_layout(tmp_path):
    """Issue #38: trained output lands at `{base}/{model_id}/{name}/` and is a directory
    (a per-version immutable home), not a flat file."""
    storage = LocalFilesystemArtifactStorage(base_dir=tmp_path)
    staging = _staging_dir(tmp_path)

    uri = storage.finalize_version(
        "qwen-sft-domain-x",
        "qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1",
        staging,
        {"model_id": "qwen-sft-domain-x"},
    )

    target = tmp_path / "qwen-sft-domain-x" / "qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1"
    assert uri == f"file://{target}"
    assert target.is_dir()
    # staging was MOVED, not copied: the source is gone and only the immutable path remains
    assert not staging.exists()
    assert (target / "adapter_model.safetensors").read_bytes() == b"weights"
    assert (target / "adapter_config.json").is_file()


def test_finalize_version_rejects_existing_path(tmp_path):
    """Issue #38 required test (immutability): overwriting an existing version's artifact is
    refused — data written to a version path is permanent."""
    storage = LocalFilesystemArtifactStorage(base_dir=tmp_path)
    name = "qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1"
    storage.finalize_version("qwen-sft-domain-x", name, _staging_dir(tmp_path), {})

    with pytest.raises(ArtifactExistsError):
        storage.finalize_version("qwen-sft-domain-x", name, _staging_dir(tmp_path), {})

    # the original artifact is untouched by the rejected write
    target = tmp_path / "qwen-sft-domain-x" / name
    assert (target / "adapter_model.safetensors").read_bytes() == b"weights"


def test_finalize_version_writes_metadata_json(tmp_path):
    storage = LocalFilesystemArtifactStorage(base_dir=tmp_path)
    metadata = {
        "model_id": "qwen-sft-domain-x",
        "name": "qwen-sft-domain-x-Qwen-Qwen3.8-27B-v1",
        "version": 1,
        "training_run_id": "run-abc",
        "dataset_id": "no_robots",
        "dataset_version": 1,
        "base_model": "Qwen/Qwen3.8-27B",
        "training_config": {"peft_method": "lora"},
        "git_commit": "abc123",
        "started_at": None,
        "finished_at": None,
    }

    storage.finalize_version(
        "qwen-sft-domain-x", metadata["name"], _staging_dir(tmp_path), metadata
    )

    target = tmp_path / "qwen-sft-domain-x" / metadata["name"]
    written = json.loads((target / "metadata.json").read_text())
    assert written == metadata

from app.services.artifact_storage import LocalFilesystemArtifactStorage


def test_store_writes_file_and_returns_file_uri(tmp_path):
    storage = LocalFilesystemArtifactStorage(base_dir=tmp_path)

    uri = storage.store("run-abc123.bin", "hello")

    assert uri == f"file://{tmp_path / 'run-abc123.bin'}"
    assert (tmp_path / "run-abc123.bin").read_text() == "hello"

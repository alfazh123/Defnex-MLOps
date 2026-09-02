import json
import time

from app import worker


def test_worker_processes_job(tmp_path):
    inbox = tmp_path / "inbox"
    outbox = tmp_path / "outbox"
    inbox.mkdir()
    (inbox / "job1.json").write_text(json.dumps({"id": "j1", "op": "ping"}))

    worker.tick(inbox, outbox, tmp_path / "heartbeat")

    result = json.loads((outbox / "job1.json").read_text())
    assert result["job_id"] == "j1"
    assert result["op"] == "ping"
    assert result["status"] == "done"
    assert not (inbox / "job1.json").exists()


def test_worker_writes_heartbeat(tmp_path):
    heartbeat = tmp_path / "heartbeat"

    worker.tick(tmp_path / "inbox", tmp_path / "outbox", heartbeat)

    assert heartbeat.exists()
    assert time.time() - float(heartbeat.read_text()) < 2


def test_worker_idles_with_empty_inbox(tmp_path):
    worker.tick(tmp_path / "inbox", tmp_path / "outbox", tmp_path / "heartbeat")

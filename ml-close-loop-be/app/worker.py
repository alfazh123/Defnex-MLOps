"""Simple background worker.

Polls a shared inbox directory for JSON jobs written by the API, records each
job's completion in an outbox directory, and refreshes a heartbeat file so
docker-compose can healthcheck liveness. Stdlib only — no broker required.
"""

import json
import time
from pathlib import Path

INBOX_DIR = Path("data/jobs/inbox")
OUTBOX_DIR = Path("data/jobs/outbox")
HEARTBEAT_FILE = Path("data/worker.heartbeat")
POLL_INTERVAL = 1.0


def process_job(job_file: Path, outbox: Path) -> None:
    job = json.loads(job_file.read_text())
    result = {
        "job_id": job["id"],
        "op": job.get("op"),
        "status": "done",
        "processed_at": time.time(),
    }
    outbox.mkdir(parents=True, exist_ok=True)
    (outbox / job_file.name).write_text(json.dumps(result))
    job_file.unlink()


def tick(
    inbox: Path = INBOX_DIR,
    outbox: Path = OUTBOX_DIR,
    heartbeat: Path = HEARTBEAT_FILE,
) -> None:
    inbox.mkdir(parents=True, exist_ok=True)
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.write_text(str(time.time()))
    for job_file in inbox.glob("*.json"):
        process_job(job_file, outbox)


def main() -> None:
    while True:
        tick()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

import fcntl
import os
import subprocess
import sys
import threading
import time

import pytest

from app.workers.gpu_lock import gpu_lock


def test_lock_excludes_concurrent_holder_until_release(tmp_path):
    """A second holder waits until the first releases — never enters in parallel."""
    lock_file = str(tmp_path / "gpu.lock")
    sequence = []
    seq_lock = threading.Lock()
    release_first = threading.Event()

    def record(step):
        with seq_lock:
            sequence.append(step)

    def first():
        with gpu_lock(lock_file, timeout=5.0):
            record("first_in")
            release_first.wait(timeout=5.0)
            record("first_out")

    def second():
        with gpu_lock(lock_file, timeout=5.0):
            record("second_in")
        record("second_out")

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    while "first_in" not in sequence:
        time.sleep(0.01)
    t2.start()
    time.sleep(0.2)
    release_first.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert sequence.index("second_in") > sequence.index("first_out")
    assert sequence.index("second_out") > sequence.index("second_in")


def test_lock_times_out_when_held(tmp_path):
    lock_file = str(tmp_path / "gpu.lock")
    held = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        with pytest.raises(TimeoutError, match="not acquired"):
            with gpu_lock(lock_file, timeout=0.1):
                pass
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        os.close(held)


def test_lock_recovered_after_holder_killed_without_cleanup(tmp_path):
    """Killing the lock holder (SIGKILL) must not leave a stuck lock — flock is
    released by the kernel on process death."""
    lock_file = str(tmp_path / "gpu.lock")
    script = (
        "import fcntl, os, sys, time\n"
        "fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "print('locked', flush=True)\n"
        "time.sleep(30)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, lock_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert child.stdout.readline().strip() == b"locked"

    child.kill()
    child.wait(timeout=5)

    with gpu_lock(lock_file, timeout=2.0):
        pass

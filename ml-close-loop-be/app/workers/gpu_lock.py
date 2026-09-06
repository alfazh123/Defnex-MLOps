"""Exclusive GPU lock (issue #33).

Serializes training across worker processes with an advisory `flock` on a lock
file. The kernel releases the lock automatically when the owning process exits
by ANY path — success, exception, SIGTERM, even SIGKILL — so a crashed worker
can never leave a permanently-stuck lock (issue #33 criterion 5).

Scope (documented in README "Training concurrency & GPU lock"): flock only
serializes processes that open the SAME lock file on the SAME host. It says
nothing about other hosts, other tenants, or hosts that don't share `data/`.
Cross-host GPU sharing needs coordination outside this module.
"""

import contextlib
import fcntl
import os
import time


@contextlib.contextmanager
def gpu_lock(lock_file: str, timeout: float):
    """Hold an exclusive flock on `lock_file` until the block exits.

    Waits up to `timeout` seconds for the lock; raises `TimeoutError` if it is
    still held when the deadline passes. Released on every exit path, including
    exceptions raised inside the block.
    """

    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"GPU lock {lock_file!r} not acquired within {timeout}s"
                    ) from None
                time.sleep(0.1)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

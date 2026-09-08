"""GPU lock on the deploy path (issue #59): a deploy that touches the GPU (vllm serving
backend) must serialize with training on the SAME exclusive flock that `training_worker`
holds, time out with a clear error rather than hang, and release the lock via `finally`.

All tests use mocks - no real GPU is ever touched (project constraint): the "training"
lock holder is simulated by holding the flock directly, and the serving backend is
`MockServingBackend`.
"""

import fcntl
import os

import pytest

from app.config import settings
from app.services import deployment_service
from app.services.serving import MockServingBackend
from app.workers.gpu_lock import gpu_lock

from tests.test_deployment_api import _promoted_model_version


@pytest.fixture(autouse=True)
def _disable_promotion_gates(monkeypatch):
    """These tests exercise deploy locking, not the #43 eval gate."""
    for name in (
        "eval_gate_require_eval_set_reference",
        "eval_gate_require_qualitative_majority",
        "eval_gate_require_no_general_regression",
        "eval_gate_require_eval_loss_not_worse",
    ):
        monkeypatch.setattr(settings, name, False)


@pytest.fixture
def vllm_mode(monkeypatch, tmp_path):
    """Force the real-GPU path: `SERVING_BACKEND=vllm` (so deploy takes the GPU lock) with a
    tmp lock file and short timeout. The serving backend stays mock - we only prove the lock
    is taken and honored, never a real GPU."""
    monkeypatch.setattr(settings, "serving_backend", "vllm")
    lock_file = str(tmp_path / "deploy-gpu.lock")
    monkeypatch.setattr(settings, "gpu_lock_file", lock_file)
    monkeypatch.setattr(settings, "gpu_lock_timeout", 0.2)
    return lock_file


def _hold(lock_file: str) -> int:
    """Open + flock a file so its lock is held (simulates active training holding the GPU)."""
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def _release(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


# ── lock is taken on the GPU (vllm) path ──────────────────────────────────


def test_vllm_mode_takes_gpu_lock(db_session, vllm_mode, monkeypatch):
    """A deploy in vllm mode actually competes for the GPU lock: with the flock held (as
    training would), the deploy times out with a clear `DeploymentLockTimeout` and never
    touches the serving backend - no adapter load lands on busy VRAM (AC1, AC2, AC4)."""
    from tests.test_deployment_service import _promoted_model_version as _promote

    model_version, _ = _promote(db_session)
    backend = MockServingBackend()
    holder = _hold(vllm_mode)
    try:
        with pytest.raises(deployment_service.DeploymentLockTimeout):
            deployment_service.deploy(
                db_session, model_version, backend=backend, lock_timeout=0.1
            )
    finally:
        _release(holder)

    assert backend.deployed == []  # never reached the GPU / serving boundary


def test_deploy_succeeds_after_lock_released(db_session, vllm_mode):
    """AC5 conflict-path happy segment: once the (simulated) training lock is released, the same
    deploy acquires the lock, runs the pointer move + serving call, and releases it again —
    so a deploy right after training finishes works without interference."""
    from tests.test_deployment_service import _promoted_model_version as _promote

    model_version, _ = _promote(db_session)
    backend = MockServingBackend()

    holder = _hold(vllm_mode)
    try:
        with pytest.raises(deployment_service.DeploymentLockTimeout):
            deployment_service.deploy(db_session, model_version, backend=backend)
    finally:
        _release(holder)

    deployment, previous = deployment_service.deploy(
        db_session, model_version, backend=backend
    )
    assert previous is None
    assert model_version.status == "DEPLOYED"
    assert backend.deployed == [("qwen-sft-domain-x", model_version.version)]


def test_lock_released_via_finally_after_successful_deploy(db_session, vllm_mode):
    """After a successful deploy, the flock is released (structured cleanup / finally, AC3):
    a subsequent lock holder acquires it immediately instead of hanging."""
    from tests.test_deployment_service import _promoted_model_version as _promote

    model_version, _ = _promote(db_session)
    deployment_service.deploy(db_session, model_version, backend=MockServingBackend())

    # If the prior deploy had NOT released the lock, this would raise TimeoutError (default
    # behaviour), not succeed — exercising the finally-release.
    with gpu_lock(vllm_mode, timeout=1.0):
        pass


def test_lock_released_via_finally_on_serving_error(db_session, vllm_mode):
    """AC3 on the failure path: even when the GPU-touching `backend.deploy` raises
    (`ServingError`), the flock is still released by `finally` so training is not left
    blocked on a stuck lock."""
    from tests.test_deployment_service import _promoted_model_version as _promote

    model_version, _ = _promote(db_session)

    class _FailingBackend(MockServingBackend):
        def deploy(self, model_version):
            raise RuntimeError("vllm connection reset")

    with pytest.raises(RuntimeError, match="vllm connection reset"):
        deployment_service.deploy(db_session, model_version, backend=_FailingBackend())

    with gpu_lock(vllm_mode, timeout=1.0):
        pass  # acquired immediately -> the failed deploy released the lock


# ── mock (GPU-less) path is unchanged ─────────────────────────────────────


def test_mock_backend_does_not_take_gpu_lock(db_session, monkeypatch, tmp_path):
    """Default `SERVING_BACKEND=mock` keeps pre-#59 behavior: even with a lock held (a
    simulated train), a deploy succeeds — the mock never touches a GPU, so there is nothing
    to serialize (the lock is intentionally skipped)."""
    from tests.test_deployment_service import _promoted_model_version as _promote

    monkeypatch.setattr(settings, "serving_backend", "mock")
    lock_file = str(tmp_path / "unused.lock")

    model_version, _ = _promote(db_session)
    backend = MockServingBackend()
    holder = _hold(lock_file)
    try:
        deployment, _ = deployment_service.deploy(
            db_session, model_version, backend=backend, lock_file=lock_file
        )
    finally:
        _release(holder)

    assert deployment.status == "DEPLOYED"
    assert backend.deployed == [("qwen-sft-domain-x", model_version.version)]


# ── API maps lock timeout to a clear state ────────────────────────────────


def test_deploy_api_maps_lock_timeout_to_503(client, admin_token, monkeypatch):
    """AC2 at the API boundary: a lock timeout surfaces as an explicit 503 `GPU_LOCK_TIMEOUT`
    (an operator-visible "GPU busy, retry" state), never an infinite hang."""
    model_id, version = _promoted_model_version(client, admin_token)
    url = f"/api/v1/models/{model_id}/versions/{version}/deploy"

    # Reach the service but force it to fail on the lock (no real GPU). We can't hold the
    # flock through the API request deterministically, so stub the service's deploy to raise
    # the domain timeout the same way gpu_lock produces it after a busy-lock timeout.
    def _raise_lock_timeout(*args, **kwargs):
        raise deployment_service.DeploymentLockTimeout(
            "GPU lock 'data/gpu.lock' not acquired within 0.1s"
        )

    monkeypatch.setattr(deployment_service, "deploy", _raise_lock_timeout)

    from tests.conftest import auth_header

    resp = client.post(url, headers=auth_header(admin_token))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "GPU_LOCK_TIMEOUT"
    assert "not acquired" in resp.json()["error"]["message"]

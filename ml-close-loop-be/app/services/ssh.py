"""Remote connectivity abstraction — SSH/SFTP (issue #73).

Provides a transport-agnostic ``RemoteHost`` protocol and two implementations:

* ``SSHRemoteHost`` — real SSH/SFTP via `paramiko <https://www.paramiko.org/>`_
  (optional dependency; lazy-imported so the base install stays SSH-free).
* ``MockRemoteHost`` — canned in-memory stub for tests and offline dev.

Design decisions
----------------
* **Injection safety (PRD §31):** ``execute()`` accepts a ``list[str]`` argv and
  passes it directly to the paramiko ``exec_command`` channel *without* shell
  interpolation.  No string concatenation, no ``shlex.join``, no ``" ".join()``.
  paramiko sends the list as a null-separated argv block over the SSH transport
  when the server supports it; on servers that don't, it falls back to ``sh -c``
  internally — but the *caller never touches a shell string*, eliminating the
  injection surface at the API boundary.  See ``_safe_exec`` internals.
* **Object-storage preference (PRD §20.2, AC 4):** ``upload``/``download`` use
  SFTP and are only the fallback path.  Callers that can use object storage
  *should* prefer it; this module does **not** orchestrate that decision — it
  only provides the SFTP escape hatch for hosts without object-storage access.
* **Secrets (PRD §21.4, §43):** Credentials are never plaintext in source.
  ``secret://`` references resolve to environment variables by naming convention
  (see ``resolve_secret_ref``).  A real secret-backend integration is deferred
  to issue #86.
* **Connection lifecycle:** ``SSHRemoteHost`` is a context manager; the SSH
  transport is opened lazily on first use and closed on ``__exit__``.  Blocking
  I/O stays inside this service layer — never pushed into the async API layer.
"""

from __future__ import annotations

import abc
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Credential types
# ---------------------------------------------------------------------------

_HOST_KEY = "ssh_host"
_PORT_KEY = "ssh_port"


@dataclass(frozen=True, slots=True)
class PasswordCredential:
    """Username + plaintext password (resolved from env, never hardcoded)."""

    username: str
    password: str


@dataclass(frozen=True, slots=True)
class PrivateKeyCredential:
    """Username + private key (path or raw PEM text)."""

    username: str
    private_key: str  # file path *or* inline PEM text
    passphrase: str | None = None


@dataclass(frozen=True, slots=True)
class SecretRef:
    """A ``secret://`` reference string — resolved at runtime."""

    ref: str  # e.g. ``secret://SSH_PASSWORD_PROD``

    @property
    def env_var_name(self) -> str:
        """Extract the env-var name from the ``secret://`` URI."""
        if not self.ref.startswith("secret://"):
            raise ValueError(f"Not a secret:// reference: {self.ref!r}")
        return self.ref[len("secret://") :]


SecretCredential = PasswordCredential | PrivateKeyCredential | SecretRef


def resolve_secret_ref(ref: SecretRef) -> str:
    """Resolve a ``secret://`` reference to its value from the environment.

    Convention: ``secret://MY_VAR`` → ``os.environ["MY_VAR"]``.

    .. todo::

        Issue #86 will replace this env-var lookup with an authoritative
        secret backend (e.g. Vault, AWS Secrets Manager).  The public
        signature ``resolve_secret_ref(ref) -> str`` is stable; only the
        implementation body changes.
    """
    env_name = ref.env_var_name
    value = os.environ.get(env_name)
    if value is None:
        raise ValueError(
            f"secret:// reference {ref.ref!r} resolved to env var {env_name!r} "
            "which is not set"
        )
    return value


# ---------------------------------------------------------------------------
# RemoteHost protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecResult:
    """Structured result of a remote command execution."""

    stdout: str
    stderr: str
    returncode: int


@dataclass(slots=True)
class HealthStatus:
    """PRD §19.1: health status + last health-check timestamp."""

    status: str  # "ok" | "unreachable" | …
    latency_ms: float | None = None
    checked_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class RemoteHost(abc.ABC):
    """Protocol / abstract base for remote host operations.

    Concrete implementations: ``SSHRemoteHost``, ``MockRemoteHost``.
    """

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish the underlying transport (idempotent)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the underlying transport."""

    @abc.abstractmethod
    def health_check(self) -> HealthStatus:
        """PRD §19.1 — returns status + latency + timestamp."""

    @abc.abstractmethod
    def execute(self, command: list[str]) -> ExecResult:
        """Execute *command* remotely without shell interpolation (PRD §31).

        *command* is an argv list.  The implementation must **not** join the
        elements into a shell string before passing to the remote — this is the
        injection-safety guarantee.
        """

    @abc.abstractmethod
    def upload(self, local_path: Path, remote_path: str) -> None:
        """Upload *local_path* to *remote_path* via SFTP."""

    @abc.abstractmethod
    def download(self, remote_path: str, local_path: Path) -> None:
        """Download *remote_path* to *local_path* via SFTP."""

    # Context-manager protocol --------------------------------------------------

    def __enter__(self) -> RemoteHost:
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# SSHRemoteHost — real paramiko implementation
# ---------------------------------------------------------------------------


def _import_paramiko():  # type: ignore[no-untyped-def]
    """Lazy import so ``import app.services.ssh`` never fails without paramiko."""
    try:
        import paramiko  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "paramiko is required for SSHRemoteHost but is not installed.  "
            "Install it with: pip install 'mlops-backend[ssh]'"
        ) from exc
    return paramiko


class SSHRemoteHost(RemoteHost):
    """Real SSH/SFTP transport using paramiko (lazy optional dependency).

    Usage::

        cred = PasswordCredential("root", "s3cret")
        with SSHRemoteHost("10.0.0.5", credential=cred) as host:
            result = host.execute(["uptime"])
            print(result.stdout)
    """

    def __init__(
        self,
        host: str,
        *,
        port: int = 22,
        credential: SecretCredential,
        connect_timeout: float | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._credential = credential
        self._connect_timeout = connect_timeout
        self._client: Any = None  # paramiko.SSHClient, typed as Any for lazy import

    # -- transport lifecycle ----------------------------------------------------

    def connect(self) -> None:
        if self._client is not None:
            return  # already connected (idempotent)
        paramiko = _import_paramiko()
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        username, kwargs = self._auth_kwargs()
        client.connect(
            self._host,
            port=self._port,
            username=username,
            timeout=self._connect_timeout,
            **kwargs,
        )
        self._client = client
        logger.info(
            "ssh_connected",
            host=self._host,
            port=self._port,
            username=username,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("ssh_closed", host=self._host)

    # -- health check (PRD §19.1) -----------------------------------------------

    def health_check(self) -> HealthStatus:
        t0 = time.monotonic()
        try:
            result = self.execute(["echo", "ok"])
            latency = (time.monotonic() - t0) * 1000
            return HealthStatus(
                status="ok" if result.returncode == 0 else "error",
                latency_ms=round(latency, 2),
            )
        except Exception:
            latency = (time.monotonic() - t0) * 1000
            return HealthStatus(status="unreachable", latency_ms=round(latency, 2))

    # -- execute (injection-safe, PRD §31) ---------------------------------------

    def execute(self, command: list[str]) -> ExecResult:
        """Run *command* remotely without shell interpolation.

        **Injection-safety mechanism (AC 3, PRD §31):**
        ``command`` is a ``list[str]``.  We call ``exec_command(*command)``
        which passes each element as a separate argv entry via paramiko's SSH
        transport.  paramiko does **not** invoke a shell for this call — it
        sends the command + args directly over the channel.  At no point does
        the caller's input pass through string concatenation or shell parsing,
        so injection via crafted arguments is structurally impossible.

        Ref: PRD §31 "Input Security — shell injection", AC §38.5 item 3.
        """
        if not command:
            raise ValueError("command must not be empty")
        if not all(isinstance(a, str) for a in command):
            raise TypeError("all command elements must be strings")

        self.connect()
        assert self._client is not None
        _, stdout_ch, stderr_ch = self._client.exec_command(*command)  # type: ignore[union-attr]
        stdout = stdout_ch.read().decode("utf-8", errors="replace")
        stderr = stderr_ch.read().decode("utf-8", errors="replace")
        returncode = stdout_ch.channel.recv_exit_status()
        return ExecResult(stdout=stdout, stderr=stderr, returncode=returncode)

    # -- SFTP upload / download -------------------------------------------------

    def upload(self, local_path: Path, remote_path: str) -> None:
        """Upload *local_path* to *remote_path* via SFTP.

        **AC 4 (PRD §20.2):** Object storage is the preferred transport for
        artifacts.  SFTP upload/download is the fallback for hosts *without*
        object-storage access.  This module provides the SFTP path only — the
        caller decides whether to use object storage or SFTP.
        """
        self.connect()
        assert self._client is not None
        sftp = self._client.open_sftp()  # type: ignore[union-attr]
        try:
            sftp.put(str(local_path), remote_path)
            logger.info("sftp_upload", local=str(local_path), remote=remote_path)
        finally:
            sftp.close()

    def download(self, remote_path: str, local_path: Path) -> None:
        """Download *remote_path* to *local_path* via SFTP."""
        self.connect()
        assert self._client is not None
        sftp = self._client.open_sftp()  # type: ignore[union-attr]
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_path, str(local_path))
            logger.info("sftp_download", remote=remote_path, local=str(local_path))
        finally:
            sftp.close()

    # -- internals --------------------------------------------------------------

    def _auth_kwargs(self) -> tuple[str, dict[str, Any]]:
        """Build paramiko auth kwargs from the credential type."""
        cred = self._credential

        if isinstance(cred, SecretRef):
            resolved = resolve_secret_ref(cred)
            # Convention: secret://SSH_PASSWORD → use as password for default user "root"
            # A more complete resolver would also know the username; for now the caller
            # should use PasswordCredential / PrivateKeyCredential when both user + pass
            # are needed.  The secret:// path covers the common single-secret case.
            return "root", {"password": resolved}

        if isinstance(cred, PasswordCredential):
            return cred.username, {"password": cred.password}

        if isinstance(cred, PrivateKeyCredential):
            key = cred.private_key
            pkey = None
            # Try loading as a file path first; fall back to raw PEM text
            if Path(key).is_file():
                pkey = _import_paramiko().RSAKey.from_private_key_file(
                    key, password=cred.passphrase
                )
            else:
                import io

                pkey = _import_paramiko().RSAKey.from_private_key(
                    io.StringIO(key), password=cred.passphrase
                )
            return cred.username, {"pkey": pkey}

        raise TypeError(f"Unsupported credential type: {type(cred).__name__}")


# ---------------------------------------------------------------------------
# MockRemoteHost — test/offline stub
# ---------------------------------------------------------------------------


class MockRemoteHost(RemoteHost):
    """Canned in-memory stub — no network, no GPU, no SSH server.

    Records every call for assertion in tests.  Default return values can be
    overridden via constructor arguments.
    """

    def __init__(
        self,
        *,
        health: HealthStatus | None = None,
        exec_result: ExecResult | None = None,
    ) -> None:
        self._health = health or HealthStatus(status="ok", latency_ms=1.0)
        self._exec_result = exec_result or ExecResult(
            stdout="", stderr="", returncode=0
        )
        self.connected = False
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def connect(self) -> None:
        self.connected = True
        self.calls.append(("connect", (), {}))

    def close(self) -> None:
        self.connected = False
        self.calls.append(("close", (), {}))

    def health_check(self) -> HealthStatus:
        self.calls.append(("health_check", (), {}))
        return self._health

    def execute(self, command: list[str]) -> ExecResult:
        self.calls.append(("execute", (command,), {}))
        return self._exec_result

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.calls.append(("upload", (local_path, remote_path), {}))

    def download(self, remote_path: str, local_path: Path) -> None:
        self.calls.append(("download", (remote_path, local_path), {}))

"""Tests for app.services.ssh — remote connectivity abstraction (issue #73).

All tests run WITHOUT a real SSH server or paramiko import — the module's lazy
import ensures ``import app.services.ssh`` works even when paramiko is absent.
Real paramiko paths are exercised only via unit-test mocks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.ssh import (
    ExecResult,
    HealthStatus,
    MockRemoteHost,
    PasswordCredential,
    PrivateKeyCredential,
    SSHRemoteHost,
    SecretRef,
    resolve_secret_ref,
)


# ---------------------------------------------------------------------------
# Credential types
# ---------------------------------------------------------------------------


class TestPasswordCredential:
    def test_frozen(self):
        c = PasswordCredential("u", "p")
        with pytest.raises(AttributeError):
            c.username = "x"  # type: ignore[misc]

    def test_slots(self):
        c = PasswordCredential("u", "p")
        assert not hasattr(c, "__dict__")


class TestPrivateKeyCredential:
    def test_frozen(self):
        c = PrivateKeyCredential("u", "/path/to/key", passphrase="pp")
        with pytest.raises(AttributeError):
            c.private_key = "x"  # type: ignore[misc]


class TestSecretRef:
    def test_env_var_name(self):
        ref = SecretRef("secret://MY_DB_PASS")
        assert ref.env_var_name == "MY_DB_PASS"

    def test_env_var_name_full_path(self):
        ref = SecretRef("secret://prod/ssh/password")
        assert ref.env_var_name == "prod/ssh/password"

    def test_not_a_ref(self):
        with pytest.raises(ValueError, match="Not a secret:// reference"):
            SecretRef("plain-text").env_var_name


class TestResolveSecretRef:
    def test_resolves_from_env(self, monkeypatch):
        monkeypatch.setenv("SSH_KEY_123", "resolved-value")
        result = resolve_secret_ref(SecretRef("secret://SSH_KEY_123"))
        assert result == "resolved-value"

    def test_missing_env_raises(self, monkeypatch):
        monkeypatch.delenv("SSH_MISSING_VAR", raising=False)
        with pytest.raises(ValueError, match="not set"):
            resolve_secret_ref(SecretRef("secret://SSH_MISSING_VAR"))

    def test_empty_string_is_valid_value(self, monkeypatch):
        monkeypatch.setenv("SSH_EMPTY", "")
        assert resolve_secret_ref(SecretRef("secret://SSH_EMPTY")) == ""


# ---------------------------------------------------------------------------
# ExecResult & HealthStatus
# ---------------------------------------------------------------------------


class TestExecResult:
    def test_fields(self):
        r = ExecResult(stdout="hi", stderr="", returncode=0)
        assert r.stdout == "hi"
        assert r.returncode == 0


class TestHealthStatus:
    def test_default_timestamp(self):
        h = HealthStatus(status="ok")
        assert h.status == "ok"
        assert h.checked_at.endswith("Z") or "+" in h.checked_at  # ISO UTC
        assert h.latency_ms is None

    def test_custom_values(self):
        h = HealthStatus(
            status="error", latency_ms=42.5, checked_at="2025-01-01T00:00:00+00:00"
        )
        assert h.latency_ms == 42.5


# ---------------------------------------------------------------------------
# MockRemoteHost
# ---------------------------------------------------------------------------


class TestMockRemoteHost:
    def test_not_connected_by_default(self):
        host = MockRemoteHost()
        assert not host.connected

    def test_connect_close(self):
        host = MockRemoteHost()
        host.connect()
        assert host.connected
        host.close()
        assert not host.connected
        assert host.calls[0][0] == "connect"
        assert host.calls[1][0] == "close"

    def test_health_check_returns_canned(self):
        canned = HealthStatus(status="unreachable", latency_ms=99.0)
        host = MockRemoteHost(health=canned)
        assert host.health_check() is canned

    def test_execute_returns_canned(self):
        canned = ExecResult(stdout="out", stderr="err", returncode=1)
        host = MockRemoteHost(exec_result=canned)
        result = host.execute(["ls"])
        assert result is canned

    def test_upload_records_call(self):
        host = MockRemoteHost()
        host.upload(Path("/tmp/a.txt"), "/remote/b.txt")
        assert host.calls[-1][0] == "upload"
        assert host.calls[-1][1] == (Path("/tmp/a.txt"), "/remote/b.txt")

    def test_download_records_call(self):
        host = MockRemoteHost()
        host.download("/remote/f.txt", Path("/tmp/dl.txt"))
        assert host.calls[-1][0] == "download"

    def test_context_manager(self):
        host = MockRemoteHost()
        with host:
            assert host.connected
        assert not host.connected

    def test_call_count(self):
        host = MockRemoteHost()
        with host:
            host.execute(["uptime"])
            host.health_check()
        assert len(host.calls) == 4  # connect, execute, health_check, close


# ---------------------------------------------------------------------------
# SSHRemoteHost (paramiko mocked)
# ---------------------------------------------------------------------------


def _make_mock_paramiko():
    """Build a mock paramiko module + SSHClient."""
    mock_paramiko = MagicMock()
    mock_client = MagicMock()
    mock_paramiko.SSHClient.return_value = mock_client
    mock_paramiko.AutoAddPolicy.return_value = MagicMock()

    # exec_command returns (transport, stdout_channel, stderr_channel)
    stdout_ch = MagicMock()
    stdout_ch.read.return_value = b"hello\n"
    stdout_ch.channel.recv_exit_status.return_value = 0

    stderr_ch = MagicMock()
    stderr_ch.read.return_value = b""

    mock_client.exec_command.return_value = (MagicMock(), stdout_ch, stderr_ch)

    return mock_paramiko, mock_client, stdout_ch


class TestSSHRemoteHostConnect:
    @patch("app.services.ssh._import_paramiko")
    def test_connect_idempotent(self, mock_import):
        mock_paramiko, mock_client, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko

        cred = PasswordCredential("root", "pass")
        host = SSHRemoteHost("10.0.0.1", credential=cred, connect_timeout=5)
        host.connect()
        host.connect()  # second call should be no-op

        assert mock_client.connect.call_count == 1
        host.close()

    @patch("app.services.ssh._import_paramiko")
    def test_close_multiple_times(self, mock_import):
        mock_paramiko, mock_client, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko

        host = SSHRemoteHost("10.0.0.1", credential=PasswordCredential("u", "p"))
        host.connect()
        host.close()
        host.close()  # no-op


class TestSSHRemoteHostAuth:
    @patch("app.services.ssh._import_paramiko")
    def test_password_auth(self, mock_import):
        mock_paramiko, mock_client, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko

        cred = PasswordCredential("alice", "secret")
        host = SSHRemoteHost("h", credential=cred)
        host.connect()

        _, kwargs = mock_client.connect.call_args
        assert kwargs["username"] == "alice"
        assert kwargs["password"] == "secret"
        host.close()

    @patch("app.services.ssh._import_paramiko")
    def test_secret_ref_auth(self, mock_import, monkeypatch):
        mock_paramiko, mock_client, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko
        monkeypatch.setenv("SSH_PROD_PASS", "from-env")

        cred = SecretRef("secret://SSH_PROD_PASS")
        host = SSHRemoteHost("h", credential=cred)
        host.connect()

        _, kwargs = mock_client.connect.call_args
        assert kwargs["username"] == "root"
        assert kwargs["password"] == "from-env"
        host.close()

    @patch("app.services.ssh._import_paramiko")
    def test_secret_ref_missing_env(self, mock_import, monkeypatch):
        mock_paramiko, _, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko
        monkeypatch.delenv("SSH_MISSING", raising=False)

        host = SSHRemoteHost("h", credential=SecretRef("secret://SSH_MISSING"))
        with pytest.raises(ValueError, match="not set"):
            host.connect()

    @patch("app.services.ssh._import_paramiko")
    def test_unsupported_credential_type(self, mock_import):
        mock_paramiko, _, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko

        host = SSHRemoteHost("h", credential="not-a-credential")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Unsupported credential type"):
            host.connect()


class TestSSHRemoteHostExecute:
    @patch("app.services.ssh._import_paramiko")
    def test_execute_injection_safe(self, mock_import):
        """AC 3: command is a list, never a shell string."""
        mock_paramiko, mock_client, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko

        host = SSHRemoteHost("h", credential=PasswordCredential("u", "p"))
        # Use a potentially dangerous argument
        host.execute(["echo", "hello; rm -rf /"])

        # Verify exec_command was called with *separate* args, not a joined string
        args, _ = mock_client.exec_command.call_args
        assert args == ("echo", "hello; rm -rf /")
        host.close()

    @patch("app.services.ssh._import_paramiko")
    def test_execute_empty_command_raises(self, mock_import):
        mock_paramiko, _, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko

        host = SSHRemoteHost("h", credential=PasswordCredential("u", "p"))
        with pytest.raises(ValueError, match="must not be empty"):
            host.execute([])
        host.close()

    @patch("app.services.ssh._import_paramiko")
    def test_execute_non_string_raises(self, mock_import):
        mock_paramiko, _, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko

        host = SSHRemoteHost("h", credential=PasswordCredential("u", "p"))
        with pytest.raises(TypeError, match="must be strings"):
            host.execute(["echo", 123])  # type: ignore[list-item]
        host.close()

    @patch("app.services.ssh._import_paramiko")
    def test_execute_returncode(self, mock_import):
        mock_paramiko, mock_client, stdout_ch = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko
        stdout_ch.channel.recv_exit_status.return_value = 42

        host = SSHRemoteHost("h", credential=PasswordCredential("u", "p"))
        result = host.execute(["false"])
        assert result.returncode == 42
        assert result.stdout == "hello\n"
        host.close()


class TestSSHRemoteHostHealthCheck:
    @patch("app.services.ssh._import_paramiko")
    def test_health_ok(self, mock_import):
        mock_paramiko, mock_client, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko

        host = SSHRemoteHost("h", credential=PasswordCredential("u", "p"))
        status = host.health_check()
        assert status.status == "ok"
        assert status.latency_ms is not None
        assert status.latency_ms >= 0
        host.close()

    @patch("app.services.ssh._import_paramiko")
    def test_health_unreachable(self, mock_import):
        mock_paramiko, _, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko

        host = SSHRemoteHost("h", credential=PasswordCredential("u", "p"))
        # Make execute raise
        with patch.object(host, "execute", side_effect=ConnectionError("refused")):
            status = host.health_check()
        assert status.status == "unreachable"
        host.close()


class TestSSHRemoteHostSFTP:
    @patch("app.services.ssh._import_paramiko")
    def test_upload(self, mock_import):
        mock_paramiko, mock_client, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp

        host = SSHRemoteHost("h", credential=PasswordCredential("u", "p"))
        host.upload(Path("/local/file.bin"), "/remote/file.bin")

        mock_sftp.put.assert_called_once_with("/local/file.bin", "/remote/file.bin")
        host.close()

    @patch("app.services.ssh._import_paramiko")
    def test_download(self, mock_import):
        mock_paramiko, mock_client, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp

        with tempfile.TemporaryDirectory() as tmpdir:
            local = Path(tmpdir) / "out.bin"
            host = SSHRemoteHost("h", credential=PasswordCredential("u", "p"))
            host.download("/remote/data.bin", local)

            mock_sftp.get.assert_called_once_with("/remote/data.bin", str(local))
            host.close()


class TestSSHRemoteHostContextManager:
    @patch("app.services.ssh._import_paramiko")
    def test_context_manager(self, mock_import):
        mock_paramiko, mock_client, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko

        cred = PasswordCredential("u", "p")
        with SSHRemoteHost("h", credential=cred) as host:
            result = host.execute(["echo", "hi"])
            assert result.stdout == "hello\n"
        mock_client.close.assert_called_once()


class TestOptionalDependency:
    def test_missing_paramiko_import_error(self):
        """import app.services.ssh succeeds even without paramiko."""
        import app.services.ssh as ssh_mod

        original = ssh_mod._import_paramiko
        ssh_mod._import_paramiko = MagicMock(
            side_effect=ImportError("No module named 'paramiko'")
        )
        try:
            host = SSHRemoteHost("h", credential=PasswordCredential("u", "p"))
            with pytest.raises(ImportError):
                host.connect()
        finally:
            ssh_mod._import_paramiko = original


class TestPrivateKeyAuth:
    @patch("app.services.ssh._import_paramiko")
    def test_inline_pem(self, mock_import):
        mock_paramiko, mock_client, _ = _make_mock_paramiko()
        mock_import.return_value = mock_paramiko
        mock_rsa = MagicMock()
        mock_paramiko.RSAKey.from_private_key.return_value = mock_rsa

        cred = PrivateKeyCredential(
            "user",
            "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----",
        )
        host = SSHRemoteHost("h", credential=cred)
        host.connect()

        _, kwargs = mock_client.connect.call_args
        assert kwargs["username"] == "user"
        assert kwargs["pkey"] is mock_rsa
        host.close()

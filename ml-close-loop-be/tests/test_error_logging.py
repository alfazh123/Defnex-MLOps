import logging
from unittest.mock import patch, AsyncMock

import httpx
import pytest

from tests.conftest import auth_header


class TestHTTPExceptionLogging:
    def test_401_logs_method_path_status(self, client, admin_token, caplog):
        with caplog.at_level(logging.WARNING, logger="app.main"):
            resp = client.get(
                "/api/v1/training-runs/nonexistent",
                headers=auth_header(admin_token),
            )

        assert resp.status_code == 404
        assert any("http_exception" in r.message for r in caplog.records)

    def test_401_logs_error_code(self, client, caplog):
        with caplog.at_level(logging.WARNING, logger="app.main"):
            resp = client.post(
                "/api/v1/auth/login",
                json={"username": "nobody", "password": "Nope1234"},
            )

        assert resp.status_code == 401
        assert any("http_exception" in r.message for r in caplog.records)


class TestAuthErrorLogging:
    def test_failed_login_logs_username_not_password(self, client, caplog):
        client.post(
            "/api/v1/auth/register",
            json={"username": "testuser", "password": "Test1234", "role": "user"},
        )

        with caplog.at_level(logging.WARNING, logger="app.api.auth"):
            resp = client.post(
                "/api/v1/auth/login",
                json={"username": "testuser", "password": "Wrong1234"},
            )

        assert resp.status_code == 401
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("login_failed" in r.message for r in warning_records)
        for record in warning_records:
            if "login_failed" in record.message:
                assert "Wrong1234" not in record.message
                break

    def test_duplicate_register_logs_username(self, client, caplog):
        client.post(
            "/api/v1/auth/register",
            json={"username": "dupuser", "password": "Dup12345", "role": "user"},
        )

        with caplog.at_level(logging.WARNING, logger="app.api.auth"):
            resp = client.post(
                "/api/v1/auth/register",
                json={"username": "dupuser", "password": "Dup12345", "role": "user"},
            )

        assert resp.status_code == 409
        assert any("register_username_taken" in r.message for r in caplog.records)


class TestTrainingErrorLogging:
    def test_training_run_create_does_not_hit_unsloth(
        self, client, admin_token, caplog
    ):
        """Create only queues the run; it must not call Unsloth, so no API failure can
        surface here. Worker owns execution (issue #32)."""
        from unittest.mock import patch

        h = auth_header(admin_token)
        client.post(
            "/api/v1/datasets/no_robots/versions",
            json={
                "source_type": "huggingface",
                "source_dataset": "test/dataset",
                "source_commit_or_snapshot_date": "2026-08-01",
                "source_format": "chatml",
            },
            headers=h,
        )
        report = client.post(
            "/api/v1/datasets/no_robots/versions/1/validate",
            json={
                "records": [
                    {
                        "id": "r1",
                        "messages": [
                            {
                                "role": "user",
                                "content": "What is the capital of France?",
                            },
                            {
                                "role": "assistant",
                                "content": " ".join(f"word{i}" for i in range(25)),
                            },
                        ],
                        "metadata": {
                            "source_dataset": "no_robots",
                            "source_id": "sq-1",
                        },
                    }
                ]
            },
            headers=h,
        )
        assert report.status_code == 201
        assert report.json()["gate_decision"] == "PASS"

        with patch("app.api.training.unsloth_client._get_client") as mock_get:
            with caplog.at_level(logging.ERROR, logger="app.api.training"):
                resp = client.post(
                    "/api/v1/training-runs",
                    json={
                        "dataset_id": "no_robots",
                        "dataset_version": 1,
                        "model_id": "test-model",
                        "base_model": "test/base",
                        "training_config": {"epochs": 1},
                        "triggered_by": "user-1",
                    },
                    headers=h,
                )

        assert resp.status_code == 201
        assert resp.json()["status"] == "PENDING"
        mock_get.assert_not_called()


class TestUnslothErrorLogging:
    def test_unsloth_api_error_logs_endpoint_and_status(self, caplog):
        from app.services import unsloth_client
        from unittest.mock import MagicMock

        def _always_return(response):
            def _fn(*args, **kwargs):
                return response

            return _fn

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Server Error"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=MagicMock(), response=mock_resp
        )

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=_always_return(mock_resp))

        async def run():
            with patch.object(unsloth_client, "_get_client", return_value=mock_client):
                with caplog.at_level(
                    logging.ERROR, logger="app.services.unsloth_client"
                ):
                    with pytest.raises(httpx.HTTPStatusError):
                        await unsloth_client.start_training("run-123", "model", {})

        import asyncio

        asyncio.run(run())

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("unsloth_api_error" in r.message for r in error_records)

    def test_unsloth_connection_error_logs_context(self, caplog):
        from app.services import unsloth_client

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        async def run():
            with patch.object(unsloth_client, "_get_client", return_value=mock_client):
                with caplog.at_level(
                    logging.ERROR, logger="app.services.unsloth_client"
                ):
                    with pytest.raises(httpx.ConnectError):
                        await unsloth_client.start_training("run-456", "model", {})

        import asyncio

        asyncio.run(run())

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("unsloth_api_error" in r.message for r in error_records)

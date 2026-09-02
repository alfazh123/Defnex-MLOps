from app.api import models as models_api
from app.config import settings
from tests.conftest import auth_header


def test_returns_models_and_default_with_valid_token(client, user_token):
    response = client.get("/api/v1/models/available", headers=auth_header(user_token))
    assert response.status_code == 200
    body = response.json()
    assert "models" in body
    assert "default" in body
    assert isinstance(body["models"], list)
    assert isinstance(body["default"], str)


def test_response_shape_matches_settings(client, user_token):
    response = client.get("/api/v1/models/available", headers=auth_header(user_token))
    assert response.status_code == 200
    body = response.json()
    assert body["models"] == [
        m.strip() for m in settings.unsloth_models.split(",") if m.strip()
    ]
    assert body["default"] == settings.unsloth_default_model


def test_default_is_in_models(client, user_token):
    body = client.get(
        "/api/v1/models/available", headers=auth_header(user_token)
    ).json()
    assert body["default"] in body["models"]


def test_returns_401_without_token(client):
    response = client.get("/api/v1/models/available")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "MISSING_TOKEN"


def test_custom_models_override_appears_in_response(client, user_token, monkeypatch):
    monkeypatch.setattr(
        models_api.settings, "unsloth_models", "a,b,c,extra", raising=False
    )
    monkeypatch.setattr(
        models_api.settings, "unsloth_default_model", "extra", raising=False
    )
    body = client.get(
        "/api/v1/models/available", headers=auth_header(user_token)
    ).json()
    assert body["models"] == ["a", "b", "c", "extra"]
    assert body["default"] == "extra"

"""Tests for pagination on list endpoints."""

from tests.conftest import auth_header

DATASET_CREATE = {
    "source_type": "huggingface",
    "source_dataset": "HuggingFaceH4/no_robots",
    "source_commit_or_snapshot_date": "2026-08-01",
    "source_format": "chatml",
}


def _seed_users_via_client(client, admin_token):
    """Create users through the API, using a fresh limiter-reset approach."""
    from app.limiter import limiter

    for i in range(5):
        limiter.reset()
        client.post(
            "/auth/register",
            json={"username": f"seed{i}", "password": "Seed1234", "role": "user"},
        )
    limiter.reset()


def _create_training_run(client, h, dataset_id="no_robots", dataset_version=1):
    client.post(f"/datasets/{dataset_id}/versions", json=DATASET_CREATE, headers=h)
    return client.post(
        "/training-runs",
        json={
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "model_id": "model-a",
            "base_model": "Qwen/Qwen3.8-27B",
            "training_config": {"peft_method": "lora", "lora_r": 8, "lora_alpha": 8},
        },
        headers=h,
    )


# --- /users ---


def test_users_pagination_returns_paginated_response(client, admin_token):
    h = auth_header(admin_token)
    _seed_users_via_client(client, admin_token)

    resp = client.get("/users?page=1&size=3", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 3
    assert data["total"] == 6  # admin + 5 seeded
    assert data["page"] == 1
    assert data["size"] == 3
    assert data["pages"] == 2


def test_users_pagination_page_2(client, admin_token):
    h = auth_header(admin_token)
    _seed_users_via_client(client, admin_token)

    resp = client.get("/users?page=2&size=3", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 3
    assert data["page"] == 2


def test_users_out_of_range_page_returns_empty(client, admin_token):
    h = auth_header(admin_token)
    resp = client.get("/users?page=999&size=20", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []


# --- /datasets ---


def test_datasets_pagination(client, admin_token):
    h = auth_header(admin_token)
    for name in ["ds1", "ds2", "ds3"]:
        client.post(f"/datasets/{name}/versions", json=DATASET_CREATE, headers=h)

    resp = client.get("/datasets?page=1&size=2", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 2
    assert data["total"] == 3
    assert data["pages"] == 2


def test_datasets_empty_returns_paginated(client, admin_token):
    h = auth_header(admin_token)
    resp = client.get("/datasets", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0
    assert data["pages"] == 0


# --- /training-runs ---


def test_training_runs_pagination(client, admin_token):
    h = auth_header(admin_token)
    client.post("/datasets/no_robots/versions", json=DATASET_CREATE, headers=h)
    for _ in range(4):
        _create_training_run(client, h)

    resp = client.get("/training-runs?page=1&size=2", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 2
    assert data["total"] == 4
    assert data["pages"] == 2


def test_training_runs_empty_returns_paginated(client, admin_token):
    h = auth_header(admin_token)
    resp = client.get("/training-runs", headers=h)
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


# --- defaults and validation ---


def test_users_default_page_size(client, admin_token):
    h = auth_header(admin_token)
    _seed_users_via_client(client, admin_token)

    resp = client.get("/users", headers=h)
    data = resp.json()
    assert len(data["items"]) == 6  # all fit in default size=20
    assert data["page"] == 1
    assert data["size"] == 20
    assert data["total"] == 6


def test_max_size_capped_at_100(client, admin_token):
    h = auth_header(admin_token)
    resp = client.get("/users?size=200", headers=h)
    assert resp.status_code == 422


def test_page_must_be_at_least_1(client, admin_token):
    h = auth_header(admin_token)
    resp = client.get("/users?page=0", headers=h)
    assert resp.status_code == 422

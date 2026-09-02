# ============================================================
# FROZEN — shared fixtures, do not modify without team review.
# Parallel QA agents (Phase 1–7) must NOT change this file.
# ============================================================
import contextlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.limiter import limiter
from app.main import app


@pytest.fixture
def count_queries():
    """Context manager that counts SQL statements executed during its body."""

    @contextlib.contextmanager
    def _count():
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        counters = {"n": 0}

        @event.listens_for(engine, "before_cursor_execute")
        def _count_statements(
            conn, cursor, statement, parameters, context, executemany
        ):
            counters["n"] += 1

        with Session(engine) as session:
            yield session, counters
        engine.dispose()

    return _count


@pytest.fixture
def db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    limiter.reset()
    test_client = TestClient(app)
    test_client.engine = engine
    yield test_client
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture
def admin_token(client) -> str:
    """Register an admin user and return their JWT token."""
    client.post(
        "/api/v1/auth/register",
        json={"username": "admin", "password": "Admin1234", "role": "admin"},
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "Admin1234"}
    )
    return resp.json()["access_token"]


@pytest.fixture
def user_token(client) -> str:
    """Register a regular user and return their JWT token."""
    client.post(
        "/api/v1/auth/register",
        json={"username": "alice", "password": "Alice1234", "role": "user"},
    )
    resp = client.post(
        "/api/v1/auth/login", json={"username": "alice", "password": "Alice1234"}
    )
    return resp.json()["access_token"]


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.limiter import limiter
from app.main import app
from app.services import auth_service


@pytest.fixture
def db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
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
    client.post("/auth/register", json={"username": "admin", "password": "admin123", "role": "admin"})
    resp = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    return resp.json()["access_token"]


@pytest.fixture
def user_token(client) -> str:
    """Register a regular user and return their JWT token."""
    client.post("/auth/register", json={"username": "alice", "password": "alice123", "role": "user"})
    resp = client.post("/auth/login", json={"username": "alice", "password": "alice123"})
    return resp.json()["access_token"]


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

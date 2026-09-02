from datetime import datetime, timedelta, timezone

from app.services import auth_service


def test_hash_password_is_bcrypt_and_verifies(db_session):
    hashed = auth_service.hash_password("Secret123")
    assert hashed.startswith("$2")
    assert hashed != "Secret123"
    assert auth_service.verify_password("Secret123", hashed) is True


def test_verify_password_rejects_wrong_password(db_session):
    hashed = auth_service.hash_password("Secret123")
    assert auth_service.verify_password("Wrong456", hashed) is False


def test_create_access_token_has_exp_and_type(db_session):
    token = auth_service.create_access_token({"sub": "1", "role": "user"})
    payload = auth_service.decode_token(token)
    assert payload is not None
    assert payload["type"] == "access"
    assert payload["sub"] == "1"
    assert payload["role"] == "user"
    assert "exp" in payload


def test_create_refresh_token_has_exp_and_type(db_session):
    token = auth_service.create_refresh_token({"sub": "1", "role": "user"})
    payload = auth_service.decode_token(token)
    assert payload is not None
    assert payload["type"] == "refresh"
    assert payload["sub"] == "1"
    assert "exp" in payload


def test_decode_token_returns_none_for_garbage(db_session):
    assert auth_service.decode_token("not.a.token") is None


def test_get_user_by_username_returns_none_when_missing(db_session):
    assert auth_service.get_user_by_username(db_session, "nobody") is None


def test_get_user_by_id_returns_none_when_missing(db_session):
    assert auth_service.get_user_by_id(db_session, 99999) is None


def test_create_user_stores_username_role_and_hashes_password(db_session):
    user = auth_service.create_user(db_session, "bob", "Secret123", "admin")
    db_session.flush()
    assert user.username == "bob"
    assert user.role == "admin"
    assert user.hashed_password != "Secret123"
    assert auth_service.verify_password("Secret123", user.hashed_password)


def test_create_user_default_role_is_user(db_session):
    user = auth_service.create_user(db_session, "bob", "Secret123")
    db_session.flush()
    assert user.role == "user"


def test_list_users_empty_on_fresh_db(db_session):
    users, total = auth_service.list_users(db_session)
    assert users == []
    assert total == 0


def test_list_users_respects_limit_and_offset(db_session):
    for i in range(5):
        auth_service.create_user(db_session, f"user{i}", "Secret123")
        db_session.flush()

    users, total = auth_service.list_users(db_session, limit=2, offset=0)
    assert total == 5
    assert len(users) == 2

    users2, _ = auth_service.list_users(db_session, limit=2, offset=2)
    assert [u.username for u in users2] == ["user2", "user3"]


def test_delete_user_returns_false_when_missing(db_session):
    assert auth_service.delete_user(db_session, 99999) is False


def test_delete_user_removes_user(db_session):
    user = auth_service.create_user(db_session, "bob", "Secret123")
    db_session.flush()
    assert auth_service.delete_user(db_session, user.id) is True
    db_session.flush()
    assert auth_service.get_user_by_id(db_session, user.id) is None


def test_access_token_expired_decodes_to_none(db_session):
    from jose import jwt

    from app.config import settings

    expired = jwt.encode(
        {
            "sub": "1",
            "role": "user",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            "type": "access",
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    assert auth_service.decode_token(expired) is None

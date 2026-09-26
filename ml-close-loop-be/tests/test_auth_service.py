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


def test_work_factor_matches_what_passlib_wrote(db_session):
    """Existing `hashed_password` rows were written by passlib's CryptContext at 12 rounds.
    New hashes must use the same cost so deployed credentials keep verifying and no user's
    hash silently becomes cheaper to crack than it was."""
    hashed = auth_service.hash_password("Secret123")
    assert hashed.startswith(f"$2b${auth_service.BCRYPT_ROUNDS}$")
    assert auth_service.BCRYPT_ROUNDS == 12


def test_password_longer_than_the_bcrypt_limit_is_accepted(db_session):
    """bcrypt only consumes 72 bytes. The input that triggered the whole CI failure was
    longer than that, and passlib raised `ValueError: password cannot be longer than 72
    bytes` out of its own import-time self-test -- taking every auth endpoint down with it.
    Now it truncates, which is what bcrypt has always done with those bytes."""
    long_password = "x" * 200
    hashed = auth_service.hash_password(long_password)
    assert auth_service.verify_password(long_password, hashed) is True
    # Truncation is the documented bcrypt behaviour, so the first 72 bytes verify...
    assert auth_service.verify_password("x" * 72, hashed) is True
    # ...and a different long password does not.
    assert auth_service.verify_password("y" * 200, hashed) is False


def test_multibyte_password_is_measured_in_bytes_not_characters(db_session):
    """72 is a BYTE limit. A 30-character CJK password is ~90 bytes, so a character-based
    check would let it through and bcrypt would still raise."""
    multibyte = "密" * 30  # 3 bytes each = 90 bytes
    hashed = auth_service.hash_password(multibyte)
    assert auth_service.verify_password(multibyte, hashed) is True


def test_verify_password_returns_false_for_a_corrupt_hash(db_session):
    """A malformed `hashed_password` row means "this login fails", not "the endpoint 500s"."""
    assert auth_service.verify_password("Secret123", "not-a-bcrypt-hash") is False
    assert auth_service.verify_password("Secret123", "") is False


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


def test_issue_token_pair_shares_family_id(db_session):
    user = auth_service.create_user(db_session, "bob", "Secret123")
    db_session.flush()

    access, refresh = auth_service.issue_token_pair(user)
    access_payload = auth_service.decode_token(access)
    refresh_payload = auth_service.decode_token(refresh)

    assert access_payload["family_id"] == refresh_payload["family_id"]
    assert access_payload["jti"] != refresh_payload["jti"]


def test_decode_token_never_checks_revocation_for_access_tokens(db_session):
    """Access tokens are never checked against any revocation store (#125):
    even after the token's family is revoked, decode_token still returns it."""
    user = auth_service.create_user(db_session, "bob", "Secret123")
    db_session.flush()

    access, _ = auth_service.issue_token_pair(user)
    family_id = auth_service.decode_token(access)["family_id"]

    auth_service.revoke_refresh_family(db_session, family_id)
    db_session.flush()

    assert auth_service.decode_token(access, db_session) is not None


def test_rotate_refresh_token_rejects_access_token_type(db_session):
    user = auth_service.create_user(db_session, "bob", "Secret123")
    db_session.flush()

    access, _ = auth_service.issue_token_pair(user)
    assert auth_service.rotate_refresh_token(access, db_session) is None


def test_rotate_refresh_token_rejects_unknown_user(db_session):
    token = auth_service.create_refresh_token({"sub": "99999", "role": "user"})
    assert auth_service.rotate_refresh_token(token, db_session) is None


def test_rotate_refresh_token_normal_rotation(db_session):
    user = auth_service.create_user(db_session, "bob", "Secret123")
    db_session.flush()

    _, refresh = auth_service.issue_token_pair(user)
    result = auth_service.rotate_refresh_token(refresh, db_session)

    assert result is not None
    rotated_user, new_access, new_refresh = result
    assert rotated_user.id == user.id
    assert new_refresh != refresh
    # Same family carried forward.
    assert (
        auth_service.decode_token(new_refresh)["family_id"]
        == auth_service.decode_token(refresh)["family_id"]
    )


def test_rotate_refresh_token_reuse_kills_whole_family(db_session):
    user = auth_service.create_user(db_session, "bob", "Secret123")
    db_session.flush()

    _, old_refresh = auth_service.issue_token_pair(user)
    result = auth_service.rotate_refresh_token(old_refresh, db_session)
    assert result is not None
    _, _, new_refresh = result

    # Reuse of the already-rotated token is rejected...
    assert auth_service.rotate_refresh_token(old_refresh, db_session) is None
    # ...and revokes the sibling token that was never itself reused.
    assert auth_service.rotate_refresh_token(new_refresh, db_session) is None


def test_revoke_refresh_family_from_access_token_is_noop_on_garbage(db_session):
    """Must not raise on a malformed/garbage access token (best-effort logout)."""
    auth_service.revoke_refresh_family_from_access_token("not.a.token", db_session)


def test_revoke_refresh_family_is_idempotent(db_session):
    """Revoking an already-revoked family must not raise (e.g. double logout)."""
    auth_service.revoke_refresh_family(db_session, "fam-1")
    db_session.flush()
    auth_service.revoke_refresh_family(db_session, "fam-1")
    db_session.flush()


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

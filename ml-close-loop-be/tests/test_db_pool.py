from app.config import Settings


def test_settings_has_pool_defaults():
    s = Settings()
    assert s.db_pool_size == 5
    assert s.db_max_overflow == 10
    assert s.db_pool_timeout == 30
    assert s.db_pool_recycle == 1800


def test_settings_pool_values_overridable(monkeypatch):
    monkeypatch.setenv("DB_POOL_SIZE", "20")
    monkeypatch.setenv("DB_MAX_OVERFLOW", "5")
    monkeypatch.setenv("DB_POOL_TIMEOUT", "60")
    monkeypatch.setenv("DB_POOL_RECYCLE", "3600")
    s = Settings()
    assert s.db_pool_size == 20
    assert s.db_max_overflow == 5
    assert s.db_pool_timeout == 60
    assert s.db_pool_recycle == 3600


def test_sqlite_engine_has_no_pool_settings():
    from app.db.session import engine, is_sqlite

    assert is_sqlite is True
    pool = engine.pool
    assert pool.size() == 5
    assert pool._max_overflow == 10

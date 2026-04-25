from datetime import UTC, datetime, timedelta

import pytest

from src.auth.exceptions import AuthConfigError, AuthTokenError
from src.auth.jwt import JWTManager


def test_jwt_round_trip() -> None:
    manager = JWTManager("test-secret", expire_days=7)
    now = datetime(2026, 4, 25, tzinfo=UTC)

    token = manager.create_token("JuyaoHuang", avatar_url="https://avatar", now=now)
    payload = manager.verify_token(token, now=now + timedelta(days=1))

    assert payload["sub"] == "JuyaoHuang"
    assert payload["avatar_url"] == "https://avatar"


def test_jwt_rejects_expired_token() -> None:
    manager = JWTManager("test-secret", expire_days=7)
    now = datetime(2026, 4, 25, tzinfo=UTC)

    token = manager.create_token("JuyaoHuang", now=now)

    with pytest.raises(AuthTokenError, match="expired"):
        manager.verify_token(token, now=now + timedelta(days=8))


def test_jwt_rejects_tampered_token() -> None:
    manager = JWTManager("test-secret", expire_days=7)
    token = manager.create_token("JuyaoHuang")
    header, payload, _signature = token.split(".")
    tampered = f"{header}.{payload}.invalid"

    with pytest.raises(AuthTokenError, match="signature"):
        manager.verify_token(tampered)


def test_jwt_rejects_missing_secret_placeholder() -> None:
    with pytest.raises(AuthConfigError, match="secret"):
        JWTManager("${JWT_SECRET_KEY}")


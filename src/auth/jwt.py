"""Small HS256 JWT implementation using the Python standard library."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from src.auth.exceptions import AuthConfigError, AuthTokenError


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode((value + padding).encode("ascii"))
    except Exception as exc:  # noqa: BLE001
        raise AuthTokenError("Invalid token encoding") from exc


def _json_dumps(data: dict[str, Any]) -> bytes:
    return json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")


class JWTManager:
    """Create and verify HS256 JWT tokens."""

    def __init__(
        self,
        secret_key: str,
        *,
        algorithm: str = "HS256",
        expire_days: int = 7,
    ) -> None:
        if algorithm != "HS256":
            raise AuthConfigError("Only HS256 JWT tokens are supported")
        if not secret_key or secret_key.startswith("${"):
            raise AuthConfigError("JWT secret key is not configured")
        if expire_days <= 0:
            raise AuthConfigError("JWT expire_days must be positive")

        self.secret_key = secret_key.encode("utf-8")
        self.algorithm = algorithm
        self.expire_days = expire_days

    def create_token(
        self,
        username: str,
        *,
        avatar_url: str | None = None,
        now: datetime | None = None,
    ) -> str:
        """Create a signed JWT token for a GitHub username."""
        issued_at = now or datetime.now(UTC)
        expires_at = issued_at + timedelta(days=self.expire_days)
        payload: dict[str, Any] = {
            "sub": username,
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        if avatar_url:
            payload["avatar_url"] = avatar_url

        header = {"alg": self.algorithm, "typ": "JWT"}
        signing_input = (
            f"{_b64url_encode(_json_dumps(header))}.{_b64url_encode(_json_dumps(payload))}"
        )
        signature = self._sign(signing_input)
        return f"{signing_input}.{signature}"

    def verify_token(self, token: str, *, now: datetime | None = None) -> dict[str, Any]:
        """Verify a token and return its payload."""
        parts = token.split(".")
        if len(parts) != 3:
            raise AuthTokenError("Invalid token format")

        header_raw, payload_raw, signature = parts
        signing_input = f"{header_raw}.{payload_raw}"
        expected_signature = self._sign(signing_input)
        if not hmac.compare_digest(signature, expected_signature):
            raise AuthTokenError("Invalid token signature")

        try:
            header = json.loads(_b64url_decode(header_raw))
            payload = json.loads(_b64url_decode(payload_raw))
        except json.JSONDecodeError as exc:
            raise AuthTokenError("Invalid token JSON") from exc

        if header.get("alg") != self.algorithm:
            raise AuthTokenError("Invalid token algorithm")
        if not isinstance(payload.get("sub"), str) or not payload["sub"]:
            raise AuthTokenError("Token subject is missing")

        current_time = int((now or datetime.now(UTC)).timestamp())
        exp = payload.get("exp")
        if not isinstance(exp, int) or exp <= current_time:
            raise AuthTokenError("Token has expired")

        return payload

    def _sign(self, signing_input: str) -> str:
        digest = hmac.new(
            self.secret_key,
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()
        return _b64url_encode(digest)


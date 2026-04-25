"""Application-level authentication service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.auth.exceptions import AuthConfigError, AuthTokenError, AuthUnauthorizedError
from src.auth.jwt import JWTManager
from src.auth.oauth import GitHubOAuth, GitHubUser
from src.auth.whitelist import Whitelist


@dataclass(frozen=True)
class AuthenticatedUser:
    """Authenticated user identity exposed to routes."""

    username: str
    avatar_url: str | None = None
    name: str | None = None


class AuthService:
    """Auth facade used by routes, middleware, and WebSocket handlers."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        jwt_config = config.get("jwt", {})
        github_config = config.get("github", {})
        whitelist_config = config.get("whitelist", {})

        self.jwt_manager: JWTManager | None = None
        self.github_oauth: GitHubOAuth | None = None
        self.whitelist = Whitelist(whitelist_config.get("users", []))
        self.frontend_callback_url = str(
            config.get("frontend", {}).get("callback_url")
            or "http://localhost:5173/auth/callback"
        )
        self.frontend_login_url = str(
            config.get("frontend", {}).get("login_url") or "http://localhost:5173/login"
        )

        if self.enabled:
            self.jwt_manager = JWTManager(
                str(jwt_config.get("secret_key", "")),
                algorithm=str(jwt_config.get("algorithm", "HS256")),
                expire_days=int(jwt_config.get("expire_days", 7)),
            )
            self.github_oauth = GitHubOAuth(
                client_id=str(github_config.get("client_id", "")),
                client_secret=str(github_config.get("client_secret", "")),
                callback_url=str(github_config.get("callback_url", "")),
                scope=str(github_config.get("scope", "read:user")),
            )

    def create_token_for_github_user(self, user: GitHubUser) -> str:
        if not self.jwt_manager:
            raise AuthConfigError("JWT manager is not configured")
        return self.jwt_manager.create_token(user.username, avatar_url=user.avatar_url)

    def require_allowed_user(self, user: GitHubUser) -> None:
        if not self.whitelist.is_allowed(user.username):
            raise AuthUnauthorizedError(f"GitHub user '{user.username}' is not whitelisted")

    def authenticate_bearer_token(self, authorization: str | None) -> AuthenticatedUser:
        if not self.enabled:
            return AuthenticatedUser(username="default")
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthTokenError("Missing bearer token")
        return self.authenticate_token(authorization.removeprefix("Bearer ").strip())

    def authenticate_token(self, token: str | None) -> AuthenticatedUser:
        if not self.enabled:
            return AuthenticatedUser(username="default")
        if not token:
            raise AuthTokenError("Missing token")
        if not self.jwt_manager:
            raise AuthConfigError("JWT manager is not configured")
        payload = self.jwt_manager.verify_token(token)
        username = str(payload["sub"])
        if not self.whitelist.is_allowed(username):
            raise AuthUnauthorizedError(f"GitHub user '{username}' is not whitelisted")
        avatar_url = payload.get("avatar_url")
        return AuthenticatedUser(
            username=username,
            avatar_url=avatar_url if isinstance(avatar_url, str) else None,
        )


"""Authentication helpers for GitHub OAuth, JWT, and whitelist checks."""

from src.auth.dependencies import DEFAULT_USER_ID, get_request_user_id, get_websocket_user_id
from src.auth.exceptions import AuthConfigError, AuthError, AuthUnauthorizedError
from src.auth.jwt import JWTManager
from src.auth.oauth import GitHubOAuth, GitHubUser
from src.auth.service import AuthenticatedUser, AuthService
from src.auth.whitelist import Whitelist

__all__ = [
    "DEFAULT_USER_ID",
    "AuthenticatedUser",
    "AuthConfigError",
    "AuthError",
    "AuthService",
    "AuthUnauthorizedError",
    "GitHubOAuth",
    "GitHubUser",
    "JWTManager",
    "Whitelist",
    "get_request_user_id",
    "get_websocket_user_id",
]

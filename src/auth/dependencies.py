"""Authentication helpers shared by HTTP routes and WebSocket handlers."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request, WebSocket, status

from src.auth.exceptions import AuthError
from src.auth.service import AuthService

DEFAULT_USER_ID = "default"


def get_auth_service(app: Any) -> AuthService:
    service = getattr(app.state, "auth_service", None)
    if isinstance(service, AuthService):
        return service
    service = AuthService(getattr(app.state, "config", {}).get("auth", {}))
    app.state.auth_service = service
    return service


def get_request_user_id(request: Request) -> str:
    """Return current user id for REST routes."""
    if hasattr(request.state, "user_id"):
        return str(request.state.user_id)
    auth_service = get_auth_service(request.app)
    if not auth_service.enabled:
        return DEFAULT_USER_ID
    try:
        user = auth_service.authenticate_bearer_token(request.headers.get("Authorization"))
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    request.state.user_id = user.username
    return user.username


def get_websocket_user_id(websocket: WebSocket) -> str:
    """Return current user id for WebSocket routes."""
    auth_service = get_auth_service(websocket.app)
    if not auth_service.enabled:
        return DEFAULT_USER_ID
    token = websocket.query_params.get("token")
    try:
        user = auth_service.authenticate_token(token)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    return user.username


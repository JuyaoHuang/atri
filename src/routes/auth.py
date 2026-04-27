"""Authentication API routes."""

from __future__ import annotations

import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from src.auth.dependencies import DEFAULT_USER_ID, get_auth_service
from src.auth.exceptions import AuthError
from src.auth.session import SESSION_COOKIE_NAME, SESSION_COOKIE_PATH, SESSION_COOKIE_SAMESITE

router = APIRouter(prefix="/api/auth", tags=["auth"])

OAUTH_STATE_COOKIE_NAME = "atri_oauth_state"
OAUTH_STATE_COOKIE_MAX_AGE_SECONDS = 10 * 60
OAUTH_STATE_COOKIE_PATH = "/api/auth"


class AuthStatusResponse(BaseModel):
    enabled: bool


class AuthLoginResponse(BaseModel):
    enabled: bool
    authorization_url: str | None = None


class AuthUserResponse(BaseModel):
    username: str
    avatar_url: str | None = None
    name: str | None = None
    auth_enabled: bool


class AuthLogoutResponse(BaseModel):
    success: bool


def _redirect_with_params(base_url: str, params: dict[str, str]) -> RedirectResponse:
    separator = "&" if "?" in base_url else "?"
    return RedirectResponse(f"{base_url}{separator}{urlencode(params)}")


def _is_secure_oauth_cookie(request: Request) -> bool:
    auth_service = get_auth_service(request.app)
    callback_url = auth_service.github_oauth.callback_url if auth_service.github_oauth else ""
    return callback_url.startswith("https://")


def _is_secure_session_cookie(request: Request) -> bool:
    auth_service = get_auth_service(request.app)
    return auth_service.frontend_callback_url.startswith("https://")


def _set_oauth_state_cookie(request: Request, response: Response, state: str) -> None:
    response.set_cookie(
        key=OAUTH_STATE_COOKIE_NAME,
        value=state,
        max_age=OAUTH_STATE_COOKIE_MAX_AGE_SECONDS,
        path=OAUTH_STATE_COOKIE_PATH,
        secure=_is_secure_oauth_cookie(request),
        httponly=True,
        samesite="lax",
    )


def _set_session_cookie(request: Request, response: Response, token: str) -> None:
    auth_service = get_auth_service(request.app)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=auth_service.session_cookie_max_age_seconds,
        path=SESSION_COOKIE_PATH,
        secure=_is_secure_session_cookie(request),
        httponly=True,
        samesite=SESSION_COOKIE_SAMESITE,
    )


def _clear_session_cookie(request: Request, response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path=SESSION_COOKIE_PATH,
        secure=_is_secure_session_cookie(request),
        httponly=True,
        samesite=SESSION_COOKIE_SAMESITE,
    )


def _clear_oauth_state_cookie(request: Request, response: Response) -> None:
    response.delete_cookie(
        key=OAUTH_STATE_COOKIE_NAME,
        path=OAUTH_STATE_COOKIE_PATH,
        secure=_is_secure_oauth_cookie(request),
        httponly=True,
        samesite="lax",
    )


def _redirect_with_cleared_oauth_state(
    request: Request,
    base_url: str,
    params: dict[str, str],
) -> RedirectResponse:
    response = _redirect_with_params(base_url, params)
    _clear_oauth_state_cookie(request, response)
    return response


def _is_valid_oauth_state(request: Request, state: str | None) -> bool:
    expected_state = request.cookies.get(OAUTH_STATE_COOKIE_NAME)
    if not state or not expected_state:
        return False
    return secrets.compare_digest(state, expected_state)


def _has_valid_session_cookie(request: Request) -> bool:
    auth_service = get_auth_service(request.app)
    try:
        auth_service.authenticate_credentials(
            authorization=None,
            session_token=request.cookies.get(SESSION_COOKIE_NAME),
        )
    except AuthError:
        return False
    return True


@router.get("/status", response_model=AuthStatusResponse)
async def get_auth_status(request: Request) -> AuthStatusResponse:
    auth_service = get_auth_service(request.app)
    return AuthStatusResponse(enabled=auth_service.enabled)


@router.get("/login", response_model=AuthLoginResponse)
async def get_login_url(
    request: Request,
    response: Response,
    state: str | None = Query(None),  # Kept for backward-compatible clients; ignored.
) -> AuthLoginResponse:
    auth_service = get_auth_service(request.app)
    if not auth_service.enabled:
        return AuthLoginResponse(enabled=False, authorization_url=None)
    if auth_service.github_oauth is None:
        raise HTTPException(status_code=500, detail="GitHub OAuth is not configured")

    oauth_state = secrets.token_urlsafe(32)
    _set_oauth_state_cookie(request, response, oauth_state)
    return AuthLoginResponse(
        enabled=True,
        authorization_url=auth_service.github_oauth.get_authorization_url(state=oauth_state),
    )


@router.get("/callback")
async def github_callback(
    request: Request,
    code: str | None = Query(None),
    state: str | None = Query(None),
    error: str | None = Query(None),
) -> RedirectResponse:
    auth_service = get_auth_service(request.app)
    if not auth_service.enabled:
        return _redirect_with_cleared_oauth_state(
            request,
            auth_service.frontend_callback_url,
            {"auth": "disabled"},
        )
    if error:
        return _redirect_with_cleared_oauth_state(
            request,
            auth_service.frontend_callback_url,
            {"error": error},
        )
    if not code:
        return _redirect_with_cleared_oauth_state(
            request,
            auth_service.frontend_callback_url,
            {"error": "missing_code"},
        )
    if auth_service.github_oauth is None:
        return _redirect_with_cleared_oauth_state(
            request,
            auth_service.frontend_callback_url,
            {"error": "oauth_config"},
        )
    if not _is_valid_oauth_state(request, state):
        if _has_valid_session_cookie(request):
            return _redirect_with_cleared_oauth_state(
                request,
                auth_service.frontend_callback_url,
                {"success": "1"},
            )
        return _redirect_with_cleared_oauth_state(
            request,
            auth_service.frontend_callback_url,
            {"error": "invalid_state"},
        )

    try:
        access_token = await auth_service.github_oauth.exchange_code_for_token(code)
        github_user = await auth_service.github_oauth.get_user_info(access_token)
        auth_service.require_allowed_user(github_user)
        token = auth_service.create_token_for_github_user(github_user)
    except AuthError as exc:
        return _redirect_with_cleared_oauth_state(
            request,
            auth_service.frontend_callback_url,
            {"error": "unauthorized", "detail": str(exc)},
        )
    except Exception:
        return _redirect_with_cleared_oauth_state(
            request,
            auth_service.frontend_callback_url,
            {"error": "github_oauth_failed"},
        )

    response = _redirect_with_cleared_oauth_state(
        request,
        auth_service.frontend_callback_url,
        {"success": "1"},
    )
    _set_session_cookie(request, response, token)
    return response


@router.get("/me", response_model=AuthUserResponse)
async def get_current_user(request: Request) -> AuthUserResponse:
    auth_service = get_auth_service(request.app)
    if not auth_service.enabled:
        return AuthUserResponse(username=DEFAULT_USER_ID, auth_enabled=False)
    try:
        user = auth_service.authenticate_credentials(
            authorization=request.headers.get("Authorization"),
            session_token=request.cookies.get(SESSION_COOKIE_NAME),
        )
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    return AuthUserResponse(
        username=user.username,
        avatar_url=user.avatar_url,
        name=user.name,
        auth_enabled=True,
    )


@router.post("/logout", response_model=AuthLogoutResponse)
async def logout(request: Request, response: Response) -> AuthLogoutResponse:
    _clear_session_cookie(request, response)
    return AuthLogoutResponse(success=True)

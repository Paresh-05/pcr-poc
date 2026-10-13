"""Access control for the demo page's own API (/ui-api): the browser never holds the API key.

- No DEMO_PASSCODE set: only requests from this computer (loopback) are served.
- DEMO_PASSCODE set: visitors enter it once; the server then sets an HttpOnly, SameSite=Strict cookie holding a
  signed, expiring token (HMAC keyed with the server's API key).
State-changing requests must also come from the page's own origin.
"""
import hashlib
import hmac
import secrets
import time
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, StringConstraints
from typing import Annotated

from app.config import Settings, get_settings
from app.ratelimit import login_rate_limit

COOKIE = "dtc_ui"
SESSION_SECONDS = 12 * 3600
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _sign(settings: Settings, exp: int) -> str:
    key = settings.api_key.get_secret_value().encode()
    return hmac.new(key, f"dtc-ui:{exp}".encode(), hashlib.sha256).hexdigest()


def make_token(settings: Settings) -> str:
    exp = int(time.time()) + SESSION_SECONDS
    return f"{exp}.{_sign(settings, exp)}"


def token_is_valid(settings: Settings, token: str | None) -> bool:
    try:
        exp_text, sig = (token or "").split(".", 1)
        exp = int(exp_text)
    except ValueError:
        return False
    return exp > time.time() and hmac.compare_digest(sig, _sign(settings, exp))


def is_authenticated(request: Request, settings: Settings) -> bool:
    if settings.demo_passcode:
        return token_is_valid(settings, request.cookies.get(COOKIE))
    return (request.client.host if request.client else "") in LOOPBACK


def require_ui_access(request: Request, settings: Settings = Depends(get_settings)) -> None:
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Cross-origin requests are not allowed")
    if not is_authenticated(request, settings):
        if settings.demo_passcode:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Passcode required")
        raise HTTPException(status.HTTP_403_FORBIDDEN, "The demo page is only available on this computer")


class PasscodeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passcode: Annotated[str, StringConstraints(min_length=1, max_length=200)]


session_router = APIRouter(prefix="/ui-api/session")


@session_router.get("")
async def session_state(request: Request, settings: Settings = Depends(get_settings)) -> dict:
    return {"passcode_required": bool(settings.demo_passcode), "authenticated": is_authenticated(request, settings)}


@session_router.post("", dependencies=[Depends(login_rate_limit)])
async def log_in(body: PasscodeIn, request: Request, response: Response, settings: Settings = Depends(get_settings)) -> dict:
    origin = request.headers.get("origin")
    if origin and urlparse(origin).netloc != request.headers.get("host"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Cross-origin requests are not allowed")
    if not settings.demo_passcode:
        return {"passcode_required": False, "authenticated": is_authenticated(request, settings)}
    if not secrets.compare_digest(body.passcode.encode(), settings.demo_passcode.get_secret_value().encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong passcode")
    response.set_cookie(
        COOKIE, make_token(settings), max_age=SESSION_SECONDS, httponly=True, samesite="strict",
        secure=request.url.scheme == "https", path="/ui-api",
    )
    return {"passcode_required": True, "authenticated": True}

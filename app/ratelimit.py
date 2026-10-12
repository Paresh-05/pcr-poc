"""Minimal in-memory sliding-window rate limiter (single process; swap for Redis when scaling)."""
import time
from collections import deque

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings, get_settings

_hits: dict[str, deque] = {}


def rate_limit(request: Request, settings: Settings = Depends(get_settings)) -> None:
    key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _hits.setdefault(key, deque())
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests")
    window.append(now)

import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.db import create_pool
from app.llm import OpenAILLM
from app.logging_config import setup_logging
from app.routes import make_router
from app.ui_access import require_ui_access, session_router
from app.usage import UsageRecorder

setup_logging(get_settings())
logger = logging.getLogger("dtcbot")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    logger.info("Starting DTC Bot (env=%s, llm_enabled=%s, model=%s)", settings.app_env, settings.llm_enabled, settings.llm_model)
    app.state.pool = create_pool(settings)
    await app.state.pool.open()
    logger.info("Database pool opened (host=%s db=%s)", settings.postgres_host, settings.postgres_db)
    app.state.llm = OpenAILLM(settings, UsageRecorder(app.state.pool)) if settings.llm_enabled else None
    if app.state.llm is None:
        logger.warning("LLM not configured (LLM_API_KEY / LLM_MODEL); diagnosis endpoints are disabled")
    yield
    logger.info("Shutting down: closing database pool")
    await app.state.pool.close()


def require_api_key(key: str | None = Depends(api_key_header), settings: Settings = Depends(get_settings)) -> None:
    expected = settings.api_key.get_secret_value()
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unauthorized")


_dev = get_settings().is_dev
app = FastAPI(
    title="DTC Bot",
    lifespan=lifespan,
    docs_url="/docs" if _dev else None,
    redoc_url=None,
    openapi_url="/openapi.json" if _dev else None,
)

UI_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("%s %s -> unhandled exception after %dms", request.method, request.url.path, int((time.monotonic() - started) * 1000))
        raise
    elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.info("%s %s -> %d (%dms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith(("/ui", "/ui-api")):
        response.headers["Content-Security-Policy"] = UI_CSP
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error: %s", type(exc).__name__)
    return JSONResponse({"error": "Something went wrong. Please try again later."}, status_code=500)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/health/db", dependencies=[Depends(require_api_key)])
async def health_db(request: Request) -> dict:
    async with request.app.state.pool.connection() as conn:
        cur = await conn.execute("SELECT count(*) FROM pcr_cases")
        (cases,) = await cur.fetchone()
        cur = await conn.execute("SELECT count(*) FROM manual_chunks")
        (manual,) = await cur.fetchone()
    return {"status": "ok", "cases": cases, "manual_chunks": manual}


app.include_router(make_router(require_api_key))  # for programs: X-API-Key header

if _dev:
    app.include_router(make_router(require_ui_access), prefix="/ui-api")
    app.include_router(session_router)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")

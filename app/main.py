import logging
import secrets
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from app import conversations
from app.config import Settings, get_settings
from app.db import create_pool
from app.llm import OpenAILLM
from app.ratelimit import rate_limit
from app.schemas import StartConversation, UserMessage

logger = logging.getLogger("dtcbot")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.pool = create_pool(settings)
    await app.state.pool.open()
    app.state.llm = OpenAILLM(settings) if settings.llm_enabled else None
    if app.state.llm is None:
        logger.warning("LLM not configured (LLM_API_KEY / LLM_MODEL); conversation endpoints are disabled")
    yield
    await app.state.pool.close()


def require_api_key(key: str | None = Depends(api_key_header), settings: Settings = Depends(get_settings)) -> None:
    expected = settings.api_key.get_secret_value()
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unauthorized")


def require_llm(request: Request):
    llm = request.app.state.llm
    if llm is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "AI assistant is not configured")
    return llm


_dev = get_settings().is_dev
app = FastAPI(
    title="DTC Bot",
    lifespan=lifespan,
    docs_url="/docs" if _dev else None,
    redoc_url=None,
    openapi_url="/openapi.json" if _dev else None,
)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled error: %s", type(exc).__name__)
    return JSONResponse({"error": "Something went wrong. Please try again later."}, status_code=500)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/health/db", dependencies=[Depends(require_api_key)])
async def health_db(request: Request) -> dict:
    async with request.app.state.pool.connection() as conn:
        cur = await conn.execute("SELECT count(*) FROM cases")
        (cases,) = await cur.fetchone()
        cur = await conn.execute("SELECT count(*) FROM manual_sections")
        (manual,) = await cur.fetchone()
    return {"status": "ok", "cases": cases, "manual_sections": manual}


@app.post("/conversations", status_code=201, dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def start_conversation(body: StartConversation, request: Request, llm=Depends(require_llm)) -> dict:
    return await conversations.start_conversation(request.app.state.pool, llm, body.dtc_code, body.model_line)


@app.post("/conversations/{conversation_id}/messages", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def send_message(conversation_id: uuid.UUID, body: UserMessage, request: Request, llm=Depends(require_llm)):
    try:
        return await conversations.send_message(request.app.state.pool, llm, conversation_id, body.content)
    except conversations.NotFound:
        return JSONResponse({"error": "Conversation not found"}, status_code=404)
    except conversations.LimitReached:
        return JSONResponse({"error": "This conversation reached its message limit. Start a new one."}, status_code=409)


@app.get("/conversations/{conversation_id}", dependencies=[Depends(require_api_key)])
async def get_conversation(conversation_id: uuid.UUID, request: Request):
    try:
        return await conversations.get_conversation(request.app.state.pool, conversation_id)
    except conversations.NotFound:
        return JSONResponse({"error": "Conversation not found"}, status_code=404)

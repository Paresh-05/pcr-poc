import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from app import llm
from app.config import Settings, get_settings
from app.db import create_pool
from app.embeddings import embed_query
from app.manual_client import fetch_manual
from app.ratelimit import rate_limit
from app.retrieval import retrieve_cases
from app.schemas import AnalyzeRequest
from app.stats import compute_stats

logger = logging.getLogger("dtcbot")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _embedding_dim(settings: Settings, pool) -> int:
    """Vector size of the DB column if hybrid search can be used, else 0 (keyword only).

    Hybrid needs: switch on, an embedding key, and embeddings already stored in the database.
    """
    if not settings.embeddings_enabled:
        logger.info("retrieval mode: keyword only (EMBEDDINGS_ENABLED=false)")
        return 0
    if settings.effective_embedding_key is None:
        logger.warning("EMBEDDINGS_ENABLED=true but no embedding key set; using keyword only")
        return 0
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT count(*) FROM cases WHERE complaint_embedding IS NOT NULL")
        (n,) = await cur.fetchone()
        cur = await conn.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'cases'::regclass AND attname = 'complaint_embedding'"
        )
        (dim,) = await cur.fetchone()
    if n == 0:
        logger.warning("EMBEDDINGS_ENABLED=true but no embeddings stored; run scripts/embed.py. Using keyword only")
        return 0
    logger.info("retrieval mode: hybrid (%d embedded cases, %d dims)", n, dim)
    return dim


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.pool = create_pool(settings)
    await app.state.pool.open()
    app.state.embedding_dim = await _embedding_dim(settings, app.state.pool)
    yield
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
        (count,) = await cur.fetchone()
    return {"status": "ok", "cases": count}


@app.post("/analyze", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def analyze(body: AnalyzeRequest, request: Request, settings: Settings = Depends(get_settings)) -> dict:
    pool = request.app.state.pool
    query_vec = None
    if request.app.state.embedding_dim:
        try:
            query_vec = await embed_query(settings, body.complaint, request.app.state.embedding_dim)
        except Exception as exc:  # embedding outage must not break the request
            logger.error("query embedding failed, using keyword only: %s", type(exc).__name__)
    cases = await retrieve_cases(
        pool, body.dtc_code, body.model_line, body.complaint,
        query_vec=query_vec, max_distance=settings.embedding_max_distance,
    )
    stats = await compute_stats(pool, body.dtc_code, body.model_line)
    manual = await fetch_manual(settings, body.dtc_code, body.model_line, body.module)

    notices = {
        "used": f"Official manual section used ({manual.module}).",
        "needs_module": "Several manual modules match this DTC; resend with `module` set to one of manual.modules.",
        "not_found": "No manual section found for this DTC and model; verify against the official manual.",
        "unavailable": "Manual service unavailable; verify steps against the official manual.",
        "disabled": "Manual service not configured; verify steps against the official manual.",
    }
    notice = notices[manual.status]

    analysis = None
    if not cases and manual.status != "used":
        notice += " No matching past cases found for this DTC or complaint."
    elif settings.effective_llm_key is None:
        notice += " LLM analysis disabled (no LLM key set); showing retrieved evidence only."
    else:
        try:
            result = await llm.analyze(
                settings, body.dtc_code, body.model_line, body.complaint, stats, cases,
                manual_text=manual.text if manual.status == "used" else None,
            )
            analysis = result.model_dump()
        except Exception as exc:  # never leak provider errors to the client
            logger.error("llm analysis failed: %s", type(exc).__name__)
            notice += " LLM analysis failed; showing retrieved evidence only."

    return {
        "notice": notice,
        "analysis": analysis,
        "manual": {"status": manual.status, "module": manual.module, "modules": manual.modules},
        "evidence": {
            "retrieval_mode": "hybrid" if query_vec else "keyword",
            "stats": stats,
            "cases": [
                {
                    "id": c.id,
                    "tier": c.tier,
                    "model_line": c.model_line,
                    "dtc_code": c.dtc_code,
                    "complaint": c.complaint,
                    "diagnosis": c.diagnosis,
                    "failed_parts": c.failed_parts,
                    "score": round(c.score, 4),
                }
                for c in cases
            ],
        },
    }

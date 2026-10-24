"""The diagnosis API. Built twice from one definition: with API-key auth (/) and with demo-page auth (/ui-api)."""
import json
import logging
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app import diagnosis
from app.manual import complete_manual_dtcs, has_complete_manual
from app.ratelimit import rate_limit
from app.redact import redact
from app.schemas import OutcomeIn, StartDiagnosis

logger = logging.getLogger("dtcbot.routes")

_MODEL_LINE = re.compile(r"^[A-Z0-9_]{2,32}$")


def require_llm(request: Request) -> None:
    if request.app.state.llm is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "AI assistant is not configured")


def make_router(auth) -> APIRouter:
    router = APIRouter(dependencies=[Depends(auth)])

    @router.get("/catalog")
    async def catalog_models(request: Request) -> dict:
        async with request.app.state.pool.connection() as conn:
            cur = await conn.execute("SELECT model_line, count(*) FROM pcr_cases GROUP BY 1 ORDER BY 2 DESC, 1")
            return {"models": [{"model_line": m, "cases": n} for m, n in await cur.fetchall()]}

    @router.get("/catalog/{model_line}")
    async def catalog_dtcs(model_line: str, request: Request):
        model_line = model_line.strip().upper()
        if not _MODEL_LINE.match(model_line):
            return JSONResponse({"error": "Invalid model"}, status_code=422)
        complete_dtcs = await complete_manual_dtcs(request.app.state.pool, model_line)
        async with request.app.state.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT dtc_code, count(*) AS n FROM pcr_cases WHERE model_line = %s "
                "GROUP BY dtc_code ORDER BY n DESC, dtc_code",
                (model_line,),
            )
            rows = await cur.fetchall()
        return {"model_line": model_line, "dtcs": [
            {"dtc_code": d, "cases": n, "has_manual": True} for d, n in rows if d in complete_dtcs
        ]}

    @router.post("/diagnoses", status_code=201, dependencies=[Depends(rate_limit), Depends(require_llm)])
    async def start_diagnosis(body: StartDiagnosis, request: Request) -> dict:
        s = request.app.state
        logger.info("POST /diagnoses: model_line=%s dtc_code=%s query=%s", body.model_line, body.dtc_code,
                    json.dumps(redact(body.query_text)) if body.query_text else None)
        # The catalog (dropdown) flow gives an exact code up front, so it's worth a clear 404/422 before
        # spending an LLM call. Free-text queries may resolve to no code, or a code not yet in our catalog
        # check - those fall through to diagnosis.start_diagnosis, which returns status="no_evidence" instead.
        if body.dtc_code and not body.query_text:
            async with s.pool.connection() as conn:
                cur = await conn.execute(
                    "SELECT 1 FROM pcr_cases WHERE dtc_code = %s AND model_line = %s LIMIT 1",
                    (body.dtc_code, body.model_line),
                )
                if await cur.fetchone() is None:
                    logger.warning("start_diagnosis: dtc_code=%s not found for model_line=%s -> 404", body.dtc_code, body.model_line)
                    raise HTTPException(
                        status.HTTP_404_NOT_FOUND,
                        "This fault code was not found for the selected model",
                    )
            if not await has_complete_manual(s.pool, body.dtc_code, body.model_line):
                logger.warning("start_diagnosis: no manual for dtc_code=%s model_line=%s -> 422", body.dtc_code, body.model_line)
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "A manual procedure is not available for this model and fault code",
                )
        return await diagnosis.start_diagnosis(s.pool, s.llm, s.settings, body.model_line, body.dtc_code, body.query_text)

    # No /answers or GET /diagnoses/{id} endpoint: the whole tree is returned by /diagnoses (above) and
    # walked client-side - no per-answer round-trip, no resume.

    @router.get("/diagnoses/{diagnosis_id}/cost")
    async def get_cost(diagnosis_id: uuid.UUID, request: Request) -> dict:
        return await diagnosis.list_calls(request.app.state.pool, diagnosis_id)

    @router.post("/diagnoses/{diagnosis_id}/outcome", dependencies=[Depends(rate_limit)])
    async def save_outcome(diagnosis_id: uuid.UUID, body: OutcomeIn, request: Request):
        try:
            return await diagnosis.save_outcome(request.app.state.pool, diagnosis_id, body.result, body.actual_cause, body.parts_replaced, body.notes)
        except diagnosis.NotFound:
            return JSONResponse({"error": "Diagnosis not found"}, status_code=404)

    @router.get("/outcomes/summary")
    async def outcomes_summary(request: Request) -> dict:
        return await diagnosis.outcome_summary(request.app.state.pool)

    return router

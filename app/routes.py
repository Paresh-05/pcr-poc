"""The diagnosis API. Built twice from one definition: with API-key auth (/) and with demo-page auth (/ui-api)."""
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app import diagnosis
from app.ratelimit import rate_limit
from app.redact import redact
from app.schemas import AnswerIn, OutcomeIn, StartDiagnosis

_MODEL_LINE = re.compile(r"^[A-Z0-9_]{2,32}$")

# DTC + model pairs that have manual text AND enough past cases to make a good demo, with their typical complaint.
_FEATURED_SQL = """
    SELECT c.dtc_code, c.model_line, count(*) AS n,
           (SELECT lower(btrim(x.customer_verbatim)) FROM cases x
             WHERE x.dtc_code = c.dtc_code AND x.model_line = c.model_line
             GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT 1) AS typical
    FROM cases c
    WHERE EXISTS (SELECT 1 FROM manual_dtcs d JOIN manual_model_map mm ON mm.manual_model = d.manual_model
                  WHERE d.dtc_code = c.dtc_code AND mm.model_line = c.model_line)
    GROUP BY c.dtc_code, c.model_line HAVING count(*) >= 40
    ORDER BY n DESC LIMIT 12
"""


def require_llm(request: Request) -> None:
    if request.app.state.llm is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "AI assistant is not configured")


def make_router(auth) -> APIRouter:
    router = APIRouter(dependencies=[Depends(auth)])

    @router.get("/catalog")
    async def catalog_models(request: Request) -> dict:
        """Car models present in the case data, busiest first."""
        async with request.app.state.pool.connection() as conn:
            cur = await conn.execute("SELECT model_line, count(*) FROM cases GROUP BY 1 ORDER BY 2 DESC, 1")
            return {"models": [{"model_line": m, "cases": n} for m, n in await cur.fetchall()]}

    @router.get("/catalog/featured")
    async def catalog_featured(request: Request) -> dict:
        async with request.app.state.pool.connection() as conn:
            cur = await conn.execute(_FEATURED_SQL)
            rows = await cur.fetchall()
        return {"featured": [{"dtc_code": d, "model_line": m, "cases": n, "typical_complaint": redact(t or "")[:90]} for d, m, n, t in rows]}

    @router.get("/catalog/{model_line}")
    async def catalog_dtcs(model_line: str, request: Request):
        """The most common DTCs for one model, with case counts and whether manual text is loaded."""
        model_line = model_line.strip().upper()
        if not _MODEL_LINE.match(model_line):
            return JSONResponse({"error": "Invalid model"}, status_code=422)
        async with request.app.state.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT c.dtc_code, count(*) AS n, bool_or(m.dtc_code IS NOT NULL) AS has_manual FROM cases c "
                "LEFT JOIN (SELECT DISTINCT d.dtc_code, mm.model_line FROM manual_dtcs d "
                "           JOIN manual_model_map mm ON mm.manual_model = d.manual_model) m "
                "  ON m.dtc_code = c.dtc_code AND m.model_line = c.model_line "
                "WHERE c.model_line = %s GROUP BY c.dtc_code ORDER BY n DESC, c.dtc_code LIMIT 300",
                (model_line,),
            )
            return {"model_line": model_line, "dtcs": [{"dtc_code": d, "cases": n, "has_manual": bool(h)} for d, n, h in await cur.fetchall()]}

    @router.post("/diagnoses", status_code=201, dependencies=[Depends(rate_limit), Depends(require_llm)])
    async def start_diagnosis(body: StartDiagnosis, request: Request) -> dict:
        s = request.app.state
        return await diagnosis.start_diagnosis(s.pool, s.llm, s.graph, body.dtc_code, body.model_line)

    @router.post("/diagnoses/{diagnosis_id}/answers", dependencies=[Depends(rate_limit), Depends(require_llm)])
    async def answer(diagnosis_id: uuid.UUID, body: AnswerIn, request: Request):
        s = request.app.state
        try:
            return await diagnosis.answer_diagnosis(s.pool, s.llm, s.graph, diagnosis_id, body.answer, body.note)
        except diagnosis.NotFound:
            return JSONResponse({"error": "Diagnosis not found"}, status_code=404)
        except diagnosis.NotAsking:
            return JSONResponse({"error": "This diagnosis is finished and takes no more answers"}, status_code=409)
        except diagnosis.Conflict:
            return JSONResponse({"error": "That question was already answered"}, status_code=409)

    @router.get("/diagnoses/{diagnosis_id}")
    async def get_diagnosis(diagnosis_id: uuid.UUID, request: Request):
        try:
            return await diagnosis.get_diagnosis(request.app.state.pool, diagnosis_id)
        except diagnosis.NotFound:
            return JSONResponse({"error": "Diagnosis not found"}, status_code=404)

    @router.get("/diagnoses/{diagnosis_id}/cost")
    async def get_cost(diagnosis_id: uuid.UUID, request: Request) -> dict:
        return await diagnosis.list_calls(request.app.state.pool, diagnosis_id)

    @router.post("/diagnoses/{diagnosis_id}/outcome", dependencies=[Depends(rate_limit)])
    async def save_outcome(diagnosis_id: uuid.UUID, body: OutcomeIn, request: Request):
        try:
            return await diagnosis.save_outcome(request.app.state.pool, diagnosis_id, body.result, body.actual_cause, body.parts_replaced, body.notes)
        except diagnosis.NotFound:
            return JSONResponse({"error": "Diagnosis not found"}, status_code=404)
        except diagnosis.NotConcluded:
            return JSONResponse({"error": "Feedback can be given once the diagnosis has a conclusion"}, status_code=409)

    @router.get("/outcomes/summary")
    async def outcomes_summary(request: Request) -> dict:
        return await diagnosis.outcome_summary(request.app.state.pool)

    return router

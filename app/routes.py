"""The diagnosis API. Built twice from one definition: with API-key auth (/) and with demo-page auth (/ui-api)."""
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app import diagnosis
from app.manual import complete_manual_dtcs, has_complete_manual
from app.ratelimit import rate_limit
from app.schemas import AnswerIn, OutcomeIn, StartDiagnosis

_MODEL_LINE = re.compile(r"^[A-Z0-9_]{2,32}$")


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

    @router.get("/catalog/{model_line}")
    async def catalog_dtcs(model_line: str, request: Request):
        """DTCs that have both exact-model cases and a complete, non-truncated manual procedure."""
        model_line = model_line.strip().upper()
        if not _MODEL_LINE.match(model_line):
            return JSONResponse({"error": "Invalid model"}, status_code=422)
        complete_dtcs = await complete_manual_dtcs(request.app.state.pool, model_line)
        async with request.app.state.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT dtc_code, count(*) AS n FROM cases WHERE model_line = %s "
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
        async with s.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM cases WHERE dtc_code = %s AND model_line = %s LIMIT 1",
                (body.dtc_code, body.model_line),
            )
            if await cur.fetchone() is None:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND,
                    "This fault code was not found for the selected model",
                )
        if not await has_complete_manual(s.pool, body.dtc_code, body.model_line):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "A complete, non-truncated manual procedure is not available for this model and fault code",
            )
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

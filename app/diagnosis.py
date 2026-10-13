"""Diagnosis service: runs the LangGraph loop and stores its state, steps and costs in Postgres."""
import uuid

import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.casefile import gather_evidence
from app.diagnosis_graph import ANSWERS, MAX_QUESTIONS
from app.llm import LLM
from app.redact import redact
from app.usage import RunInfo, current_run


class NotFound(Exception):
    pass


class NotAsking(Exception):
    """The diagnosis is already concluded (or had no evidence), so it takes no more answers."""


class NotConcluded(Exception):
    """Feedback is only accepted once the diagnosis has reached a conclusion."""


class Conflict(Exception):
    """Another answer for the same question was saved first."""


def _view(diag_id: uuid.UUID, status: str, state: dict) -> dict:
    return {
        "diagnosis_id": str(diag_id),
        "status": status,
        "question": state.get("pending_question") if status == "asking" else None,
        "answer_options": list(ANSWERS) if status == "asking" else [],
        "questions_asked": len(state.get("history", [])),
        "max_questions": MAX_QUESTIONS,
        "candidates": state.get("candidates", []),
        "conclusion": state.get("conclusion"),
    }


def _stored_state(result: dict) -> dict:
    keys = ("candidates", "history", "pending_question", "conclusion")
    return {k: result.get(k) for k in keys} | {"history": result.get("history", [])}


async def run_cost(pool: AsyncConnectionPool, run_id: uuid.UUID) -> dict | None:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT calls, input_tokens, cached_input_tokens, output_tokens, reasoning_tokens, cost_usd, unpriced_calls "
            "FROM run_costs WHERE run_id = %s", (run_id,))
        row = await cur.fetchone()
    if not row:
        return None
    return {"calls": row[0], "input_tokens": row[1], "cached_input_tokens": row[2], "output_tokens": row[3],
            "reasoning_tokens": row[4], "cost_usd": float(row[5]), "unpriced_calls": row[6]}


async def start_diagnosis(pool: AsyncConnectionPool, llm: LLM, graph, dtc: str, model: str) -> dict:
    diag_id = uuid.uuid4()
    token = current_run.set(RunInfo(diag_id, "diagnosis", dtc, model))
    try:
        evidence = await gather_evidence(pool, llm, dtc, model)
        summary = {
            "scope": evidence.scope,
            "total_cases": evidence.stats["total_cases"],
            "top_parts": evidence.stats["top_parts"],
            "manual": {"status": evidence.manual.status, "modules": evidence.manual.module_names,
                       "truncated": evidence.manual.truncated},
        }
        if not evidence.has_evidence:
            state = {"candidates": [], "history": [], "pending_question": None, "conclusion": None}
            status = "no_evidence"
        else:
            result = await graph.ainvoke({"mode": "start", "case_file": evidence.case_file, "candidates": [], "history": []})
            state, status = _stored_state(result), result["status"]

        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO diagnoses (id, dtc_code, model_line, scope, manual_status, case_file, status, state) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (diag_id, dtc, model, evidence.scope, evidence.manual.status, evidence.case_file, status, Jsonb(state)),
            )
    finally:
        current_run.reset(token)
    return {**_view(diag_id, status, state), "evidence": summary, "cost": await run_cost(pool, diag_id)}


async def answer_diagnosis(pool: AsyncConnectionPool, llm: LLM, graph, diag_id: uuid.UUID, answer: str, note: str) -> dict:
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT case_file, status, state FROM diagnoses WHERE id = %s", (diag_id,))
        row = await cur.fetchone()
    if row is None:
        raise NotFound
    case_file, status, state = row
    if status != "asking":
        raise NotAsking

    token = current_run.set(RunInfo(diag_id, "diagnosis"))
    try:
        result = await graph.ainvoke({
            "mode": "answer", "case_file": case_file, "candidates": state["candidates"], "history": state["history"],
            "pending_question": state["pending_question"], "answer": answer, "note": note,
        })
    finally:
        current_run.reset(token)

    new_state, new_status = _stored_state(result), result["status"]
    step_no = len(new_state["history"])
    last = new_state["history"][-1]
    try:
        async with pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO diagnosis_steps (diagnosis_id, step_no, question, answer, note, candidates) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (diag_id, step_no, last["question"], last["answer"], last["note"] or None, Jsonb(new_state["candidates"])),
                )
                await conn.execute(
                    "UPDATE diagnoses SET status = %s, state = %s, updated_at = now() WHERE id = %s",
                    (new_status, Jsonb(new_state), diag_id),
                )
    except psycopg.errors.UniqueViolation:
        raise Conflict from None
    return {**_view(diag_id, new_status, new_state), "cost": await run_cost(pool, diag_id)}


async def get_diagnosis(pool: AsyncConnectionPool, diag_id: uuid.UUID) -> dict:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT dtc_code, model_line, scope, manual_status, status, state, created_at FROM diagnoses WHERE id = %s",
            (diag_id,))
        row = await cur.fetchone()
        if row is None:
            raise NotFound
        cur = await conn.execute(
            "SELECT step_no, question, answer, note, candidates FROM diagnosis_steps WHERE diagnosis_id = %s ORDER BY step_no",
            (diag_id,))
        steps = [{"step": n, "question": q, "answer": a, "note": nt, "candidates_after": c} for n, q, a, nt, c in await cur.fetchall()]
    return {
        **_view(diag_id, row[4], row[5]),
        "dtc_code": row[0], "model_line": row[1], "scope": row[2], "manual_status": row[3],
        "created_at": row[6].isoformat(), "steps": steps, "cost": await run_cost(pool, diag_id),
        "outcome": await get_outcome(pool, diag_id),
    }


async def list_calls(pool: AsyncConnectionPool, diag_id: uuid.UUID) -> dict:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT purpose, model, input_tokens, cached_input_tokens, output_tokens, reasoning_tokens, latency_ms, cost_usd, created_at "
            "FROM llm_calls WHERE run_id = %s ORDER BY id", (diag_id,))
        calls = [
            {"purpose": p, "model": m, "input_tokens": i, "cached_input_tokens": c, "output_tokens": o,
             "reasoning_tokens": r, "latency_ms": ms, "cost_usd": float(cost) if cost is not None else None, "at": at.isoformat()}
            for p, m, i, c, o, r, ms, cost, at in await cur.fetchall()
        ]
    return {"diagnosis_id": str(diag_id), "run_total": await run_cost(pool, diag_id), "calls": calls}


async def get_outcome(pool: AsyncConnectionPool, diag_id: uuid.UUID) -> dict | None:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT result, actual_cause, parts_replaced, notes, updated_at FROM diagnosis_outcomes WHERE diagnosis_id = %s",
            (diag_id,))
        row = await cur.fetchone()
    if not row:
        return None
    return {"result": row[0], "actual_cause": row[1] or "", "parts_replaced": row[2], "notes": row[3] or "", "saved_at": row[4].isoformat()}


async def save_outcome(pool: AsyncConnectionPool, diag_id: uuid.UUID, result: str, actual_cause: str,
                       parts: list[str], notes: str) -> dict:
    """Store the mechanic's feedback (one per diagnosis; sending it again replaces it). Free text is scrubbed of private details."""
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT status FROM diagnoses WHERE id = %s", (diag_id,))
        row = await cur.fetchone()
        if row is None:
            raise NotFound
        if row[0] != "concluded":
            raise NotConcluded
        await conn.execute(
            "INSERT INTO diagnosis_outcomes (diagnosis_id, result, actual_cause, parts_replaced, notes) "
            "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (diagnosis_id) DO UPDATE SET result = EXCLUDED.result, "
            "actual_cause = EXCLUDED.actual_cause, parts_replaced = EXCLUDED.parts_replaced, notes = EXCLUDED.notes, updated_at = now()",
            (diag_id, result, redact(actual_cause) or None, Jsonb([redact(p) for p in parts]), redact(notes) or None),
        )
    return await get_outcome(pool, diag_id)


async def outcome_summary(pool: AsyncConnectionPool) -> dict:
    """How often the diagnosis turned out right, from mechanic feedback."""
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT count(*) FROM diagnoses WHERE status = 'concluded'")
        (concluded,) = await cur.fetchone()
        cur = await conn.execute(
            "SELECT d.model_line, o.result, count(*) FROM diagnosis_outcomes o JOIN diagnoses d ON d.id = o.diagnosis_id "
            "GROUP BY 1, 2 ORDER BY 1, 2")
        rows = await cur.fetchall()
    totals = {"worked": 0, "partly": 0, "not_worked": 0}
    by_model: dict[str, dict] = {}
    for model, result, n in rows:
        totals[result] += n
        by_model.setdefault(model, {"worked": 0, "partly": 0, "not_worked": 0})[result] += n

    def rate(t: dict) -> float | None:
        n = sum(t.values())
        return round(100 * t["worked"] / n, 1) if n else None

    return {"concluded_diagnoses": concluded, "with_feedback": sum(totals.values()), **totals,
            "fixed_rate_pct": rate(totals), "by_model": {m: {**t, "fixed_rate_pct": rate(t)} for m, t in by_model.items()}}

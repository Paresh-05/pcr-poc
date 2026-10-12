"""Past repair cases for a DTC: exact SQL filter on DTC + model, plus stats and complaint groups."""
from dataclasses import dataclass

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.redact import redact

NO_PART = "NOT APPLICABLE"

SCOPE_EXACT = "dtc_and_model"
SCOPE_ALL_MODELS = "dtc_all_models"
SCOPE_NONE = "none"

_WHERE = {
    SCOPE_EXACT: "dtc_code = %(dtc)s AND model_line = %(model)s",
    SCOPE_ALL_MODELS: "dtc_code = %(dtc)s",
}


@dataclass
class Case:
    id: int
    model_line: str
    complaint: str
    diagnosis: str
    failed_parts: list[str]


async def resolve_scope(pool: AsyncConnectionPool, dtc: str, model: str) -> str:
    """Same DTC + model if any cases exist; otherwise the same DTC on all models; otherwise none."""
    async with pool.connection() as conn:
        for scope in (SCOPE_EXACT, SCOPE_ALL_MODELS):
            cur = await conn.execute(f"SELECT 1 FROM cases WHERE {_WHERE[scope]} LIMIT 1", {"dtc": dtc, "model": model})
            if await cur.fetchone():
                return scope
    return SCOPE_NONE


async def load_cases(pool: AsyncConnectionPool, dtc: str, model: str, scope: str) -> list[Case]:
    if scope == SCOPE_NONE:
        return []
    async with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)  # per-cursor: a pooled connection must not keep a custom row factory
        await cur.execute(
            f"SELECT id, model_line, customer_verbatim, diagnosis, failed_parts FROM cases "
            f"WHERE {_WHERE[scope]} ORDER BY id",
            {"dtc": dtc, "model": model},
        )
        return [
            Case(r["id"], r["model_line"], redact(r["customer_verbatim"]), redact(r["diagnosis"]), r["failed_parts"])
            for r in await cur.fetchall()
        ]


async def complaint_groups(pool: AsyncConnectionPool, dtc: str, model: str, scope: str, limit: int = 15) -> list[tuple[str, int]]:
    """Most common distinct customer complaints (case-insensitive) with how many cases reported each."""
    if scope == SCOPE_NONE:
        return []
    async with pool.connection() as conn:
        cur = await conn.execute(
            f"SELECT lower(btrim(customer_verbatim)) AS c, count(*) AS n FROM cases WHERE {_WHERE[scope]} "
            f"GROUP BY 1 ORDER BY n DESC, c LIMIT %(limit)s",
            {"dtc": dtc, "model": model, "limit": limit},
        )
        return [(redact(c)[:160], n) for c, n in await cur.fetchall()]


async def compute_stats(pool: AsyncConnectionPool, dtc: str, model: str, scope: str) -> dict:
    if scope == SCOPE_NONE:
        return {"scope": scope, "total_cases": 0, "no_part_replaced_pct": 0, "top_parts": []}
    where, params = _WHERE[scope], {"dtc": dtc, "model": model, "no_part": NO_PART}
    async with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        await cur.execute(f"SELECT count(*) AS n FROM cases WHERE {where}", params)
        total = (await cur.fetchone())["n"]
        await cur.execute(f"SELECT count(*) AS n FROM cases WHERE {where} AND failed_parts ? %(no_part)s", params)
        no_part = (await cur.fetchone())["n"]
        await cur.execute(
            f"SELECT part, count(*) AS cases FROM cases, LATERAL jsonb_array_elements_text(failed_parts) AS part "
            f"WHERE {where} AND part <> %(no_part)s GROUP BY part ORDER BY cases DESC, part LIMIT 8",
            params,
        )
        parts = await cur.fetchall()
    return {
        "scope": scope,
        "total_cases": total,
        "no_part_replaced_pct": round(100 * no_part / total),
        "top_parts": [{"part": p["part"], "cases": p["cases"], "pct": round(100 * p["cases"] / total)} for p in parts],
    }

"""SQL aggregation over all matching cases: what usually fixed this DTC."""
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

NO_PART = "NOT APPLICABLE"

_TOTAL = "SELECT count(*) AS n FROM cases WHERE {where}"
_PARTS = """
    SELECT part, count(*) AS cases
    FROM cases, LATERAL jsonb_array_elements_text(failed_parts) AS part
    WHERE {where} AND part <> %(no_part)s
    GROUP BY part ORDER BY cases DESC, part LIMIT 5
"""
_NO_PART = "SELECT count(*) AS n FROM cases WHERE {where} AND failed_parts ? %(no_part)s"


async def compute_stats(pool: AsyncConnectionPool, dtc: str, model: str) -> dict:
    """Stats for same DTC+model; falls back to same DTC (all models) if there are no exact cases."""
    params = {"dtc": dtc, "model": model, "no_part": NO_PART}
    async with pool.connection() as conn:
        conn.row_factory = dict_row
        for scope, where in (
            ("dtc_and_model", "dtc_code = %(dtc)s AND model_line = %(model)s"),
            ("dtc_all_models", "dtc_code = %(dtc)s"),
        ):
            total = (await (await conn.execute(_TOTAL.format(where=where), params)).fetchone())["n"]
            if total == 0:
                continue
            no_part = (await (await conn.execute(_NO_PART.format(where=where), params)).fetchone())["n"]
            parts = await (await conn.execute(_PARTS.format(where=where), params)).fetchall()
            return {
                "scope": scope,
                "total_cases": total,
                "no_part_replaced_pct": round(100 * no_part / total),
                "top_parts": [
                    {"part": p["part"], "cases": p["cases"], "pct": round(100 * p["cases"] / total)}
                    for p in parts
                ],
            }
    return {"scope": "none", "total_cases": 0, "no_part_replaced_pct": 0, "top_parts": []}

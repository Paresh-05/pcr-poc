"""Case retrieval: exact SQL filter on DTC/model, then keyword (BM25-style) and optional vector ranking."""
import re
from dataclasses import dataclass

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.redact import redact

MIN_CASES = 5
EXAMPLE_LIMIT = 8
MAX_DIAGNOSIS_CHARS = 1500
RRF_K = 60  # reciprocal rank fusion constant
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class Case:
    id: int
    dtc_code: str
    model_line: str
    complaint: str
    diagnosis: str
    failed_parts: list[str]
    tier: int
    score: float


def build_tsquery(complaint: str) -> str:
    """OR-query from alphanumeric tokens only, so no tsquery syntax can be injected."""
    tokens = list(dict.fromkeys(_TOKEN.findall(complaint.lower())))[:20]
    return " | ".join(tokens)


_COLUMNS = "id, dtc_code, model_line, customer_verbatim, diagnosis, failed_parts"

# Keyword only.
_SELECT_KEYWORD = f"""
    SELECT {_COLUMNS}, {{bm25}} AS score
    FROM cases
    WHERE {{where}} AND NOT (id = ANY(%(excl)s))
    ORDER BY score DESC, id
    LIMIT %(limit)s
"""

# Hybrid: rank by keyword and by vector distance separately, then fuse with reciprocal rank fusion.
_SELECT_HYBRID = f"""
    WITH scored AS (
        SELECT {_COLUMNS}, {{bm25}} AS bm25,
               complaint_embedding <=> %(vec)s::vector AS dist
        FROM cases
        WHERE {{where}} AND NOT (id = ANY(%(excl)s))
    ), ranked AS (
        SELECT *,
               rank() OVER (ORDER BY bm25 DESC) AS r_bm25,
               rank() OVER (ORDER BY dist ASC NULLS LAST) AS r_vec
        FROM scored
    )
    SELECT {_COLUMNS},
           (CASE WHEN bm25 > 0 THEN 1.0 / ({RRF_K} + r_bm25) ELSE 0 END
            + CASE WHEN dist IS NOT NULL THEN 1.0 / ({RRF_K} + r_vec) ELSE 0 END) AS score
    FROM ranked
    ORDER BY score DESC, id
    LIMIT %(limit)s
"""


def _to_case(row: dict, tier: int) -> Case:
    return Case(
        id=row["id"],
        dtc_code=row["dtc_code"],
        model_line=row["model_line"],
        complaint=redact(row["customer_verbatim"]),
        diagnosis=redact(row["diagnosis"])[:MAX_DIAGNOSIS_CHARS],
        failed_parts=row["failed_parts"],
        tier=tier,
        score=float(row["score"]),
    )


async def retrieve_cases(
    pool: AsyncConnectionPool,
    dtc: str,
    model: str,
    complaint: str,
    query_vec: str | None = None,
    max_distance: float = 0.5,
) -> list[Case]:
    """Tier 1: same DTC + model. Tier 2: same DTC, other models. Tier 3: similar complaint, any DTC.

    With `query_vec` (a pgvector literal) ranking is hybrid; without it, keyword only.
    """
    tsq = build_tsquery(complaint)
    hybrid = query_vec is not None
    bm25 = "ts_rank_cd(complaint_tsv, to_tsquery('english', %(q)s))" if tsq else "0::float"
    template = _SELECT_HYBRID if hybrid else _SELECT_KEYWORD
    params = {"dtc": dtc, "model": model, "q": tsq, "vec": query_vec, "max_dist": max_distance}

    tier3_matches = []
    if tsq:
        tier3_matches.append("complaint_tsv @@ to_tsquery('english', %(q)s)")
    if hybrid:
        tier3_matches.append("complaint_embedding <=> %(vec)s::vector < %(max_dist)s")
    tier3_where = "dtc_code <> %(dtc)s AND (" + " OR ".join(tier3_matches) + ")" if tier3_matches else None

    found: list[Case] = []
    async with pool.connection() as conn:
        conn.row_factory = dict_row

        async def run(where: str, tier: int, limit: int) -> None:
            sql = template.format(bm25=bm25, where=where)
            args = {**params, "limit": limit, "excl": [c.id for c in found]}
            cur = await conn.execute(sql, args)
            found.extend(_to_case(r, tier) for r in await cur.fetchall())

        await run("dtc_code = %(dtc)s AND model_line = %(model)s", 1, EXAMPLE_LIMIT)
        if len(found) < MIN_CASES:
            await run("dtc_code = %(dtc)s AND model_line <> %(model)s", 2, EXAMPLE_LIMIT - len(found))
        if len(found) < MIN_CASES and tier3_where:
            await run(tier3_where, 3, EXAMPLE_LIMIT - len(found))
    return found

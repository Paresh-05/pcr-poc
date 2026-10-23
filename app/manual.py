"""Which DTCs have a manual procedure on file, for the catalog pickers."""
from psycopg_pool import AsyncConnectionPool


async def has_complete_manual(pool: AsyncConnectionPool, dtc_code: str, model_line: str) -> bool:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT 1 FROM manual_chunks WHERE %s = ANY(dtc_codes) AND model_line = %s LIMIT 1",
            (dtc_code, model_line),
        )
        return await cur.fetchone() is not None


async def complete_manual_dtcs(pool: AsyncConnectionPool, model_line: str) -> set[str]:
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT DISTINCT unnest(dtc_codes) FROM manual_chunks WHERE model_line = %s", (model_line,))
        return {row[0] for row in await cur.fetchall()}

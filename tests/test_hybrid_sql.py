"""Exercises the hybrid SQL in a throwaway temp table (nothing is written to the real `cases` table)."""
import asyncio
import contextlib
import os
import random
from pathlib import Path

import psycopg
import pytest

from app.config import Settings
from app.embeddings import to_vector_literal
from app.retrieval import retrieve_cases

class _OneConnPool:
    """Minimal stand-in for AsyncConnectionPool that always hands out one connection."""

    def __init__(self, conn):
        self.conn = conn

    @contextlib.asynccontextmanager
    async def connection(self):
        yield self.conn


def _vec(seed: int, dim: int) -> list[float]:
    rnd = random.Random(seed)
    return [rnd.uniform(-1, 1) for _ in range(dim)]


def _settings() -> Settings | None:
    if not Path(".env").exists():
        return None
    return Settings()


async def _scenario(settings: Settings) -> None:
    async with await psycopg.AsyncConnection.connect(settings.db_conninfo) as conn:
        try:
            await conn.execute("CREATE TEMP TABLE cases AS SELECT * FROM public.cases WHERE dtc_code = 'P200200' OR id < 400")
        except psycopg.Error:
            pytest.skip("database not available")
        pool = _OneConnPool(conn)
        cur = await conn.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'public.cases'::regclass AND attname = 'complaint_embedding'"
        )
        (dim,) = await cur.fetchone()

        # Give every P200200 case a random vector; make one case's vector the "query meaning".
        cur = await conn.execute("SELECT id FROM cases WHERE dtc_code = 'P200200' AND model_line = 'HECTOR' ORDER BY id")
        ids = [r[0] for r in await cur.fetchall()]
        for cid in ids:
            await conn.execute("UPDATE cases SET complaint_embedding = %s::vector WHERE id = %s",
                               (to_vector_literal(_vec(cid, dim)), cid))
        target = ids[-1]
        qvec = to_vector_literal(_vec(target, dim))

        # Complaint shares no words with any case, so keyword score is 0 for all; only vectors can find it.
        hybrid = await retrieve_cases(pool, "P200200", "HECTOR", "zzzz qqqq", query_vec=qvec)
        assert hybrid and hybrid[0].id == target and hybrid[0].tier == 1

        keyword = await retrieve_cases(pool, "P200200", "HECTOR", "zzzz qqqq", query_vec=None)
        assert keyword and keyword[0].id != target  # keyword-only cannot know which case "means" the same


def test_hybrid_ranking_uses_vectors():
    settings = _settings()
    if settings is None:
        pytest.skip("no .env")
    asyncio.run(_scenario(settings), loop_factory=asyncio.SelectorEventLoop)

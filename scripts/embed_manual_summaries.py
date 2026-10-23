"""Step 2: embed every manual_chunk_summaries row that doesn't have an embedding yet."""
import asyncio
import logging
import sys
from pathlib import Path

from pgvector import Vector
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db import create_pool
from app.llm import OpenAILLM
from app.usage import UsageRecorder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("embed_manual_summaries")


async def main() -> None:
    settings = get_settings()
    pool = create_pool(settings)
    await pool.open()
    llm = OpenAILLM(settings, UsageRecorder(pool))

    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("SELECT id, summary_for_search FROM manual_chunk_summaries WHERE embedding IS NULL")
        rows = await cur.fetchall()

    log.info(f"{len(rows)} summaries to embed")
    for i, row in enumerate(rows, 1):
        vec = Vector(await llm.embed(row["summary_for_search"]))
        async with pool.connection() as conn:
            await conn.execute("UPDATE manual_chunk_summaries SET embedding = %s WHERE id = %s", (vec, row["id"]))
        if i % 50 == 0:
            log.info(f"...embedded {i}/{len(rows)}")

    log.info(f"done: {len(rows)} embedded")
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)

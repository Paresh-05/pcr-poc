"""Ingest DTC manual procedures from a JSON file into manual_chunks, with embeddings.

Expected JSON: a list of objects, one per chunk (one chunk per dtc_code + submodule; a chunk whose
procedure is identical across codes may list more than one code in dtc_codes):

  [
    {
      "dtc_codes": ["C100016", "C100017"],
      "model_line": "HECTOR",
      "submodule": "Anti-lock Brake System",
      "description": "DTC C100016: Battery Voltage Low ... System Description: ...",
      "inspection_steps": "1. Clear the malfunction codes ... 2. Check if other DTCs ..."
    },
    ...
  ]

Usage: uv run python scripts/ingest_manual.py path/to/manual_chunks.json
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from pgvector import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db import create_pool
from app.llm import OpenAILLM
from app.usage import UsageRecorder

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ingest_manual")


async def main(json_path: Path) -> None:
    settings = get_settings()
    pool = create_pool(settings)
    await pool.open()
    llm = OpenAILLM(settings, UsageRecorder(pool))

    chunks = json.loads(json_path.read_text(encoding="utf-8"))
    inserted = 0
    for c in chunks:
        dtc_codes = [d.strip().upper() for d in c["dtc_codes"]]
        model_line = c["model_line"].strip().upper()
        description = c["description"].strip()
        inspection_steps = c["inspection_steps"].strip()
        embed_text = f"{description}\n{inspection_steps}"
        embedding = Vector(await llm.embed(embed_text))

        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO manual_chunks (dtc_codes, model_line, submodule, description, inspection_steps, embedding) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (dtc_codes, model_line, c.get("submodule"), description, inspection_steps, embedding),
            )
        inserted += 1
        if inserted % 20 == 0:
            log.info("ingested %d manual chunks so far...", inserted)

    await pool.close()
    log.info("done: %d manual chunks inserted", inserted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    args = parser.parse_args()
    # Windows needs a selector event loop for psycopg async (see run.py).
    asyncio.run(main(args.json_path), loop_factory=asyncio.SelectorEventLoop)

"""Load the 368 generated manual markdown files into manual_chunks ONLY - no summary generation, no
embedding, no LLM calls at all. Pure local-file-to-DB insert, with the correct model_line from
toc_sections.json. Skips sections already present (resume-safe).

Usage: uv run python scripts/load_manual_markdown_only.py [--limit N]
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db import create_pool
from build_manual_markdown import section_filename  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("load_manual_markdown_only")

SECTIONS_PATH = Path(__file__).resolve().parent / "sample_data" / "toc_sections.json"
MARKDOWN_DIR = Path(__file__).resolve().parent / "sample_data" / "manual_markdown"


async def main(limit: int | None) -> None:
    settings = get_settings()
    pool = create_pool(settings)
    await pool.open()

    sections = json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))
    if limit:
        sections = sections[:limit]
    total = len(sections)
    log.info(f"{total} sections to check (markdown only, no summaries, no embeddings)")

    inserted = skipped = missing_file = 0
    for i, section in enumerate(sections, 1):
        name = section_filename(section["codes"], section["submodule"], section["start_page"])
        md_path = MARKDOWN_DIR / name
        if not md_path.exists():
            log.warning(f"[{i}/{total}] SKIP {name}: markdown file not found")
            missing_file += 1
            continue

        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM manual_chunks WHERE model_line = %s AND submodule = %s AND start_page = %s LIMIT 1",
                (section["model_line"], section["submodule"], section["start_page"]),
            )
            if await cur.fetchone() is not None:
                skipped += 1
                continue

            full_markdown = md_path.read_text(encoding="utf-8")
            await conn.execute(
                "INSERT INTO manual_chunks (dtc_codes, model_line, submodule, start_page, end_page, full_markdown) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (section["codes"], section["model_line"], section["submodule"],
                 section["start_page"], section["end_page"], full_markdown),
            )
            inserted += 1
            if inserted % 50 == 0:
                log.info(f"[{i}/{total}] ...inserted {inserted} so far")

    log.info(f"done: {inserted} inserted, {skipped} already present, {missing_file} missing markdown file")
    await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main(args.limit), loop_factory=asyncio.SelectorEventLoop)
